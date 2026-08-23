//! Live-session building blocks for `fw robot listen` (bd-rt-listen-epic-polh).
//!
//! This module hosts the real-time driver's audio-side state. First resident:
//! [`SessionBuffer`] (bd-rt-buffer-a6l5) — the bounded rolling buffer that
//! turns "re-decode a growing file" (O(n²) over a session) into "re-decode a
//! bounded window" (O(1) per step). The driver loop, emission policies, and
//! VAD land here in their own beads (bd-rt-listen-cmd-i48i and friends).
//!
//! Design (locked in the bead):
//! - Bounded rolling buffer (default cap 12 s), trimmed at committed word
//!   boundaries minus a keep-back margin (default 200 ms, mirroring
//!   whisper.cpp `--keep`) so a word is never clipped at the seam.
//! - Linguistic context crosses trims as decoder PROMPT text, not audio:
//!   within-utterance committed text, plus the tail of the previous
//!   utterance (~200 chars, word-boundary trimmed) seeding the next
//!   utterance (`--no-context` disables). String prompt is the v1 mechanism,
//!   matching whisper_streaming's proven approach.
//! - Exact session clock: buffer index -> absolute session time via a
//!   trimmed-samples counter in SAMPLES (never floats), so arbitrary
//!   push/trim sequences keep the mapping consistent. The resampler's
//!   sub-5 ms group delay is deliberately ignored (far under the 20 ms
//!   mel frame; bd-rt-resampler-pbk9 note).
//! - Silence retention: trims always keep at least `min_tail_sec` of audio
//!   so a new speech onset after long silence still gets its full VAD
//!   pre-pad (bd-rt-buffer polish item 2).
//! - Allocation-stable: front-trims use `Vec::drain`, which memmoves in
//!   place and never shrinks capacity — after warmup the buffer stops
//!   allocating (asserted by test).
//!
//! Why not seek-resume in the engine instead: cross-attention K/V is
//! invalidated whenever encoder input grows, so carrying decoder state
//! across ticks buys little for a large engine change (investigation
//! conclusion recorded in the bead). Bounded buffer + prompt carry is the
//! published pattern (whisper_streaming, SimulStreaming `--audio_max_len`,
//! whisper.cpp stream `--length`/`--keep`).

use crate::native_engine::mel::SAMPLE_RATE;

/// Configuration for [`SessionBuffer`]. Defaults match the CLI defaults
/// consolidated in the driver bead (`--max-buffer-sec 12`, keep-back 200 ms,
/// `--no-context` => `prompt_carry: false`).
#[derive(Debug, Clone)]
pub struct SessionBufferConfig {
    /// Hard cap on buffered audio; beyond it [`SessionBuffer::enforce_cap`]
    /// force-trims from the front (degraded, surfaced as a warning).
    pub max_buffer_sec: f64,
    /// Audio retained BEHIND the committed watermark on a trim so the next
    /// decode still hears the seam word.
    pub keep_back_sec: f64,
    /// Minimum audio always retained (VAD pre-pad lookback), even through
    /// long silence.
    pub min_tail_sec: f64,
    /// Whether committed text carries across trims/utterances as decoder
    /// prompt (`--no-context` sets false).
    pub prompt_carry: bool,
    /// Front-truncation cap for the assembled prompt, in characters. The
    /// driver sizes this to the engine's max_prompt_ctx semantics; the
    /// default is conservative.
    pub prompt_cap_chars: usize,
    /// How much of the PREVIOUS utterance's committed text seeds the next
    /// utterance's prompt (word-boundary trimmed).
    pub cross_utterance_tail_chars: usize,
}

impl Default for SessionBufferConfig {
    fn default() -> Self {
        Self {
            max_buffer_sec: 12.0,
            keep_back_sec: 0.2,
            min_tail_sec: 0.2,
            prompt_carry: true,
            prompt_cap_chars: 896,
            cross_utterance_tail_chars: 200,
        }
    }
}

/// Report of a forced front-trim under the buffer cap (pathological
/// continuous speech the policy could not commit). The driver surfaces this
/// as `listen.warning {reason: "forced_trim"}` and counts it in
/// `session_stats`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForcedTrim {
    /// Audio dropped from the front, in seconds.
    pub dropped_sec: f64,
}

/// Bounded rolling 16 kHz mono session buffer with an exact session clock,
/// commit-boundary trimming, and prompt carry. See the module docs for the
/// design rationale.
#[derive(Debug)]
pub struct SessionBuffer {
    cfg: SessionBufferConfig,
    /// Buffered samples (16 kHz mono f32).
    samples: Vec<f32>,
    /// Samples dropped from the front since session start. Buffer index i
    /// corresponds to absolute session sample `trimmed + i`.
    trimmed: u64,
    /// Absolute session sample up to which text is committed.
    committed_through: u64,
    /// Committed text of the CURRENT utterance (grows by deltas).
    current_utterance_text: String,
    /// Word-boundary tail of the previous utterance (cross-utterance seed).
    prev_utterance_tail: Option<String>,
    /// Largest buffered length observed (for `session_stats`).
    high_water_samples: usize,
    /// Total forced-trim events (for `session_stats`).
    forced_trims: u64,
}

impl SessionBuffer {
    #[must_use]
    pub fn new(cfg: SessionBufferConfig) -> Self {
        let cap_samples = seconds_to_samples(cfg.max_buffer_sec);
        Self {
            cfg,
            // Reserve the cap up front: after this, steady-state operation
            // performs no allocations on the audio path.
            samples: Vec::with_capacity(cap_samples + SAMPLE_RATE),
            trimmed: 0,
            committed_through: 0,
            current_utterance_text: String::new(),
            prev_utterance_tail: None,
            high_water_samples: 0,
            forced_trims: 0,
        }
    }

    // -- audio ------------------------------------------------------------

    /// Append capture audio (post-resample, 16 kHz mono).
    pub fn push(&mut self, chunk: &[f32]) {
        self.samples.extend_from_slice(chunk);
        self.high_water_samples = self.high_water_samples.max(self.samples.len());
    }

    /// The current decode window (the whole buffer).
    #[must_use]
    pub fn window(&self) -> &[f32] {
        &self.samples
    }

    /// The decode window starting at an absolute session time (clamped to
    /// the buffered range). The driver slices each utterance's decodes to
    /// the utterance span so the keep-back / min-tail audio retained from
    /// the PREVIOUS utterance is never re-transcribed — the first live
    /// session showed exactly that seam duplication (text from utterance
    /// N-1 reappearing in N's transcript) when decoding the whole buffer.
    #[must_use]
    pub fn window_from(&self, session_sec: f64) -> &[f32] {
        let absolute = seconds_to_samples(session_sec.max(0.0)) as u64;
        let start = absolute.saturating_sub(self.trimmed) as usize;
        &self.samples[start.min(self.samples.len())..]
    }

    /// Buffered audio duration in seconds.
    #[must_use]
    pub fn buffer_sec(&self) -> f64 {
        samples_to_seconds(self.samples.len() as u64)
    }

    /// Session time of the START of the buffer (how much audio has been
    /// trimmed since session start).
    #[must_use]
    pub fn session_offset_sec(&self) -> f64 {
        samples_to_seconds(self.trimmed)
    }

    /// Total audio pushed since session start, in seconds.
    #[must_use]
    pub fn session_duration_sec(&self) -> f64 {
        samples_to_seconds(self.trimmed + self.samples.len() as u64)
    }

    /// Map a buffer-relative time (e.g. a segment timestamp from decoding
    /// [`Self::window`]) to absolute session time.
    #[must_use]
    pub fn session_time_of(&self, window_relative_sec: f64) -> f64 {
        self.session_offset_sec() + window_relative_sec
    }

    /// Largest buffered length observed, in seconds (for `session_stats`).
    #[must_use]
    pub fn high_water_sec(&self) -> f64 {
        samples_to_seconds(self.high_water_samples as u64)
    }

    /// Count of forced trims (for `session_stats`).
    #[must_use]
    pub fn forced_trims(&self) -> u64 {
        self.forced_trims
    }

    // -- committed watermark + trimming ------------------------------------

    /// Advance the committed watermark to an absolute session time (from the
    /// emission policy's last committed word boundary). Monotonic: earlier
    /// values are ignored; values beyond buffered audio are clamped.
    pub fn set_committed_through(&mut self, session_sec: f64) {
        let sample = seconds_to_samples(session_sec.max(0.0)) as u64;
        let end = self.trimmed + self.samples.len() as u64;
        self.committed_through = self.committed_through.max(sample.min(end));
    }

    /// Trim audio before the committed watermark minus the keep-back margin.
    /// Call ONLY between decodes (never mid-decode). Respects the minimum
    /// tail retention. Returns seconds trimmed.
    pub fn trim_to_committed(&mut self) -> f64 {
        let keep_back = seconds_to_samples(self.cfg.keep_back_sec) as u64;
        let target_start = self.committed_through.saturating_sub(keep_back);
        self.trim_front_to(target_start)
    }

    /// Enforce the buffer cap by force-trimming from the front when the
    /// policy cannot commit (pathological continuous speech). Returns the
    /// forced-trim report when a trim happened; the driver must surface it.
    pub fn enforce_cap(&mut self) -> Option<ForcedTrim> {
        let cap = seconds_to_samples(self.cfg.max_buffer_sec);
        if self.samples.len() <= cap {
            return None;
        }
        let excess = (self.samples.len() - cap) as u64;
        let target_start = self.trimmed + excess;
        let dropped = self.trim_front_to(target_start);
        if dropped > 0.0 {
            self.forced_trims += 1;
            // Audio ahead of the watermark was dropped: move the watermark up
            // so later trims/mappings stay consistent (the text for this
            // audio comes from the last partial — degraded but honest).
            self.committed_through = self.committed_through.max(self.trimmed);
            Some(ForcedTrim {
                dropped_sec: dropped,
            })
        } else {
            None
        }
    }

    /// Trim the buffer front so it starts at absolute sample
    /// `target_start`, clamped by the minimum-tail retention. Returns
    /// seconds trimmed.
    fn trim_front_to(&mut self, target_start: u64) -> f64 {
        let min_tail = seconds_to_samples(self.cfg.min_tail_sec);
        let end = self.trimmed + self.samples.len() as u64;
        // Never trim into the minimum retained tail.
        let max_start = end.saturating_sub(min_tail as u64);
        let new_start = target_start.min(max_start);
        if new_start <= self.trimmed {
            return 0.0;
        }
        let drop = (new_start - self.trimmed) as usize;
        self.samples.drain(..drop); // memmove; capacity preserved
        self.trimmed = new_start;
        samples_to_seconds(drop as u64)
    }

    // -- prompt carry -------------------------------------------------------

    /// Record committed text (a delta) for the current utterance.
    pub fn append_committed_text(&mut self, text: &str) {
        if self.cfg.prompt_carry && !text.is_empty() {
            if !self.current_utterance_text.is_empty()
                && !self.current_utterance_text.ends_with(' ')
            {
                self.current_utterance_text.push(' ');
            }
            self.current_utterance_text.push_str(text);
        }
    }

    /// Close the current utterance: its committed tail becomes the seed for
    /// the next utterance's prompt (cross-utterance carry, bd-rt-buffer
    /// polish item 1).
    pub fn end_utterance(&mut self) {
        if !self.cfg.prompt_carry {
            self.current_utterance_text.clear();
            return;
        }
        let text = std::mem::take(&mut self.current_utterance_text);
        let tail = word_boundary_tail(&text, self.cfg.cross_utterance_tail_chars);
        if !tail.is_empty() {
            self.prev_utterance_tail = Some(tail.to_owned());
        }
    }

    /// The decoder prompt for the next step decode: previous-utterance tail
    /// plus the current utterance's committed text, front-truncated to the
    /// configured cap (oldest text drops first). `None` when carry is
    /// disabled or there is nothing to carry.
    #[must_use]
    pub fn prompt(&self) -> Option<String> {
        if !self.cfg.prompt_carry {
            return None;
        }
        let mut assembled = String::new();
        if let Some(tail) = &self.prev_utterance_tail {
            assembled.push_str(tail);
            if !self.current_utterance_text.is_empty() && !assembled.ends_with(' ') {
                assembled.push(' ');
            }
        }
        assembled.push_str(&self.current_utterance_text);
        if assembled.trim().is_empty() {
            return None;
        }
        Some(front_truncate_at_word(&assembled, self.cfg.prompt_cap_chars).to_owned())
    }
}

// -- helpers ----------------------------------------------------------------

fn seconds_to_samples(sec: f64) -> usize {
    (sec * SAMPLE_RATE as f64).round() as usize
}

fn samples_to_seconds(samples: u64) -> f64 {
    samples as f64 / SAMPLE_RATE as f64
}

/// The last at-most-`cap` characters of `text`, starting at a word boundary
/// where possible.
fn word_boundary_tail(text: &str, cap: usize) -> &str {
    let trimmed = text.trim();
    if trimmed.chars().count() <= cap {
        return trimmed;
    }
    // Find the byte index `cap` characters from the end.
    let start_char = trimmed.chars().count() - cap;
    let mut byte = 0;
    for (count, (idx, _)) in trimmed.char_indices().enumerate() {
        if count == start_char {
            byte = idx;
            break;
        }
    }
    let slice = &trimmed[byte..];
    // Advance to the next word boundary so we never start mid-word.
    match slice.find(' ') {
        Some(space) => slice[space..].trim_start(),
        None => slice,
    }
}

/// Front-truncate to at most `cap` characters, preferring a word boundary.
fn front_truncate_at_word(text: &str, cap: usize) -> &str {
    word_boundary_tail(text, cap)
}

// ---------------------------------------------------------------------------
// bd-rt-vad-stream-ulp5: StreamingVad — causal VAD for the live driver
//
// The batch VAD derives its activity threshold from GLOBAL whole-file RMS
// statistics (orchestrator vad_energy_detect_with_analysis) — impossible for
// a stream. This is the causal replacement: a per-frame adaptive energy
// pre-gate (running noise floor + relative dB gate) feeding a
// neural second tier ([`VoiceClassifier`]; the earshot tier was EVALUATED
// and REJECTED — see the ignored `earshot_eval_*` tests and the bead close
// comment: it passes loud harmonic music at every usable threshold).
// Energy-only is the shipped v1; the trait seam stays for future
// classifiers.
//
// TIME BASES (bd-rt-vad-stream polish item 2): 20 ms frames = 320 samples
// @16 kHz = one encoder frame (2 mel hops). This module is the single
// authority for VAD frame math; AlignAtt frames and mel frames land on the
// same 20 ms grid.
// ---------------------------------------------------------------------------

/// VAD frame length: 20 ms at 16 kHz — aligned with encoder frames (2 mel
/// hops of 160 samples). Single authority for the 20 ms grid.
pub const VAD_FRAME_SAMPLES: usize = SAMPLE_RATE / 50;

/// Convert a VAD frame index to seconds on the session clock.
#[must_use]
pub fn vad_frame_to_sec(frame: u64) -> f64 {
    frame as f64 * VAD_FRAME_SAMPLES as f64 / SAMPLE_RATE as f64
}

/// Optional second-tier voice classifier (the earshot seam). Runs only on
/// frames that pass the energy pre-gate; returning `false` vetoes them.
pub trait VoiceClassifier: Send {
    fn is_voice(&mut self, frame: &[f32]) -> bool;
}

/// Configuration for [`StreamingVad`]. Defaults match the driver CLI
/// defaults (`--vad-min-speech-ms 250`, `--vad-endpoint-ms 600`,
/// `--vad-gate-db 9` — renamed from --vad-threshold to avoid the semantic
/// collision with transcribe's flag).
#[derive(Debug, Clone)]
pub struct StreamingVadConfig {
    /// Sustained voice required before an utterance opens.
    pub min_speech_ms: u64,
    /// Sustained silence that closes an utterance.
    pub endpoint_ms: u64,
    /// Energy gate: frame must exceed the running noise floor by this many
    /// dB to count as voiced.
    pub gate_db: f64,
    /// Pre-onset audio attributed to the utterance (ring lookback).
    pub pre_pad_ms: u64,
    /// Absolute floor: frames below this dBFS never count as voiced
    /// (guards digital silence from gate-relative false positives).
    pub min_voice_dbfs: f64,
    /// Noise-floor upward drift while NOT in speech (fast re-adaptation to
    /// louder environments), dB per second.
    pub floor_rise_db_per_sec_silence: f64,
    /// Noise-floor upward drift while IN speech, dB per second. Default
    /// equals the silence rate: real speech is protected by the INSTANT
    /// downward floor reset on inter-word dips (which re-anchor the floor
    /// every few hundred ms), while steady tones (fans, hums) that opened a
    /// false utterance get reclassified as floor within ~2-3 s and the
    /// endpoint machinery closes it — the desired discriminator. Lower this
    /// only for sustained-tone speech (singing), at the cost of slower
    /// hum recovery.
    pub floor_rise_db_per_sec_speech: f64,
    /// Consecutive unvoiced frames tolerated inside a voiced run before the
    /// run resets (grace for glottal gaps).
    pub voiced_run_grace_frames: u32,
}

impl Default for StreamingVadConfig {
    fn default() -> Self {
        Self {
            min_speech_ms: 250,
            endpoint_ms: 600,
            gate_db: 9.0,
            pre_pad_ms: 200,
            min_voice_dbfs: -55.0,
            floor_rise_db_per_sec_silence: 10.0,
            floor_rise_db_per_sec_speech: 10.0,
            voiced_run_grace_frames: 2,
        }
    }
}

/// An edge event from the VAD state machine, on the session clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadEdge {
    /// Sustained speech confirmed. `t_sec` is the utterance start: first
    /// voiced frame minus pre-pad (clamped to the previous endpoint).
    SpeechStarted { t_sec: f64 },
    /// Sustained silence after speech. `t_sec` is the end of the last
    /// voiced frame.
    Endpoint { t_sec: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadState {
    Silence,
    Speech,
}

/// Causal streaming VAD: push 16 kHz mono samples, receive edge events.
/// Costs a handful of arithmetic ops per sample (<< 1 ms per driver step).
pub struct StreamingVad {
    cfg: StreamingVadConfig,
    classifier: Option<Box<dyn VoiceClassifier>>,
    /// Partial frame carried across pushes.
    partial: Vec<f32>,
    /// Frames consumed since session start.
    frames: u64,
    state: VadState,
    /// Running noise floor estimate, dBFS.
    noise_floor_db: f64,
    /// Current voiced run: (start_frame, voiced_frames, grace_left).
    voiced_run: Option<(u64, u32, u32)>,
    /// Unvoiced frames since the last voiced frame (in Speech state).
    unvoiced_run: u32,
    /// Frame AFTER the last voiced frame (end of voiced audio).
    last_voiced_end: u64,
    /// Session time before which a new utterance's pre-pad may not reach
    /// (the previous endpoint).
    floor_time_sec: f64,
    /// Frames rejected by the second-tier classifier (for stats/eval).
    classifier_vetoes: u64,
}

impl std::fmt::Debug for StreamingVad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingVad")
            .field("state", &self.state)
            .field("frames", &self.frames)
            .field("noise_floor_db", &self.noise_floor_db)
            .finish_non_exhaustive()
    }
}

impl StreamingVad {
    #[must_use]
    pub fn new(cfg: StreamingVadConfig) -> Self {
        Self {
            cfg,
            classifier: None,
            partial: Vec::with_capacity(VAD_FRAME_SAMPLES),
            frames: 0,
            state: VadState::Silence,
            noise_floor_db: -70.0,
            voiced_run: None,
            unvoiced_run: 0,
            last_voiced_end: 0,
            floor_time_sec: 0.0,
            classifier_vetoes: 0,
        }
    }

    /// Attach the optional second-tier classifier (earshot seam).
    pub fn with_classifier(mut self, classifier: Box<dyn VoiceClassifier>) -> Self {
        self.classifier = Some(classifier);
        self
    }

    /// Whether the machine currently considers speech open.
    #[must_use]
    pub fn in_speech(&self) -> bool {
        self.state == VadState::Speech
    }

    /// Current noise-floor estimate (dBFS), for logging/diagnostics.
    #[must_use]
    pub fn noise_floor_db(&self) -> f64 {
        self.noise_floor_db
    }

    /// Frames vetoed by the second-tier classifier (for stats).
    #[must_use]
    pub fn classifier_vetoes(&self) -> u64 {
        self.classifier_vetoes
    }

    /// Push new session audio (16 kHz mono, any chunk size); returns edge
    /// events in order. Chunk-size invariant: identical audio in different
    /// split points yields identical edges.
    pub fn push(&mut self, samples: &[f32]) -> Vec<VadEdge> {
        let mut edges = Vec::new();
        let mut rest = samples;
        // Complete a carried partial frame first.
        if !self.partial.is_empty() {
            let need = VAD_FRAME_SAMPLES - self.partial.len();
            let take = need.min(rest.len());
            self.partial.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.partial.len() == VAD_FRAME_SAMPLES {
                let frame = std::mem::take(&mut self.partial);
                self.step_frame(&frame, &mut edges);
            }
        }
        let whole = rest.len() - rest.len() % VAD_FRAME_SAMPLES;
        for frame in rest[..whole].as_chunks::<VAD_FRAME_SAMPLES>().0 {
            self.step_frame(frame, &mut edges);
        }
        self.partial.extend_from_slice(&rest[whole..]);
        edges
    }

    fn step_frame(&mut self, frame: &[f32], edges: &mut Vec<VadEdge>) {
        let frame_index = self.frames;
        self.frames += 1;

        let rms_db = frame_dbfs(frame);
        // Noise floor: instant drop to quieter frames; bounded upward drift
        // whose rate depends on state (fast in silence, slow in speech).
        if rms_db < self.noise_floor_db {
            self.noise_floor_db = rms_db.max(-90.0);
        } else {
            let rate = if self.state == VadState::Speech {
                self.cfg.floor_rise_db_per_sec_speech
            } else {
                self.cfg.floor_rise_db_per_sec_silence
            };
            self.noise_floor_db += rate * (VAD_FRAME_SAMPLES as f64 / SAMPLE_RATE as f64);
        }

        let mut voiced =
            rms_db > self.noise_floor_db + self.cfg.gate_db && rms_db > self.cfg.min_voice_dbfs;
        if voiced
            && let Some(classifier) = self.classifier.as_mut()
            && !classifier.is_voice(frame)
        {
            voiced = false;
            self.classifier_vetoes += 1;
        }

        if voiced {
            self.last_voiced_end = frame_index + 1;
        }

        match self.state {
            VadState::Silence => {
                if voiced {
                    let (start, count, _grace) = self.voiced_run.unwrap_or((
                        frame_index,
                        0,
                        self.cfg.voiced_run_grace_frames,
                    ));
                    let count = count + 1;
                    self.voiced_run = Some((start, count, self.cfg.voiced_run_grace_frames));
                    if u64::from(count) * 20 >= self.cfg.min_speech_ms {
                        self.state = VadState::Speech;
                        self.unvoiced_run = 0;
                        let pre_pad_sec = self.cfg.pre_pad_ms as f64 / 1000.0;
                        let t = (vad_frame_to_sec(start) - pre_pad_sec).max(self.floor_time_sec);
                        edges.push(VadEdge::SpeechStarted { t_sec: t });
                        self.voiced_run = None;
                    }
                } else if let Some((start, count, grace)) = self.voiced_run {
                    if grace > 0 {
                        self.voiced_run = Some((start, count, grace - 1));
                    } else {
                        self.voiced_run = None;
                    }
                }
            }
            VadState::Speech => {
                if voiced {
                    self.unvoiced_run = 0;
                } else {
                    self.unvoiced_run += 1;
                    if u64::from(self.unvoiced_run) * 20 >= self.cfg.endpoint_ms {
                        self.state = VadState::Silence;
                        let t = vad_frame_to_sec(self.last_voiced_end);
                        self.floor_time_sec = t;
                        edges.push(VadEdge::Endpoint { t_sec: t });
                        self.voiced_run = None;
                        self.unvoiced_run = 0;
                    }
                }
            }
        }
    }
}

/// Frame RMS level in dBFS (f32 full scale = 0 dBFS).
fn frame_dbfs(frame: &[f32]) -> f64 {
    let mean_sq = frame
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / frame.len() as f64;
    if mean_sq <= 1e-12 {
        return -120.0;
    }
    10.0 * mean_sq.log10()
}

// ---------------------------------------------------------------------------
// bd-rt-listen-cmd-i48i: the live driver — `fw robot listen`
//
// A deliberately boring, single-threaded loop composing the tested pieces:
// capture -> resample -> SessionBuffer -> StreamingVad -> step decode ->
// EmissionPolicy -> NDJSON events. It bypasses the 10-stage orchestrator on
// purpose (path-typed stages + fixed wall-clock budgets are structurally
// wrong for an unbounded session; decided in the epic, do not relitigate).
// Every piece of intelligence lives behind a trait (capture source, policy,
// VAD) so it is unit-testable and swappable.
//
// v1 scope notes (staged landing, each with its owning bead):
// - Emission policy: built-in `endpoint-commit` bootstrap (partials each
//   step, one committed delta at utterance close). AlignAtt
//   (bd-rt-alignatt-fry9) and LocalAgreement (bd-rt-local-agreement-l5x8)
//   plug into `EmissionPolicy` and flip the default when they land.
// - Confirm lane (bd-rt-confirm-lane-3okr) and persistence
//   (bd-rt-persist-a66y) attach at the utterance_end seam below; the
//   `--quality-model` / `--db` flags land WITH those beads.
// ---------------------------------------------------------------------------

use crate::error::{FwError, FwResult};
use crate::robot::{self, ListenSessionInfo, ListenSessionStats, UtteranceEndReason};

/// Where the session's audio comes from.
#[derive(Debug, Clone)]
pub enum ListenSource {
    /// Live microphone: cpal primary, ffmpeg fallback (with a warning).
    Mic {
        device: Option<String>,
        backend: CaptureBackend,
    },
    /// Raw PCM on stdin (`--source stdin-pcm`) — the composable path.
    StdinPcm {
        format: crate::capture::PcmFormat,
        sample_rate: u32,
        channels: u16,
    },
    /// Replay an audio file (WAV via hound in v1), paced or as-fast-as-possible.
    FileReplay {
        path: std::path::PathBuf,
        realtime_pace: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureBackend {
    Auto,
    Cpal,
    Ffmpeg,
}

/// Full session configuration (CLI flags marshal into this).
#[derive(Debug, Clone)]
pub struct ListenConfig {
    pub source: ListenSource,
    /// Fast model spec; `None` = language-keyed default (en => tiny.en,
    /// otherwise multilingual tiny; bd-rt-model-provision contract).
    pub fast_model: Option<String>,
    pub language: Option<String>,
    pub step_ms: u64,
    pub buffer: SessionBufferConfig,
    pub vad: StreamingVadConfig,
    pub vad_enabled: bool,
    pub max_seconds: f64,
    pub max_utterance_sec: f64,
    pub emit_partials: bool,
    pub stats_interval_sec: f64,
    pub capture_buffer_sec: f64,
    pub policy: ListenPolicy,
    pub alignatt_holdback_ms: u64,
    /// Confirm-lane quality model (bd-rt-confirm-lane-3okr).
    pub quality_model: QualityModelSetting,
    /// Max unconfirmed jobs before the oldest is dropped (`confirm_lag`
    /// warning). Live output never blocks on the quality lane.
    pub confirm_queue_bound: usize,
    /// Seconds the session-end path waits for in-flight confirms before
    /// abandoning them with a `confirm_drain_timeout` warning.
    pub confirm_drain_sec: f64,
    /// Persist the session to SQLite at utterance granularity
    /// (bd-rt-persist-a66y). Library default OFF; the CLI turns it on
    /// unless `--no-persist` (mirrors the batch transcribe flag).
    pub persist: bool,
    /// Database file for the persistence sink (canonical default:
    /// `.franken_whisper/storage.sqlite3`, same as batch/sync tooling).
    pub db_path: std::path::PathBuf,
}

/// `--quality-model` setting for the confirm lane (bd-rt-confirm-lane-3okr).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualityModelSetting {
    /// Default product behavior: large-v3-turbo when its package is
    /// installed; lane disabled otherwise. Never downloads models.
    Auto,
    /// Explicit opt-out (`--quality-model none`).
    Disabled,
    /// Explicit model spec (`--quality-model <spec>`).
    Explicit(String),
}

/// Which emission policy drives commits (bd-rt-alignatt-fry9 owns the
/// default flip to AlignAtt).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListenPolicy {
    /// AlignAtt (default): attention-gated incremental commits.
    AlignAtt,
    /// Baseline: one commit per utterance at close.
    EndpointCommit,
}

impl Default for ListenConfig {
    fn default() -> Self {
        Self {
            source: ListenSource::StdinPcm {
                format: crate::capture::PcmFormat::S16le,
                sample_rate: 16_000,
                channels: 1,
            },
            fast_model: None,
            language: None,
            step_ms: 300,
            buffer: SessionBufferConfig::default(),
            vad: StreamingVadConfig::default(),
            vad_enabled: true,
            max_seconds: 0.0,
            max_utterance_sec: 90.0,
            emit_partials: true,
            stats_interval_sec: 30.0,
            capture_buffer_sec: 30.0,
            policy: ListenPolicy::AlignAtt,
            alignatt_holdback_ms: 200,
            quality_model: QualityModelSetting::Auto,
            confirm_queue_bound: CONFIRM_QUEUE_DEFAULT_BOUND,
            confirm_drain_sec: CONFIRM_DRAIN_DEFAULT_SEC,
            persist: false,
            db_path: std::path::PathBuf::from(".franken_whisper/storage.sqlite3"),
        }
    }
}

/// What a policy decided for one step decode.
#[derive(Debug, Clone, Default)]
pub struct PolicyDecision {
    /// Newly committed text (append-only; empty when nothing commits).
    pub commit_text: String,
    /// Committed-through time on the session clock (policy's word boundary).
    pub commit_through_sec: Option<f64>,
    /// Mutable tail preview (emitted as transcript.partial when enabled).
    pub partial_tail: Option<String>,
    /// Token count backing `commit_text` (for the delta event).
    pub commit_tokens: u64,
    /// Mean confidence proxy for the committed text.
    pub commit_confidence: Option<f64>,
    /// True when the policy withheld an otherwise-eligible commit this step
    /// (quality gate / holdback zone) — counted into session stats.
    pub holdback: bool,
}

/// The emission-policy seam (bd-rt-listen-cmd + policy beads): decides,
/// per step decode over the rolling buffer, what becomes committed
/// (append-only `transcript.delta`) versus mutable preview.
pub trait EmissionPolicy: Send {
    /// A mid-utterance step decode (slice edge is LIVE — audio continues).
    /// `slice_sec` is the decoded utterance slice's audio duration; all
    /// [`PolicyDecision`] times are SLICE-relative (driver re-bases).
    fn step(
        &mut self,
        out: &crate::native_engine::decode::DecodeOutput,
        slice_sec: f64,
    ) -> PolicyDecision;
    /// The endpoint decode (audio for this utterance is COMPLETE — no
    /// holdback; commit everything not yet committed).
    fn finalize(
        &mut self,
        out: &crate::native_engine::decode::DecodeOutput,
        slice_sec: f64,
    ) -> PolicyDecision;
    /// Whether decodes must record the per-token attention tap.
    fn needs_token_attn(&self) -> bool {
        false
    }
    /// Reset per-utterance state.
    fn reset(&mut self);
    /// Stable policy name for listen.session_start.
    fn name(&self) -> &'static str;
}

/// Bootstrap policy: zero intelligence, maximum honesty. Every step's full
/// decode is the mutable preview; nothing commits until the endpoint decode
/// commits the entire utterance as one delta. This is the baseline arm the
/// latency harness keeps forever.
#[derive(Debug, Default)]
pub struct EndpointCommitPolicy;

impl EmissionPolicy for EndpointCommitPolicy {
    fn step(
        &mut self,
        out: &crate::native_engine::decode::DecodeOutput,
        _slice_sec: f64,
    ) -> PolicyDecision {
        let tail = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        PolicyDecision {
            partial_tail: if tail.is_empty() { None } else { Some(tail) },
            ..PolicyDecision::default()
        }
    }

    fn finalize(
        &mut self,
        out: &crate::native_engine::decode::DecodeOutput,
        _slice_sec: f64,
    ) -> PolicyDecision {
        let text = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let confidences: Vec<f64> = out.segments.iter().filter_map(|s| s.confidence).collect();
        let commit_confidence = if confidences.is_empty() {
            None
        } else {
            Some(confidences.iter().sum::<f64>() / confidences.len() as f64)
        };
        let end = out
            .segments
            .iter()
            .filter_map(|s| s.end_sec)
            .fold(None::<f64>, |acc, t| Some(acc.map_or(t, |a| a.max(t))));
        PolicyDecision {
            // SLICE-relative (driver re-bases with the utterance start).
            commit_through_sec: end,
            commit_tokens: out.windows.iter().map(|w| w.tokens as u64).sum(),
            commit_confidence,
            commit_text: text,
            partial_tail: None,
            holdback: false,
        }
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "endpoint-commit"
    }
}

/// AlignAtt emission policy (bd-rt-alignatt-fry9), segment-grain v1.
///
/// The published policy (Simul-Whisper / SimulStreaming): a token whose
/// alignment-head cross-attention focuses at least `holdback_frames`
/// (20 ms encoder frames) before the live slice edge is stable; commit it.
/// This v1 applies the rule at SEGMENT granularity: the attention prefix
/// rule (walk tokens in order, stop at the first token past the safe
/// limit — never commit past a hole) yields a safe TIME boundary, and
/// whole segments ending before that boundary commit. Segment grain keeps
/// append-only reconstruction trivial (utterance text == joined committed
/// segments == joined deltas); token/word-grain commits need per-token
/// text in the tap output and are this bead's recorded refinement.
///
/// The boundary guard (Simul-Whisper's drop-the-edge-word) falls out of
/// the grain: a segment whose END sits inside the holdback zone never
/// commits.
#[derive(Debug)]
pub struct AlignAttPolicy {
    /// Danger-zone width in 20 ms encoder frames (default 10 = 200 ms).
    holdback_frames: u32,
}

impl AlignAttPolicy {
    #[must_use]
    pub fn new(holdback_ms: u64) -> Self {
        Self {
            holdback_frames: ((holdback_ms / 20).max(1)) as u32,
        }
    }

    /// The attention-safe time boundary (slice-relative seconds): the
    /// largest attention frame in the contiguous safe prefix of the
    /// slice's token stream. The first token attending inside the
    /// holdback zone stops the prefix — never commit past a hole.
    fn safe_time_sec(
        &self,
        out: &crate::native_engine::decode::DecodeOutput,
        slice_sec: f64,
    ) -> f64 {
        let slice_frames = (slice_sec * 50.0) as u32;
        let limit = slice_frames.saturating_sub(self.holdback_frames);
        let mut safe_frame: u32 = 0;
        for window in &out.windows {
            for tap in &window.token_attn {
                if tap.attn_frame > limit {
                    return f64::from(safe_frame) * 0.02;
                }
                safe_frame = safe_frame.max(tap.attn_frame);
            }
        }
        f64::from(safe_frame) * 0.02
    }

    /// Commit the first `committable` segments of the slice; everything the
    /// driver hands us is fresh (the slice origin advances past committed
    /// audio), so there is no cross-decode bookkeeping to get wrong.
    fn decision(
        out: &crate::native_engine::decode::DecodeOutput,
        committable: usize,
    ) -> PolicyDecision {
        let mut decision = PolicyDecision::default();
        let fresh = &out.segments[..committable.min(out.segments.len())];
        if !fresh.is_empty() {
            decision.commit_text = fresh
                .iter()
                .map(|segment| segment.text.trim())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            let confidences: Vec<f64> = fresh
                .iter()
                .filter_map(|segment| segment.confidence)
                .collect();
            decision.commit_confidence = if confidences.is_empty() {
                None
            } else {
                Some(confidences.iter().sum::<f64>() / confidences.len() as f64)
            };
            decision.commit_through_sec = fresh
                .iter()
                .filter_map(|segment| segment.end_sec)
                .next_back();
            decision.commit_tokens = fresh.len() as u64;
        }
        let tail = out.segments[committable.min(out.segments.len())..]
            .iter()
            .map(|segment| segment.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        decision.partial_tail = if tail.is_empty() { None } else { Some(tail) };
        decision
    }
}

impl EmissionPolicy for AlignAttPolicy {
    fn step(
        &mut self,
        out: &crate::native_engine::decode::DecodeOutput,
        slice_sec: f64,
    ) -> PolicyDecision {
        // Hallucination gate: a commit is irreversible, so a slice that
        // smells like non-speech commits nothing (classic Whisper gate:
        // high no-speech probability or very low mean log-prob — first
        // campaign receipt: alignatt committed invented text on the
        // music-only negative fixture; endpoint-commit did not, because it
        // only ever emits VAD-closed utterance text). The tail still flows
        // as a mutable partial; finalize/utterance close is unaffected.
        let no_speech = out
            .windows
            .iter()
            .map(|w| w.no_speech_prob)
            .fold(0.0_f64, f64::max);
        let avg_logprob = out
            .windows
            .iter()
            .map(|w| w.avg_logprob)
            .fold(0.0_f64, f64::min);
        if no_speech > 0.6 || avg_logprob < -1.0 {
            let mut decision = Self::decision(out, 0);
            decision.holdback = true;
            tracing::debug!(
                slice_sec,
                no_speech,
                avg_logprob,
                "alignatt step: low-confidence slice, commits held"
            );
            return decision;
        }
        let safe = self.safe_time_sec(out, slice_sec);
        // Closure guard: NEVER commit the slice's final segment
        // mid-utterance, no matter how early its attention sits. A decode
        // over truncated audio routinely hallucinates a confident sentence
        // close there (first-campaign receipt: 3.0 s slice produced
        // "And so am I fellow Americans." with attention safely behind a
        // 2.06 s boundary; the real audio continued "...ask not what your
        // country can do"). A segment is trustworthy only when the decoder
        // started another one after it.
        let committable = out
            .segments
            .iter()
            .take(out.segments.len().saturating_sub(1))
            .take_while(|segment| segment.end_sec.is_some_and(|end| end <= safe))
            .count();
        let decision = Self::decision(out, committable);
        tracing::debug!(
            slice_sec,
            safe_time_sec = safe,
            committed = committable,
            segments = out.segments.len(),
            committed_chars = decision.commit_text.len(),
            partial_chars = decision.partial_tail.as_deref().map_or(0, str::len),
            "alignatt step"
        );
        decision
    }

    fn finalize(
        &mut self,
        out: &crate::native_engine::decode::DecodeOutput,
        _slice_sec: f64,
    ) -> PolicyDecision {
        // Audio complete: no holdback, no closure risk — commit everything.
        Self::decision(out, out.segments.len())
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "alignatt"
    }

    fn needs_token_attn(&self) -> bool {
        true
    }
}

/// Resolve the fast-lane model per the bd-rt-model-provision contract.
/// Returns (loaded model, model label, fallback_warning).
fn resolve_fast_model(
    config: &ListenConfig,
) -> FwResult<(
    std::sync::Arc<crate::native_engine::NativeWhisperModel>,
    String,
    Option<String>,
)> {
    let default_spec = match config.language.as_deref() {
        Some("en") => "tiny.en",
        _ => "tiny",
    };
    let spec = config.fast_model.as_deref().unwrap_or(default_spec);
    match crate::native_engine::resolve_model(spec) {
        Ok(path) => {
            let model = crate::native_engine::NativeWhisperModel::load(&path)?;
            Ok((model, spec.to_owned(), None))
        }
        Err(missing) => {
            // Fallback contract: session MUST start when the default
            // release package is present; degraded latency beats refusal.
            let fallback = crate::native_engine::resolve_model("default").map_err(|_| missing)?;
            let model = crate::native_engine::NativeWhisperModel::load(&fallback)?;
            let hint = if spec == "tiny" {
                "fw pull tiny"
            } else {
                "fw pull tiny-en"
            };
            Ok((
                model,
                "large-v3-turbo".to_owned(),
                Some(format!(
                    "fast model `{spec}` is not cached; using the turbo package as the fast lane \
                     (higher latency). Install the fast-lane package with: {hint}"
                )),
            ))
        }
    }
}

/// Load a WAV file for `--source file-replay` (any rate/channels; the
/// resampler chain normalizes). v1 accepts WAV via hound; other containers
/// go through `fw transcribe`'s batch path or stdin-pcm piping.
fn load_replay_wav(path: &std::path::Path) -> FwResult<(Vec<f32>, u32, u16)> {
    let mut reader = hound::WavReader::open(path).map_err(|error| {
        FwError::InvalidRequest(format!(
            "file-replay could not open `{}`: {error} (v1 accepts WAV; for other formats pipe \
             PCM: ffmpeg -i INPUT -f s16le - | fw robot listen --source stdin-pcm)",
            path.display()
        ))
    })?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|error| FwError::InvalidRequest(format!("bad WAV samples: {error}")))?,
        hound::SampleFormat::Int => {
            let denom = f32::from(i16::MAX) + 1.0;
            reader
                .samples::<i16>()
                .map(|s| s.map(|v| f32::from(v) / denom))
                .collect::<Result<_, _>>()
                .map_err(|error| FwError::InvalidRequest(format!("bad WAV samples: {error}")))?
        }
    };
    Ok((samples, spec.sample_rate, spec.channels))
}

/// (source, capture-backend label, device label, optional fallback warning).
type OpenedCapture = (
    Box<dyn crate::capture::CaptureSource>,
    &'static str,
    String,
    Option<String>,
);

fn open_capture_source(config: &ListenConfig) -> FwResult<OpenedCapture> {
    match &config.source {
        ListenSource::FileReplay {
            path,
            realtime_pace,
        } => {
            let (samples, rate, channels) = load_replay_wav(path)?;
            let source = if *realtime_pace {
                crate::capture::FixtureCaptureSource::new_paced(samples, rate, channels)?
            } else {
                crate::capture::FixtureCaptureSource::new_unpaced(
                    samples,
                    rate,
                    channels,
                    (u64::from(rate) * config.step_ms / 1000).max(1) as usize,
                )?
            };
            Ok((Box::new(source), "none", path.display().to_string(), None))
        }
        ListenSource::StdinPcm {
            format,
            sample_rate,
            channels,
        } => {
            use std::io::IsTerminal as _;
            let source = crate::capture::StdinPcmSource::open(
                *format,
                *sample_rate,
                *channels,
                config.capture_buffer_sec,
                std::io::stdin().is_terminal(),
            )?;
            Ok((Box::new(source), "none", "<stdin>".to_owned(), None))
        }
        ListenSource::Mic { device, backend } => {
            let device_label = device.clone().unwrap_or_else(|| "<default>".to_owned());
            match backend {
                CaptureBackend::Ffmpeg => {
                    let source = crate::capture::FfmpegCaptureSource::open(
                        device.as_deref(),
                        None,
                        None,
                        config.capture_buffer_sec,
                    )?;
                    Ok((Box::new(source), "ffmpeg", device_label, None))
                }
                CaptureBackend::Cpal => {
                    let source = crate::capture::CpalCaptureSource::open(
                        device.as_deref(),
                        config.capture_buffer_sec,
                    )?;
                    Ok((Box::new(source), "cpal", device_label, None))
                }
                CaptureBackend::Auto => {
                    match crate::capture::CpalCaptureSource::open(
                        device.as_deref(),
                        config.capture_buffer_sec,
                    ) {
                        Ok(source) => Ok((Box::new(source), "cpal", device_label, None)),
                        Err(cpal_error) => {
                            let source = crate::capture::FfmpegCaptureSource::open(
                                device.as_deref(),
                                None,
                                None,
                                config.capture_buffer_sec,
                            )
                            .map_err(|ffmpeg_error| {
                                FwError::BackendUnavailable(format!(
                                    "no capture backend available: cpal failed ({cpal_error}); \
                                     ffmpeg failed ({ffmpeg_error})"
                                ))
                            })?;
                            Ok((
                                Box::new(source),
                                "ffmpeg",
                                device_label,
                                Some(format!(
                                    "cpal capture unavailable ({cpal_error}); fell back to ffmpeg \
                                     device capture"
                                )),
                            ))
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// bd-rt-confirm-lane-3okr: background per-utterance quality confirmation
// ---------------------------------------------------------------------------

/// Default bound on unconfirmed confirm-lane jobs. When the queue is full,
/// the OLDEST unconfirmed utterance is dropped (with a `confirm_lag`
/// warning): the live fast lane must never block on the quality lane.
pub const CONFIRM_QUEUE_DEFAULT_BOUND: usize = 4;

/// Default seconds the session-end path waits for in-flight confirms before
/// abandoning them (`confirm_drain_timeout` warning).
pub const CONFIRM_DRAIN_DEFAULT_SEC: f64 = 10.0;

const CONFIRM_QUALITY_MODEL_DEFAULT: &str = "large-v3-turbo";

/// One closed utterance awaiting quality-lane re-transcription.
pub(crate) struct ConfirmJob {
    pub utterance_id: u32,
    /// Full-utterance PCM captured at close (batch-grade context: no slice
    /// truncation compromises).
    pub pcm: Vec<f32>,
    /// The fast lane's committed text for this utterance.
    pub committed_text: String,
    /// Fast lane's pinned/detected language at close (stability: the quality
    /// decode must not re-detect and diverge mid-session).
    pub language: Option<String>,
}

/// What the worker sends back to the session loop.
#[derive(Debug)]
pub(crate) enum ConfirmLaneEvent {
    Verdict(ConfirmVerdict),
    Warning {
        reason: String,
        detail: serde_json::Value,
    },
    /// The oldest queued job was dropped because the queue was full.
    DroppedOldest {
        utterance_id: u32,
        queue_bound: usize,
    },
}

/// Mapped outcome of one quality-lane decode + tracker comparison.
#[derive(Debug)]
pub(crate) struct ConfirmVerdict {
    pub utterance_id: u32,
    pub confirmed: bool,
    pub correction_id: u64,
    /// Turbo's batch-grade segments (empty on confirm).
    pub corrected_segments: Vec<crate::model::TranscriptionSegment>,
    pub drift_wer: f64,
    pub drift_confidence_delta: f64,
    pub drift_edit_distance: usize,
    pub decode_ms: u64,
    pub quality_model_id: String,
}

/// Re-transcription result contract for the injected decoder seam. The
/// production decoder lazily loads the quality model on first call; tests
/// inject fakes without touching disk.
pub(crate) enum DecodeOutcome {
    Segments(Vec<crate::model::TranscriptionSegment>, String),
    /// Quality model could not be loaded/used at all (missing package,
    /// corrupt file, OOM). The lane disables itself; the session continues
    /// fast-only (graceful degradation is LOCKED policy).
    Unavailable(String),
    /// Decode aborted because the session ended (checkpoint fired).
    Aborted,
    Failed(String),
}

pub(crate) type QualityDecoder = Box<
    dyn FnMut(&ConfirmJob, &str, &(dyn Fn() -> bool + Sync)) -> DecodeOutcome
        + Send,
>;

struct ConfirmQueueState {
    jobs: std::collections::VecDeque<ConfirmJob>,
    bound: usize,
    shutdown: bool,
}

/// Handle owned by the session loop: bounded job queue + results channel.
pub(crate) struct ConfirmLane {
    shared: std::sync::Arc<(
        std::sync::Mutex<ConfirmQueueState>,
        std::sync::Condvar,
    )>,
    abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
    results_tx: std::sync::mpsc::Sender<ConfirmLaneEvent>,
    results_rx: std::sync::mpsc::Receiver<ConfirmLaneEvent>,
    depth: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl ConfirmLane {
    /// Spawn the single background worker. `decoder` runs entirely on the
    /// worker thread; the tracker lives there too.
    #[allow(clippy::type_complexity)]
    pub(crate) fn spawn(bound: usize, mut decoder: QualityDecoder) -> Self {
        let shared = std::sync::Arc::new((
            std::sync::Mutex::new(ConfirmQueueState {
                jobs: std::collections::VecDeque::new(),
                bound,
                shutdown: false,
            }),
            std::sync::Condvar::new(),
        ));
        let abort = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let depth = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (results_tx, results_rx) = std::sync::mpsc::channel::<ConfirmLaneEvent>();
        let worker_shared = std::sync::Arc::clone(&shared);
        let worker_abort = std::sync::Arc::clone(&abort);
        let worker_depth = std::sync::Arc::clone(&depth);
        let spawned = {
            let results_tx = results_tx.clone();
            std::thread::Builder::new()
                .name("fw-confirm".to_owned())
                .spawn(move || {
                let (queue_mtx, queue_cv) = &*worker_shared;
                let mut tracker = crate::speculation::CorrectionTracker::new(
                    crate::speculation::CorrectionTolerance::default(),
                );
                let mut prev_confirmed_text = String::new();
                // window_id/seq namespace: the utterance id itself (unique
                // per session; the tracker instance is per-worker).
                loop {
                    let job = {
                        let mut state = queue_mtx.lock().expect("confirm queue poisoned");
                        loop {
                            if let Some(job) = state.jobs.pop_front() {
                                break Some(job);
                            }
                            if state.shutdown {
                                break None;
                            }
                            state = queue_cv.wait(state).expect("confirm queue poisoned");
                        }
                    };
                    let Some(job) = job else { break };
                    let queue_depth = worker_depth.load(std::sync::atomic::Ordering::Relaxed);
                    let is_abort = || {
                        worker_abort.load(std::sync::atomic::Ordering::Relaxed)
                    };
                    let decode_started = std::time::Instant::now();
                    match decoder(&job, &prev_confirmed_text, &is_abort) {
                        DecodeOutcome::Aborted => break,
                        DecodeOutcome::Unavailable(detail) => {
                            // Locked graceful-degradation shape: warn once,
                            // disable the lane, keep the session alive.
                            let _ = results_tx.send(ConfirmLaneEvent::Warning {
                                reason: "quality_model_unavailable".to_owned(),
                                detail: serde_json::json!({ "detail": detail }),
                            });
                            // Poison-pill the queue: everything already
                            // queued is skipped silently (the fast text is
                            // already published history). Depth resets to
                            // zero — nothing remains queued or in flight,
                            // so session-end drain must not report
                            // phantom abandons.
                            {
                                let mut state = queue_mtx
                                    .lock()
                                    .expect("confirm queue poisoned");
                                state.jobs.clear();
                                drop(state);
                                worker_depth.store(
                                    0,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            break;
                        }
                        DecodeOutcome::Failed(detail) => {
                            tracing::info!(
                                utterance_id = job.utterance_id,
                                queue_depth,
                                verdict = "failed",
                                wer_drift = -1.0_f64,
                                "confirm job failed"
                            );
                            let _ = results_tx.send(ConfirmLaneEvent::Warning {
                                reason: "confirm_job_failed".to_owned(),
                                detail: serde_json::json!({
                                    "detail": detail,
                                    "utterance_id": job.utterance_id,
                                }),
                            });
                        }
                        DecodeOutcome::Segments(segments, quality_model_id) => {
                            let decode_ms =
                                decode_started.elapsed().as_millis() as u64;
                            let window_id = u64::from(job.utterance_id);
                            let fast_segment = crate::model::TranscriptionSegment {
                                start_sec: Some(0.0),
                                end_sec: None,
                                text: job.committed_text.clone(),
                                speaker: None,
                                confidence: None,
                            };
                            tracker.register_partial(crate::speculation::PartialTranscript {
                                seq: window_id,
                                window_id,
                                model_id: "fast".to_owned(),
                                segments: vec![fast_segment],
                                latency_ms: 0,
                                confidence_mean: 0.0,
                                emitted_at_rfc3339: chrono::Utc::now().to_rfc3339(),
                                status: crate::speculation::PartialStatus::Pending,
                            });
                            let mapped = match tracker.submit_quality_result(
                                window_id,
                                &quality_model_id,
                                segments,
                                decode_ms,
                            ) {
                                Ok(crate::speculation::CorrectionDecision::Confirm {
                                    drift,
                                    ..
                                }) => {
                                    prev_confirmed_text = job.committed_text.clone();
                                    tracing::info!(
                                        utterance_id = job.utterance_id,
                                        queue_depth,
                                        decode_ms,
                                        verdict = "confirm",
                                        wer_drift = drift.wer_approx,
                                        "confirm verdict"
                                    );
                                    ConfirmVerdict {
                                        utterance_id: job.utterance_id,
                                        confirmed: true,
                                        correction_id: 0,
                                        corrected_segments: Vec::new(),
                                        drift_wer: drift.wer_approx,
                                        drift_confidence_delta: drift.confidence_delta,
                                        drift_edit_distance: drift.text_edit_distance,
                                        decode_ms,
                                        quality_model_id,
                                    }
                                }
                                Ok(
                                    crate::speculation::CorrectionDecision::Correct {
                                        correction,
                                    },
                                ) => {
                                    prev_confirmed_text = correction
                                        .corrected_segments
                                        .iter()
                                        .map(|s| s.text.as_str())
                                        .collect::<Vec<_>>()
                                        .join(" ")
                                        .trim()
                                        .to_owned();
                                    tracing::info!(
                                        utterance_id = job.utterance_id,
                                        queue_depth,
                                        decode_ms,
                                        verdict = "correct",
                                        wer_drift = correction.drift.wer_approx,
                                        correction_id = correction.correction_id,
                                        "confirm verdict"
                                    );
                                    ConfirmVerdict {
                                        utterance_id: job.utterance_id,
                                        confirmed: false,
                                        correction_id: correction.correction_id,
                                        corrected_segments: correction.corrected_segments,
                                        drift_wer: correction.drift.wer_approx,
                                        drift_confidence_delta: correction
                                            .drift
                                            .confidence_delta,
                                        drift_edit_distance: correction
                                            .drift
                                            .text_edit_distance,
                                        decode_ms,
                                        quality_model_id: correction.quality_model_id,
                                    }
                                }
                                Err(error) => {
                                    let _ = results_tx.send(ConfirmLaneEvent::Warning {
                                        reason: "confirm_job_failed".to_owned(),
                                        detail: serde_json::json!({
                                            "detail": error.to_string(),
                                            "utterance_id": job.utterance_id,
                                        }),
                                    });
                                    worker_depth
                                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                                    continue;
                                }
                            };
                            let _ = results_tx
                                .send(ConfirmLaneEvent::Verdict(mapped));
                        }
                    }
                    worker_depth.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                }
                })
        };
        if spawned.is_err() {
            // Worker threads are best-effort infrastructure; if the OS
            // refuses the thread the session degrades to fast-only.
            return Self {
                shared,
                abort,
                results_tx: std::sync::mpsc::channel().0,
                results_rx: std::sync::mpsc::channel().1,
                depth,
            };
        }
        Self {
            shared,
            abort,
            results_tx,
            results_rx,
            depth,
        }
    }
    /// Enqueue one closed utterance. NEVER blocks: when the queue is at its
    /// bound, the oldest unconfirmed job is dropped and reported so the
    /// session loop can emit the `confirm_lag` warning.
    pub(crate) fn submit(&self, job: ConfirmJob) {
        let (queue_mtx, queue_cv) = &*self.shared;
        let mut state = queue_mtx.lock().expect("confirm queue poisoned");
        if state.shutdown {
            return;
        }
        let mut dropped = None;
        while state.jobs.len() >= state.bound {
            if let Some(old) = state.jobs.pop_front() {
                dropped = Some(old.utterance_id);
                self.depth.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                break;
            }
        }
        state.jobs.push_back(job);
        self.depth.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        drop(state);
        queue_cv.notify_one();
        if let Some(utterance_id) = dropped {
            let _ = self.results_tx.send(ConfirmLaneEvent::DroppedOldest {
                utterance_id,
                queue_bound: self.bound(),
            });
        }
    }

    fn bound(&self) -> usize {
        self.shared.0.lock().expect("confirm queue poisoned").bound
    }

    pub(crate) fn try_recv(&self) -> Option<ConfirmLaneEvent> {
        self.results_rx.try_recv().ok()
    }

    /// Session-end collection: gather worker events for up to `drain_sec`,
    /// then shut the lane down (queue cleared, abort fired so an in-flight
    /// decode checkpoints out promptly). Returns the collected events plus
    /// the number of jobs abandoned unfinished (0 when the lane drained
    /// cleanly). Never blocks past `drain_sec`.
    pub(crate) fn drain(&self, drain_sec: f64) -> (Vec<ConfirmLaneEvent>, usize) {
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_secs_f64(drain_sec.max(0.0));
        let mut events = Vec::new();
        loop {
            if self.depth.load(std::sync::atomic::Ordering::Relaxed) == 0 {
                break;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            let wait = std::time::Duration::from_millis(50)
                .min(deadline.saturating_duration_since(now));
            match self.results_rx.recv_timeout(wait) {
                Ok(event) => events.push(event),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        {
            let (queue_mtx, queue_cv) = &*self.shared;
            let mut state = queue_mtx.lock().expect("confirm queue poisoned");
            state.shutdown = true;
            state.jobs.clear();
            drop(state);
            queue_cv.notify_all();
        }
        self.abort
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let abandoned = self.depth.load(std::sync::atomic::Ordering::Relaxed);
        (events, abandoned)
    }
}


/// Decide confirm-lane enablement + quality-model label up front (session
/// start stays fast: resolution is a cheap path stat, the LOAD stays lazy on
/// the worker thread). Matrix (polish round 1 item 2):
/// - `Disabled` => off.
/// - `Auto` => ON iff the turbo package is installed AND the fast lane did
///   NOT fall back to turbo (self-confirmation is meaningless).
/// - `Explicit(spec)` => honored unless `spec` IS the effective fast model.
pub(crate) fn resolve_confirm_lane(
    setting: &QualityModelSetting,
    effective_fast_label: &str,
) -> Option<String> {
    match setting {
        QualityModelSetting::Disabled => None,
        QualityModelSetting::Auto => {
            if effective_fast_label == CONFIRM_QUALITY_MODEL_DEFAULT {
                return None;
            }
            crate::native_engine::resolve_model(CONFIRM_QUALITY_MODEL_DEFAULT)
                .ok()
                .map(|_| CONFIRM_QUALITY_MODEL_DEFAULT.to_owned())
        }
        QualityModelSetting::Explicit(spec) => {
            if spec == effective_fast_label {
                return None;
            }
            crate::native_engine::resolve_model(spec)
                .ok()
                .map(|_| spec.clone())
        }
    }
}

// ---------------------------------------------------------------------------
// bd-rt-persist-a66y: utterance-granular SQLite persistence for live runs
// ---------------------------------------------------------------------------

/// Crash-durable sink for one live session. One session = ONE run row
/// (`backend = "native-listen"`); durability advances at UTTERANCE
/// granularity — each closed utterance appends its delta segments plus all
/// buffered stream events inside one savepoint transaction and bumps
/// `finished_at` to a "last known alive" wall time (runs.finished_at is NOT
/// NULL; a crashed session is recognizable as a listen run whose events lack
/// the session-end marker). `transcript.partial` events are deliberately NOT
/// persisted — ephemeral preview garnish, unlike batch where every event is
/// kept; the divergence is documented in the storage row-type docs.
pub(crate) struct ListenPersistSink {
    store: crate::storage::RunStore,
    run_id: String,
    pending_segments: Vec<crate::storage::ListenSegmentRow>,
    pending_events: Vec<crate::storage::ListenEventRow>,
    transcript_so_far: String,
    /// Streamed SHA-256 over the concatenated session PCM (16 kHz mono f32,
    /// little-endian), fed as audio arrives. Schema-coherent integrity
    /// metadata ONLY: the audio itself is not retained, so nothing can
    /// verify it today (round-2 anti-ceremony correction). It becomes a
    /// verifiable reference only if `--keep-session-audio` ever lands.
    pcm_hasher: sha2::Sha256,
    next_segment_idx: usize,
}

impl ListenPersistSink {
    /// Open the store, insert the run row, and start the envelope.
    pub(crate) fn open(
        db_path: &std::path::Path,
        run_id: &str,
        started_at_rfc3339: &str,
        input_path_label: &str,
        request_json: &str,
    ) -> FwResult<Self> {
        let store = crate::storage::RunStore::open(db_path)?;
        store.listen_open_run(&crate::storage::ListenRunOpen {
            run_id,
            started_at_rfc3339,
            input_path: input_path_label,
            request_json,
        })?;
        Ok(Self {
            store,
            run_id: run_id.to_owned(),
            pending_segments: Vec::new(),
            pending_events: Vec::new(),
            transcript_so_far: String::new(),
            pcm_hasher: sha2::Digest::new(),
            next_segment_idx: 0,
        })
    }

    /// Feed resampled 16 kHz mono session PCM into the streamed hash.
    pub(crate) fn feed_pcm(&mut self, samples: &[f32]) {
        use sha2::Digest as _;
        for sample in samples {
            self.pcm_hasher.update(&sample.to_le_bytes());
        }
    }

    /// Buffer one stream event for the next flush (partials skipped).
    pub(crate) fn record_event(
        &mut self,
        seq: u64,
        ts_rfc3339: &str,
        event_name: &str,
        value: &serde_json::Value,
    ) {
        if event_name == "transcript.partial" {
            return;
        }
        self.pending_events
            .push(crate::storage::ListenEventRow {
                seq,
                ts_rfc3339: ts_rfc3339.to_owned(),
                code: event_name.to_owned(),
                payload_json: value.to_string(),
            });
    }

    /// Buffer one committed delta slice of the open utterance.
    pub(crate) fn record_delta(&mut self, start_sec: f64, end_sec: f64, text: &str) {
        let idx = self.next_segment_idx;
        self.next_segment_idx += 1;
        self.pending_segments
            .push(crate::storage::ListenSegmentRow {
                idx,
                start_sec,
                end_sec,
                text: text.to_owned(),
            });
    }

    /// Append closed-utterance text to the running persisted transcript.
    pub(crate) fn append_transcript(&mut self, utterance_text: &str) {
        if utterance_text.is_empty() {
            return;
        }
        if !self.transcript_so_far.is_empty() {
            self.transcript_so_far.push(' ');
        }
        self.transcript_so_far.push_str(utterance_text);
    }

    /// Durability point: append the closed utterance's segments + buffered
    /// events atomically and advance the last-known-alive timestamp.
    pub(crate) fn flush_utterance(&mut self, now_rfc3339: &str) -> FwResult<()> {
        if self.pending_segments.is_empty() && self.pending_events.is_empty() {
            return Ok(());
        }
        let segments = std::mem::take(&mut self.pending_segments);
        let events = std::mem::take(&mut self.pending_events);
        self.store.listen_flush_utterance(
            &self.run_id,
            &segments,
            &events,
            &self.transcript_so_far.clone(),
            now_rfc3339,
        )
    }

    /// Close the run: drain remaining events, write true end time + final
    /// stats/warnings + finalized replay envelope with the PCM hash.
    pub(crate) fn close(
        mut self,
        result_json: &str,
        warnings_json: &str,
        finished_at_rfc3339: &str,
    ) -> FwResult<String> {
        use sha2::Digest as _;
        let pcm_sha256 = format!("{:x}", std::mem::take(&mut self.pcm_hasher).finalize());
        let replay_json = serde_json::json!({
            "kind": "live-session",
            "pcm_sha256": pcm_sha256,
            "note": "audio not retained; hash is an integrity fingerprint, not a replayable reference",
        })
        .to_string();
        let events = std::mem::take(&mut self.pending_events);
        self.store.listen_close_run(
            &self.run_id,
            &events,
            result_json,
            warnings_json,
            &replay_json,
            &self.transcript_so_far,
            finished_at_rfc3339,
        )?;
        Ok(pcm_sha256)
    }
}

/// Run one live session, emitting NDJSON event values through `emit`.
/// Returns `Ok(cancelled)` — `true` when the session ended on Ctrl-C (the
/// caller maps that to exit code 130). Fatal errors propagate; the CLI
/// wrapper converts them to the terminal `run_error` event.
///
/// # Utterance lifecycle contract (bd-rt-endpoint-i3k2)
///
/// Every `speech_started` is matched by exactly one later `utterance_end`
/// carrying the same `utterance_id`; ids allocate monotonically from 1 and
/// utterances never nest. `speech_started` fires eagerly at VAD onset, so an
/// utterance whose decode commits nothing still closes with
/// `utterance_end{text:"", delta_count:0}` — empty-text utterance ends are
/// normal (breath/cough triggers), never an error. Deltas for an utterance
/// always precede its `utterance_end`, and the session's `seq` strictly
/// increases across every event.
pub fn run_listen_session(
    config: &ListenConfig,
    emit: &mut dyn FnMut(serde_json::Value) -> FwResult<()>,
    is_cancelled: &(dyn Fn() -> bool + Sync),
) -> FwResult<bool> {
    use crate::native_engine::decode::{AudioCtxPolicy, DecodeParams};

    let session_started = std::time::Instant::now();
    let run_id = uuid::Uuid::new_v4().to_string();
    let mut seq: u64 = 0;
    let now_ts = || chrono::Utc::now().to_rfc3339();

    // Model first: session_start marks "ready" (agents key on it).
    let (model, fast_model_label, fallback_warning) = resolve_fast_model(config)?;
    let (mut capture, capture_backend, device_label, capture_warning) =
        open_capture_source(config)?;

    let source_label = match &config.source {
        ListenSource::Mic { .. } => "mic",
        ListenSource::StdinPcm { .. } => "stdin-pcm",
        ListenSource::FileReplay { .. } => "file-replay",
    };
    let mut policy: Box<dyn EmissionPolicy> = match config.policy {
        ListenPolicy::AlignAtt => Box::new(AlignAttPolicy::new(config.alignatt_holdback_ms)),
        ListenPolicy::EndpointCommit => Box::new(EndpointCommitPolicy),
    };
    let record_token_attn = policy.needs_token_attn();
    let confirm_disabled_by_fallback = fallback_warning.is_some();
    // Confirm lane (bd-rt-confirm-lane-3okr): resolve enablement + label
    // up front (cheap stat), spawn the worker; the model LOAD itself stays
    // lazy on the worker thread so time-to-session_start is unaffected.
    let confirm_label =
        resolve_confirm_lane(&config.quality_model, &fast_model_label);
    let confirm_lane = confirm_label.as_ref().map(|spec| {
        let spec_owned = spec.clone();
        let mut loaded: Option<(
            std::sync::Arc<crate::native_engine::NativeWhisperModel>,
            String,
        )> = None;
        let decoder: QualityDecoder = Box::new(move |job, prev_confirmed, is_abort| {
            let (model, label) = match loaded.as_ref() {
                Some(pair) => pair.clone(),
                None => {
                    let resolved = crate::native_engine::resolve_model(&spec_owned)
                        .map_err(|e| e.to_string())
                        .and_then(|path| {
                            crate::native_engine::NativeWhisperModel::load(&path)
                                .map(|m| (m, spec_owned.clone()))
                                .map_err(|e| e.to_string())
                        });
                    match resolved {
                        Ok(pair) => {
                            loaded = Some(pair.clone());
                            pair
                        }
                        Err(detail) => return DecodeOutcome::Unavailable(detail),
                    }
                }
            };
            if is_abort() {
                return DecodeOutcome::Aborted;
            }
            // Batch-grade decode contract: full 30 s padding semantics
            // (AudioCtxPolicy::Full), no cache (rolling audio can never
            // hit), language pinned to the fast lane's choice, and the
            // previous confirmed utterance as prompt continuity.
            let params = DecodeParams {
                language: job.language.clone(),
                timestamps: true,
                audio_ctx: AudioCtxPolicy::Full,
                bypass_transcript_cache: true,
                initial_prompt: (!prev_confirmed.is_empty())
                    .then(|| prev_confirmed.to_owned()),
                n_threads: 4,
                ..DecodeParams::default()
            };
            let checkpoint = || -> FwResult<()> {
                if is_abort() {
                    Err(FwError::Cancelled("confirm lane abandoned".to_owned()))
                } else {
                    Ok(())
                }
            };
            match model.transcribe(&job.pcm, &params, &checkpoint) {
                Ok(out) => DecodeOutcome::Segments(out.segments, label),
                Err(FwError::Cancelled(_)) => DecodeOutcome::Aborted,
                Err(error) => DecodeOutcome::Failed(error.to_string()),
            }
        });
        ConfirmLane::spawn(config.confirm_queue_bound, decoder)
    });

    // Persistence sink (bd-rt-persist-a66y): one run row per session,
    // utterance-granular flushes. Open failure degrades the session to
    // fast-only; the warning is EMITTED after session_start (streams start
    // with session_start, always) and recorded like every other event.
    let mut persist_open_error: Option<String> = None;
    let input_label = match &config.source {
        ListenSource::FileReplay { path, .. } => path.display().to_string(),
        ListenSource::Mic { .. } => "mic".to_owned(),
        ListenSource::StdinPcm {
            format,
            sample_rate,
            channels,
        } => format!("stdin-pcm:{format:?}/{sample_rate}Hz/{channels}ch"),
    };
    let request_json = serde_json::json!({
        "kind": "listen",
        "source": source_label,
        "fast_model": config.fast_model,
        "language": config.language,
        "step_ms": config.step_ms,
        "policy": config.policy,
        "quality_model": config.quality_model,
        "vad_enabled": config.vad_enabled,
        "emit_partials": config.emit_partials,
        "max_seconds": config.max_seconds,
    })
    .to_string();
    let mut persist_sink: Option<ListenPersistSink> = if config.persist {
        match ListenPersistSink::open(
            &config.db_path,
            &run_id,
            &now_ts(),
            &input_label,
            &request_json,
        ) {
            Ok(sink) => Some(sink),
            Err(error) => {
                persist_open_error = Some(error.to_string());
                None
            }
        }
    } else {
        None
    };
    // Warning payloads collected for warnings_json at close (the persisted
    // degradation trail, independent of stdout).
    let mut persist_warnings: Vec<serde_json::Value> = Vec::new();

    let info = ListenSessionInfo {
        source: source_label.to_owned(),
        capture_backend: capture_backend.to_owned(),
        device: device_label,
        sample_rate_hz: capture.sample_rate(),
        fast_model: fast_model_label.clone(),
        quality_model: confirm_label.clone(),
        policy: policy.name().to_owned(),
        step_ms: config.step_ms,
        partials: config.emit_partials,
        vad: serde_json::json!({
            "enabled": config.vad_enabled,
            "min_speech_ms": config.vad.min_speech_ms,
            "endpoint_ms": config.vad.endpoint_ms,
            "gate_db": config.vad.gate_db,
        }),
    };
    let session_start_value =
        robot::listen_session_start_value(&run_id, seq, &now_ts(), &info);
    if let Some(sink) = persist_sink.as_mut() {
        sink.record_event(seq, &now_ts(), "listen.session_start", &session_start_value);
    }
    emit(session_start_value)?;
    seq += 1;
    if let Some(message) = &fallback_warning {
        let value = robot::listen_warning_value(
            &run_id,
            seq,
            &now_ts(),
            "fast_model_fallback",
            serde_json::json!({"detail": message, "confirm_lane_disabled": confirm_disabled_by_fallback}),
        );
        persist_warnings.push(value.clone());
        if let Some(sink) = persist_sink.as_mut() {
            sink.record_event(seq, &now_ts(), "listen.warning", &value);
        }
        emit(value)?;
        seq += 1;
    }
    if let Some(message) = capture_warning {
        let value = robot::listen_warning_value(
            &run_id,
            seq,
            &now_ts(),
            "fallback_capture_backend",
            serde_json::json!({"detail": message}),
        );
        persist_warnings.push(value.clone());
        if let Some(sink) = persist_sink.as_mut() {
            sink.record_event(seq, &now_ts(), "listen.warning", &value);
        }
        emit(value)?;
        seq += 1;
    }
    if let Some(detail) = persist_open_error {
        let value = robot::listen_warning_value(
            &run_id,
            seq,
            &now_ts(),
            "persist_degraded",
            serde_json::json!({
                "detail": format!("persistence disabled for this session: {detail}"),
            }),
        );
        persist_warnings.push(value.clone());
        emit(value)?;
        seq += 1;
    }

    // Resampler chain: only when the source is not already 16 kHz mono.
    let mut resampler = if capture.sample_rate() != crate::native_engine::mel::SAMPLE_RATE as u32
        || capture.channels() != 1
    {
        Some(crate::capture::StreamingResampler::new(
            capture.sample_rate(),
            capture.channels(),
            crate::native_engine::mel::SAMPLE_RATE as u32,
        )?)
    } else {
        None
    };

    // Session state.
    let mut utterance_id: u32 = 0;
    let mut in_speech = false;
    let mut utterance_started_at: f64 = 0.0;
    let mut utterance_t0: f64 = 0.0;
    let mut committed_text = String::new();
    let mut utterance_committed_through: f64 = 0.0;
    let mut delta_count: u64 = 0;
    let mut partial_generation: u64 = 0;
    let mut stats = ListenSessionStats::default();
    let mut pending_confirm_since: std::collections::HashMap<u32, std::time::Instant> =
        std::collections::HashMap::new();
    let mut confirm_lag_ms: Vec<f64> = Vec::new();
    let mut step_latencies_ms: Vec<f64> = Vec::new();
    let mut last_stats_emit = std::time::Instant::now();
    let mut next_step = std::time::Instant::now();
    let mut raw = vec![0f32; 48_000];
    let mut resampled: Vec<f32> = Vec::new();
    let mut source_ended = false;
    let mut cancelled = false;

    let checkpoint = || -> FwResult<()> {
        if is_cancelled() {
            Err(FwError::Cancelled("listen session interrupted".to_owned()))
        } else {
            Ok(())
        }
    };

    // Emit + record: every stream event funnels through here so the
    // persistence sink sees exactly what stdout sees (minus partials,
    // which are deliberately not persisted). Warnings additionally
    // collect into warnings_json for the final close.
    macro_rules! emit_seq {
        ($value:expr) => {{
            {
                let event_name = $value
                    .get("event")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if event_name == "listen.warning" {
                    persist_warnings.push($value.clone());
                }
                if let Some(sink) = persist_sink.as_mut() {
                    sink.record_event(seq, &now_ts(), event_name, &$value);
                }
            }
            emit($value)?;
            seq += 1;
        }};
    }

    // Utterance-granular durability point (bd-rt-persist-a66y): called at
    // every utterance close. Flush failure emits `persist_degraded` and the
    // session continues — live output is the product.
    macro_rules! persist_flush {
        () => {{
            if let Some(sink) = persist_sink.as_mut() {
                sink.append_transcript(&committed_text);
            }
            let flush_result =
                persist_sink.as_mut().map(|sink| sink.flush_utterance(&now_ts()));
            if let Some(Err(error)) = flush_result {
                emit_seq!(robot::listen_warning_value(
                    &run_id,
                    seq,
                    &now_ts(),
                    "persist_degraded",
                    serde_json::json!({"detail": error.to_string()}),
                ));
            }
        }};
    }

    // One decode over the CURRENT UTTERANCE's audio (sliced from the
    // rolling buffer at the utterance start so previous-utterance
    // keep-back audio is never re-transcribed; linguistic continuity
    // comes from the prompt carry instead).
    let noop_checkpoint = || -> FwResult<()> { Ok(()) };
    let decode_utterance = |buffer: &SessionBuffer,
                            from_sec: f64,
                            pinned_language: &Option<String>,
                            allow_cancel: bool|
     -> FwResult<crate::native_engine::decode::DecodeOutput> {
        let params = DecodeParams {
            language: pinned_language.clone(),
            timestamps: true,
            // Auto re-enabled (bd-rt-audio-ctx-auto-empty-4c2i fixed): the
            // empty-transcript failure was the last-window rescue requiring
            // has_ts (whisper.cpp keys it on seek coverage alone), and the
            // repetition loops were an under-conditioned encoder below the
            // new AUTO_MIN_ENC_CTX floor. Probe (audio_ctx_auto_probe):
            // Auto tracks Full's text on jfk slices at 2-4x less decode.
            audio_ctx: AudioCtxPolicy::Auto,
            // Live caption stream: bracket-noise markers ([BLANK_AUDIO],
            // [MUSIC]) on trailing-silence windows are chatter, not text.
            suppress_nst: true,
            bypass_transcript_cache: true,
            record_token_attn,
            initial_prompt: buffer.prompt(),
            n_threads: 4,
            ..DecodeParams::default()
        };
        if allow_cancel {
            model.transcribe(buffer.window_from(from_sec), &params, &checkpoint)
        } else {
            // Session-end flush: Ctrl-C already fired; this one bounded
            // decode (buffer is capped) delivers the final words rather
            // than dropping the open utterance.
            model.transcribe(buffer.window_from(from_sec), &params, &noop_checkpoint)
        }
    };
    'session: loop {
        // -- collect finished confirm-lane verdicts (non-blocking) --------
        while let Some(event) =
            confirm_lane.as_ref().and_then(|lane| lane.try_recv())
        {
            handle_confirm_event(
                event,
                &run_id,
                &mut seq,
                &now_ts,
                emit,
                &mut stats,
                &mut pending_confirm_since,
                persist_sink.as_mut(),
            )?;
        }

        // -- termination checks ------------------------------------------
        if is_cancelled() {
            cancelled = true;
        }
        let out_of_time = config.max_seconds > 0.0
            && session_started.elapsed().as_secs_f64() >= config.max_seconds;
        if cancelled || out_of_time || source_ended {
            // Final flush when speech is open.
            if in_speech {
                let slice_from = utterance_t0 + utterance_committed_through;
                let out = decode_utterance(&buffer, slice_from, &pinned_language, false)?;
                let decision = policy.finalize(&out, buffer.session_duration_sec() - slice_from);
                let t1 = buffer.session_duration_sec();
                if !decision.commit_text.is_empty() {
                    if !committed_text.is_empty() {
                        committed_text.push(' ');
                    }
                    committed_text.push_str(&decision.commit_text);
                    delta_count += 1;
                    stats.deltas += 1;
                    if let Some(sink) = persist_sink.as_mut() {
                        sink.record_delta(
                            utterance_t0 + utterance_committed_through,
                            t1,
                            &decision.commit_text,
                        );
                    }
                    emit_seq!(robot::transcript_delta_value(
                        &run_id,
                        seq,
                        &now_ts(),
                        utterance_id,
                        &decision.commit_text,
                        utterance_t0 + utterance_committed_through,
                        t1,
                        decision.commit_tokens,
                        decision.commit_confidence,
                    ));
                }
                stats.utterances += 1;
                emit_seq!(robot::utterance_end_value(
                    &run_id,
                    seq,
                    &now_ts(),
                    utterance_id,
                    UtteranceEndReason::SessionEnd,
                    utterance_t0,
                    t1,
                    &committed_text,
                    delta_count,
                ));
                if let Some(lane) = confirm_lane.as_ref() {
                    pending_confirm_since
                        .insert(utterance_id, std::time::Instant::now());
                    lane.submit(ConfirmJob {
                        utterance_id,
                        pcm: buffer.window_from(utterance_t0).to_vec(),
                        committed_text: committed_text.clone(),
                        language: pinned_language.clone(),
                    });
                }
                persist_flush!();
            }
            break 'session;
        }

        // -- pull audio up to the next step deadline ----------------------
        let wait = next_step.saturating_duration_since(std::time::Instant::now());
        let read = capture.read(&mut raw, wait.min(std::time::Duration::from_millis(100)))?;
        if read.ended {
            source_ended = true;
        }
        stats.capture_overruns = read.overrun_frames_dropped;
        let fresh: &[f32] = &raw[..read.frames * usize::from(capture.channels())];
        let fresh_16k: &[f32] = if let Some(rs) = resampler.as_mut() {
            resampled.clear();
            rs.process(fresh, &mut resampled)?;
            if source_ended {
                rs.flush(&mut resampled);
            }
            &resampled
        } else {
            fresh
        };
        if !fresh_16k.is_empty() {
            if watchdog.observe(fresh_16k) {
                emit_seq!(robot::listen_warning_value(
                    &run_id,
                    seq,
                    &now_ts(),
                    "silent_input",
                    serde_json::json!({
                        "detail": crate::capture::MIC_PERMISSION_REMEDIATION,
                        "leading_silent_sec": 3.0,
                    }),
                ));
            }
            // Streamed session-PCM integrity hash (bd-rt-persist-a66y):
            // fed with the resampled 16 kHz mono stream exactly as heard.
            if let Some(sink) = persist_sink.as_mut() {
                sink.feed_pcm(fresh_16k);
            }
            buffer.push(fresh_16k);
            stats.audio_sec = buffer.session_duration_sec();

            // -- VAD edges -------------------------------------------------
            let edges = if config.vad_enabled {
                vad.push(fresh_16k)
            } else if !in_speech {
                vec![VadEdge::SpeechStarted {
                    t_sec: buffer.session_offset_sec(),
                }]
            } else {
                Vec::new()
            };
            for edge in edges {
                match edge {
                    VadEdge::SpeechStarted { t_sec } => {
                        if !in_speech {
                            in_speech = true;
                            utterance_id += 1;
                            utterance_started_at = session_started.elapsed().as_secs_f64();
                            utterance_t0 = t_sec;
                            committed_text.clear();
                            utterance_committed_through = 0.0;
                            delta_count = 0;
                            partial_generation = 0;
                            policy.reset();
                            emit_seq!(robot::speech_started_value(
                                &run_id,
                                seq,
                                &now_ts(),
                                utterance_id,
                                t_sec,
                            ));
                        }
                    }
                    VadEdge::Endpoint { t_sec } => {
                        if in_speech {
                            let flush_started = std::time::Instant::now();
                            let slice_from = utterance_t0 + utterance_committed_through;
                            let out =
                                match decode_utterance(&buffer, slice_from, &pinned_language, true)
                                {
                                    Err(FwError::Cancelled(_)) => {
                                        cancelled = true;
                                        continue 'session;
                                    }
                                    other => other?,
                                };
                            if pinned_language.is_none() {
                                pinned_language.clone_from(&out.language);
                            }
                            let decision =
                                policy.finalize(&out, buffer.session_duration_sec() - slice_from);
                            if !decision.commit_text.is_empty() {
                                if !committed_text.is_empty() {
                                    committed_text.push(' ');
                                }
                                committed_text.push_str(&decision.commit_text);
                                delta_count += 1;
                                stats.deltas += 1;
                                if stats.ttft_ms.is_none() {
                                    stats.ttft_ms =
                                        Some(session_started.elapsed().as_secs_f64() * 1000.0);
                                }
                                if let Some(sink) = persist_sink.as_mut() {
                                    sink.record_delta(
                                        utterance_t0 + utterance_committed_through,
                                        t_sec,
                                        &decision.commit_text,
                                    );
                                }
                                emit_seq!(robot::transcript_delta_value(
                                    &run_id,
                                    seq,
                                    &now_ts(),
                                    utterance_id,
                                    &decision.commit_text,
                                    utterance_t0 + utterance_committed_through,
                                    t_sec,
                                    decision.commit_tokens,
                                    decision.commit_confidence,
                                ));
                            }
                            stats.utterances += 1;
                            emit_seq!(robot::utterance_end_value(
                                &run_id,
                                seq,
                                &now_ts(),
                                utterance_id,
                                UtteranceEndReason::Endpoint,
                                utterance_t0,
                                t_sec,
                                &committed_text,
                                delta_count,
                            ));
                            tracing::info!(
                                utterance_id,
                                speech_sec = t_sec - utterance_t0,
                                deltas = delta_count,
                                chars = committed_text.len(),
                                endpoint_flush_ms = flush_started.elapsed().as_millis() as u64,
                                reason = "endpoint",
                                "utterance closed"
                            );
                            if let Some(lane) = confirm_lane.as_ref() {
                                pending_confirm_since
                                    .insert(utterance_id, std::time::Instant::now());
                                lane.submit(ConfirmJob {
                                    utterance_id,
                                    pcm: buffer.window_from(utterance_t0).to_vec(),
                                    committed_text: committed_text.clone(),
                                    language: pinned_language.clone(),
                                });
                            }
                            persist_flush!();
                            // Trim + prompt carry.
                            buffer.append_committed_text(&decision.commit_text);
                            buffer.set_committed_through(t_sec);
                            buffer.trim_to_committed();
                            buffer.end_utterance();
                            in_speech = false;
                        }
                    }
                }
            }
        }

        // -- utterance timeout --------------------------------------------
        if in_speech
            && session_started.elapsed().as_secs_f64() - utterance_started_at
                >= config.max_utterance_sec
        {
            let slice_from = utterance_t0 + utterance_committed_through;
            let out = match decode_utterance(&buffer, slice_from, &pinned_language, true) {
                Err(FwError::Cancelled(_)) => {
                    cancelled = true;
                    continue 'session;
                }
                other => other?,
            };
            let decision = policy.finalize(&out, buffer.session_duration_sec() - slice_from);
            let t1 = buffer.session_duration_sec();
            if !decision.commit_text.is_empty() {
                if !committed_text.is_empty() {
                    committed_text.push(' ');
                }
                committed_text.push_str(&decision.commit_text);
                delta_count += 1;
                stats.deltas += 1;
                if let Some(sink) = persist_sink.as_mut() {
                    sink.record_delta(
                        utterance_t0 + utterance_committed_through,
                        t1,
                        &decision.commit_text,
                    );
                }
                emit_seq!(robot::transcript_delta_value(
                    &run_id,
                    seq,
                    &now_ts(),
                    utterance_id,
                    &decision.commit_text,
                    utterance_t0 + utterance_committed_through,
                    t1,
                    decision.commit_tokens,
                    decision.commit_confidence,
                ));
            }
            stats.utterances += 1;
            emit_seq!(robot::utterance_end_value(
                &run_id,
                seq,
                &now_ts(),
                utterance_id,
                UtteranceEndReason::Timeout,
                utterance_t0,
                t1,
                &committed_text,
                delta_count,
            ));
            if let Some(lane) = confirm_lane.as_ref() {
                pending_confirm_since
                    .insert(utterance_id, std::time::Instant::now());
                lane.submit(ConfirmJob {
                    utterance_id,
                    pcm: buffer.window_from(utterance_t0).to_vec(),
                    committed_text: committed_text.clone(),
                    language: pinned_language.clone(),
                });
            }
            persist_flush!();
            buffer.append_committed_text(&decision.commit_text);
            buffer.set_committed_through(t1);
            buffer.trim_to_committed();
            buffer.end_utterance();
            // Forced end mid-speech: next utterance opens immediately.
            utterance_id += 1;
            utterance_started_at = session_started.elapsed().as_secs_f64();
            utterance_t0 = t1;
            committed_text.clear();
            utterance_committed_through = 0.0;
            delta_count = 0;
            partial_generation = 0;
            policy.reset();
            emit_seq!(robot::speech_started_value(
                &run_id,
                seq,
                &now_ts(),
                utterance_id,
                t1
            ));
        }

        // -- buffer cap ----------------------------------------------------
        if let Some(forced) = buffer.enforce_cap() {
            emit_seq!(robot::listen_warning_value(
                &run_id,
                seq,
                &now_ts(),
                "forced_trim",
                serde_json::json!({"dropped_sec": forced.dropped_sec}),
            ));
        }

        // -- step decode (mid-utterance partials) ---------------------------
        if in_speech && std::time::Instant::now() >= next_step {
            let step_started = std::time::Instant::now();
            let slice_from = utterance_t0 + utterance_committed_through;
            let out = match decode_utterance(&buffer, slice_from, &pinned_language, true) {
                Err(FwError::Cancelled(_)) => {
                    cancelled = true;
                    continue 'session;
                }
                other => other?,
            };
            if pinned_language.is_none() {
                pinned_language.clone_from(&out.language);
            }
            let slice_sec = buffer.session_duration_sec() - slice_from;
            let decision = policy.step(&out, slice_sec);
            if decision.holdback {
                stats.policy_holdbacks += 1;
            }
            // Incremental commits (AlignAtt): append-only deltas mid-utterance.
            // Committing ADVANCES the decode-slice origin past the committed
            // audio (prefix-advancing): later decodes never re-transcribe it,
            // so append-only holds by construction; linguistic continuity
            // rides the prompt (append_committed_text below).
            if !decision.commit_text.is_empty() {
                let delta_t0 = slice_from;
                if let Some(through) = decision.commit_through_sec {
                    utterance_committed_through = (slice_from - utterance_t0) + through;
                }
                let delta_t1 = utterance_t0 + utterance_committed_through;
                buffer.append_committed_text(&decision.commit_text);
                if !committed_text.is_empty() {
                    committed_text.push(' ');
                }
                committed_text.push_str(&decision.commit_text);
                delta_count += 1;
                stats.deltas += 1;
                if stats.ttft_ms.is_none() {
                    stats.ttft_ms = Some(session_started.elapsed().as_secs_f64() * 1000.0);
                }
                if let Some(sink) = persist_sink.as_mut() {
                    sink.record_delta(delta_t0, delta_t1, &decision.commit_text);
                }
                emit_seq!(robot::transcript_delta_value(
                    &run_id,
                    seq,
                    &now_ts(),
                    utterance_id,
                    &decision.commit_text,
                    delta_t0,
                    delta_t1,
                    decision.commit_tokens,
                    decision.commit_confidence,
                ));
            }
            let elapsed_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            step_latencies_ms.push(elapsed_ms);
            tracing::debug!(
                step = step_latencies_ms.len(),
                buffer_sec = buffer.buffer_sec(),
                decoded_ms = elapsed_ms as u64,
                partial_chars = decision.partial_tail.as_deref().map_or(0, str::len),
                "listen step"
            );
            if config.emit_partials
                && let Some(tail) = decision.partial_tail
            {
                partial_generation += 1;
                let segment = crate::model::TranscriptionSegment {
                    start_sec: Some(utterance_t0),
                    end_sec: Some(buffer.session_duration_sec()),
                    text: tail,
                    speaker: None,
                    confidence: None,
                };
                let mut value = robot::transcript_partial_value(&run_id, seq, &now_ts(), &segment);
                value["utterance_id"] = serde_json::json!(utterance_id);
                value["generation"] = serde_json::json!(partial_generation);
                emit_seq!(value);
            }
            // Skip logic: schedule from completion (an overrunning decode
            // skips ticks instead of queueing them).
            next_step =
                std::time::Instant::now() + std::time::Duration::from_millis(config.step_ms);
        } else if !in_speech {
            next_step =
                std::time::Instant::now() + std::time::Duration::from_millis(config.step_ms);
        }

        // -- periodic stats heartbeat ---------------------------------------
        if config.stats_interval_sec > 0.0
            && last_stats_emit.elapsed().as_secs_f64() >= config.stats_interval_sec
        {
            fill_step_stats(&mut stats, &step_latencies_ms, session_started, &buffer);
            emit_seq!(robot::listen_session_stats_value(
                &run_id,
                seq,
                &now_ts(),
                &stats,
                false
            ));
            last_stats_emit = std::time::Instant::now();
        }
    }

    // Terminal: confirm-lane drain (bd-rt-confirm-lane-3okr) — a bounded
    // wait so in-flight quality decodes land before history freezes; then
    // the final stats. run_error is the fatal terminal and is emitted by
    // the CLI wrapper when this function errors.
    if let Some(lane) = confirm_lane.as_ref() {
        let (events, abandoned) = lane.drain(config.confirm_drain_sec);
        for event in events {
            handle_confirm_event(
                event,
                &run_id,
                &mut seq,
                &now_ts,
                emit,
                &mut stats,
                &mut confirm_lag_ms,
                &mut pending_confirm_since,
                persist_sink.as_mut(),
            )?;
        }
        if abandoned > 0 {
            emit(robot::listen_warning_value(
                &run_id,
                seq,
                &now_ts(),
                "confirm_drain_timeout",
                serde_json::json!({
                    "detail": "session ended before the quality lane drained",
                    "abandoned_utterances": abandoned,
                    "drain_sec": config.confirm_drain_sec,
                }),
            ))?;
            seq += 1;
        }
    }
    if !confirm_lag_ms.is_empty() {
        let mut sorted = confirm_lag_ms.clone();
        sorted.sort_by(f64::total_cmp);
        stats.confirm_lag_p50_ms = Some(sorted[sorted.len() / 2]);
        stats.confirm_lag_p95_ms =
            Some(sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)]);
        stats.confirm_lag_max_ms = sorted.last().copied();
    }
    fill_step_stats(&mut stats, &step_latencies_ms, session_started, &buffer);
    // Final stats event is recorded into the sink BEFORE close so the
    // persisted event trail ends exactly like the stdout stream.
    let final_stats_value =
        robot::listen_session_stats_value(&run_id, seq, &now_ts(), &stats, true);
    if let Some(sink) = persist_sink.as_mut() {
        sink.record_event(seq, &now_ts(), "listen.session_stats", &final_stats_value);
    }
    // Close the run: true end time + final stats/warnings + finalized
    // replay envelope. Failure here surfaces as run_error AFTER the session
    // events were emitted (never silently lost).
    let persist_close_error = match persist_sink.take() {
        Some(sink) => sink
            .close(
                &serde_json::to_string(&stats).unwrap_or_else(|_| "{}".to_owned()),
                &persist_warnings_to_json(&persist_warnings),
                &now_ts(),
            )
            .err(),
        None => None,
    };
    emit(final_stats_value)?;
    seq += 1;
    if let Some(error) = persist_close_error {
        return Err(FwError::Storage(format!(
            "listen persistence failed on final close: {error}"
        )));
    }
    capture.stop();
    Ok(cancelled)
}

/// Serialize the collected warning payloads for `runs.warnings_json`.
fn persist_warnings_to_json(warnings: &[serde_json::Value]) -> String {
    serde_json::Value::Array(warnings.to_vec()).to_string()
}
    capture.stop();
    Ok(cancelled)
}

fn fill_step_stats(
    stats: &mut ListenSessionStats,
    step_latencies_ms: &[f64],
    session_started: std::time::Instant,
    buffer: &SessionBuffer,
) {
    stats.wall_sec = session_started.elapsed().as_secs_f64();
    stats.audio_sec = buffer.session_duration_sec();
    if !step_latencies_ms.is_empty() {
        stats.mean_step_latency_ms =
            step_latencies_ms.iter().sum::<f64>() / step_latencies_ms.len() as f64;
        let mut sorted = step_latencies_ms.to_vec();
        sorted.sort_by(f64::total_cmp);
        stats.p95_step_latency_ms = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
    }
}
/// Map one confirm-lane event to its robot NDJSON emission + session-stat
/// updates (bd-rt-confirm-lane-3okr). Shared by the in-loop drain and the
/// terminal drain so ordering/seq discipline is identical on both paths.
#[allow(clippy::too_many_arguments)]
fn handle_confirm_event(
    event: ConfirmLaneEvent,
    run_id: &str,
    seq: &mut u64,
    now_ts: &dyn Fn() -> String,
    emit: &mut dyn FnMut(serde_json::Value) -> FwResult<()>,
    stats: &mut ListenSessionStats,
    lag_samples_ms: &mut Vec<f64>,
    pending_since: &mut std::collections::HashMap<u32, std::time::Instant>,
    sink: &mut Option<ListenPersistSink>,
) -> FwResult<()> {
    // Record-then-emit so the persisted trail matches stdout exactly.
    match &event {
        ConfirmLaneEvent::Verdict(verdict) => {
            if let Some(s) = sink.as_mut() {
                let value = if verdict.confirmed {
                    robot::listen_transcript_confirm_value(
                        run_id,
                        *seq,
                        &now_ts(),
                        verdict.utterance_id,
                        &verdict.quality_model_id,
                        verdict.drift_wer,
                        verdict.drift_confidence_delta,
                        verdict.drift_edit_distance,
                        verdict.decode_ms,
                    )
                } else {
                    robot::listen_transcript_correct_value(
                        run_id,
                        *seq,
                        &now_ts(),
                        verdict.utterance_id,
                        verdict.correction_id,
                        &verdict.corrected_segments,
                        &verdict.quality_model_id,
                        verdict.drift_wer,
                        verdict.drift_confidence_delta,
                        verdict.drift_edit_distance,
                        verdict.decode_ms,
                    )
                };
                s.record_event(*seq, &now_ts(), value["event"].as_str().unwrap_or(""), &value);
            }
        }
        ConfirmLaneEvent::Warning { reason, detail } => {
            if let Some(s) = sink.as_mut() {
                let value = robot::listen_warning_value(run_id, *seq, &now_ts(), reason, detail.clone());
                s.record_event(*seq, &now_ts(), "listen.warning", &value);
            }
        }
        ConfirmLaneEvent::DroppedOldest { utterance_id, queue_bound } => {
            if let Some(s) = sink.as_mut() {
                let value = robot::listen_warning_value(
                    run_id,
                    *seq,
                    &now_ts(),
                    "confirm_lag",
                    serde_json::json!({
                        "detail": "confirm queue full; dropped oldest unconfirmed utterance",
                        "dropped_utterance_id": utterance_id,
                        "queue_bound": queue_bound,
                    }),
                );
                s.record_event(*seq, &now_ts(), "listen.warning", &value);
            }
        }
    }
    match event {
        ConfirmLaneEvent::Verdict(verdict) => {
            // Confirm lag = utterance_end -> verdict arrival (observation
            // for the latency harness; not a ledger claim).
            if let Some(started) = pending_since.remove(&verdict.utterance_id) {
                lag_samples_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            }
            if verdict.confirmed {
                stats.confirmations_emitted += 1;
                emit(robot::listen_transcript_confirm_value(
                    run_id,
                    *seq,
                    &now_ts(),
                    verdict.utterance_id,
                    &verdict.quality_model_id,
                    verdict.drift_wer,
                    verdict.drift_confidence_delta,
                    verdict.drift_edit_distance,
                    verdict.decode_ms,
                ))?;
            } else {
                stats.corrections_emitted += 1;
                emit(robot::listen_transcript_correct_value(
                    run_id,
                    *seq,
                    &now_ts(),
                    verdict.utterance_id,
                    verdict.correction_id,
                    &verdict.corrected_segments,
                    &verdict.quality_model_id,
                    verdict.drift_wer,
                    verdict.drift_confidence_delta,
                    verdict.drift_edit_distance,
                    verdict.decode_ms,
                ))?;
            }
            *seq += 1;
        }
        ConfirmLaneEvent::Warning { reason, detail } => {
            emit(robot::listen_warning_value(
                run_id,
                *seq,
                &now_ts(),
                &reason,
                detail,
            ))?;
            *seq += 1;
        }
        ConfirmLaneEvent::DroppedOldest {
            utterance_id,
            queue_bound,
        } => {
            pending_since.remove(&utterance_id);
            emit(robot::listen_warning_value(
                run_id,
                *seq,
                &now_ts(),
                "confirm_lag",
                serde_json::json!({
                    "detail": "confirm queue full; dropped oldest unconfirmed utterance",
                    "dropped_utterance_id": utterance_id,
                    "queue_bound": queue_bound,
                }),
            ))?;
            *seq += 1;
        }
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn cfg() -> SessionBufferConfig {
        SessionBufferConfig::default()
    }

    fn secs(buffer: &SessionBuffer) -> (f64, f64) {
        (buffer.session_offset_sec(), buffer.buffer_sec())
    }

    #[test]
    fn time_mapping_is_exact_across_pushes_and_trims() {
        let mut b = SessionBuffer::new(cfg());
        b.push(&vec![0.0; SAMPLE_RATE * 3]); // 3 s
        assert_eq!(secs(&b), (0.0, 3.0));
        b.set_committed_through(2.0);
        let trimmed = b.trim_to_committed();
        // Trim to 2.0 - 0.2 keep-back = 1.8 s.
        assert!((trimmed - 1.8).abs() < 1e-9, "trimmed {trimmed}");
        assert!((b.session_offset_sec() - 1.8).abs() < 1e-9);
        // A window-relative timestamp re-bases exactly.
        assert!((b.session_time_of(0.5) - 2.3).abs() < 1e-9);
        b.push(&vec![0.0; SAMPLE_RATE]); // +1 s
        assert!((b.session_duration_sec() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn property_random_push_trim_keeps_clock_consistent() {
        // Deterministic pseudo-random walk (no rand dep): LCG.
        let mut state: u64 = 0x1234_5678;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut b = SessionBuffer::new(cfg());
        let mut pushed: u64 = 0;
        for _ in 0..200 {
            let n = next() % (SAMPLE_RATE / 2) + 1;
            b.push(&vec![0.0; n]);
            pushed += n as u64;
            if next() % 3 == 0 {
                let commit_sec = (pushed as f64 / SAMPLE_RATE as f64) * 0.8;
                b.set_committed_through(commit_sec);
                b.trim_to_committed();
            }
            let _ = b.enforce_cap();
            // Invariant: offset + buffered == total pushed, exactly.
            let total = b.session_offset_sec() + b.buffer_sec();
            let expected = pushed as f64 / SAMPLE_RATE as f64;
            assert!(
                (total - expected).abs() < 1e-9,
                "clock drift: offset+buffer={total} expected={expected}"
            );
        }
    }

    #[test]
    fn cap_enforcement_force_trims_and_reports() {
        let mut b = SessionBuffer::new(cfg());
        b.push(&vec![0.0; SAMPLE_RATE * 15]); // 15 s > 12 s cap
        let forced = b.enforce_cap().expect("cap must force a trim");
        assert!((forced.dropped_sec - 3.0).abs() < 1e-9);
        assert!((b.buffer_sec() - 12.0).abs() < 1e-9);
        assert_eq!(b.forced_trims(), 1);
        // Under cap: no-op.
        assert!(b.enforce_cap().is_none());
    }

    #[test]
    fn keep_back_margin_is_honored() {
        let mut b = SessionBuffer::new(cfg());
        b.push(&vec![0.0; SAMPLE_RATE * 5]);
        b.set_committed_through(3.0);
        b.trim_to_committed();
        // Buffer must still start 200 ms BEFORE the committed watermark.
        assert!((b.session_offset_sec() - 2.8).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // bd-rt-confirm-lane-3okr: confirm-lane unit tests (injected decoder,
    // no models on disk required)
    // ------------------------------------------------------------------

    fn fake_job(id: u32, text: &str) -> ConfirmJob {
        ConfirmJob {
            utterance_id: id,
            pcm: vec![0.0; 1600],
            committed_text: text.to_owned(),
            language: Some("en".to_owned()),
        }
    }

    fn segment(text: &str) -> crate::model::TranscriptionSegment {
        crate::model::TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: None,
            text: text.to_owned(),
            speaker: None,
            confidence: None,
        }
    }

    /// Collect lane events until `want` arrived or the deadline expired.
    fn collect(lane: &ConfirmLane, want: usize, secs: u64) -> Vec<ConfirmLaneEvent> {
        let deadline = Instant::now() + Duration::from_secs(secs);
        let mut events = Vec::new();
        while events.len() < want && Instant::now() < deadline {
            match lane.try_recv() {
                Some(event) => events.push(event),
                None => std::thread::sleep(Duration::from_millis(5)),
            }
        }
        events
    }

    #[test]
    fn confirm_lane_confirms_match_and_pins_language_and_prompt() {
        type SeenLog =
            std::sync::Arc<std::sync::Mutex<Vec<(Option<String>, String, String)>>>;
        let seen: SeenLog = std::sync::Arc::default();
        let seen_for_decoder = std::sync::Arc::clone(&seen);
        let decoder: QualityDecoder = Box::new(move |job, prev_confirmed, _abort| {
            seen_for_decoder
                .lock()
                .expect("seen lock")
                .push((
                    job.language.clone(),
                    prev_confirmed.to_owned(),
                    job.committed_text.clone(),
                ));
            DecodeOutcome::Segments(
                vec![segment(&job.committed_text)],
                "fake-qm".to_owned(),
            )
        });
        let lane = ConfirmLane::spawn(4, decoder);
        lane.submit(fake_job(1, "hello world"));
        lane.submit(fake_job(2, "second utterance"));
        let events = collect(&lane, 2, 5);
        let verdicts: Vec<&ConfirmVerdict> = events
            .iter()
            .filter_map(|e| match e {
                ConfirmLaneEvent::Verdict(v) => Some(v),
                _ => None,
            })
            .collect();
        assert_eq!(verdicts.len(), 2, "both utterances confirm: {events:?}");
        assert!(verdicts.iter().all(|v| v.confirmed));
        assert_eq!(verdicts[0].quality_model_id, "fake-qm");
        // Language pinning: the quality decode receives the fast lane's
        // pinned language, not a re-detection.
        let seen = seen.lock().expect("seen lock");
        assert_eq!(seen[0].0.as_deref(), Some("en"));
        assert_eq!(seen[1].0.as_deref(), Some("en"));
        // Prompt continuity: first job has no prior confirmed text; the
        // second carries the first's confirmed text.
        assert!(seen[0].1.is_empty());
        assert_eq!(seen[1].1, "hello world");
        let (_, abandoned) = lane.drain(0.05);
        assert_eq!(abandoned, 0);
    }

    #[test]
    fn confirm_lane_corrects_beyond_wer_tolerance() {
        let decoder: QualityDecoder = Box::new(move |_job, _prev, _abort| {
            DecodeOutcome::Segments(
                vec![segment(
                    "the quick brown fox jumps over the lazy dog and then \
                     keeps running far beyond every expectation",
                )],
                "fake-qm".to_owned(),
            )
        });
        let lane = ConfirmLane::spawn(4, decoder);
        lane.submit(fake_job(1, "hello"));
        let events = collect(&lane, 1, 5);
        assert_eq!(events.len(), 1, "expected exactly one event: {events:?}");
        match &events[0] {
            ConfirmLaneEvent::Verdict(v) => {
                assert!(!v.confirmed, "divergent text must correct");
                assert!(!v.corrected_segments.is_empty());
                assert!(v.drift_wer > 0.1, "wer drift {} must exceed tolerance", v.drift_wer);
            }
            other => panic!("expected verdict, got {other:?}"),
        }
    }

    #[test]
    fn queue_bound_drops_oldest_with_warning_event() {
        let decoder: QualityDecoder = Box::new(move |job, _prev, _abort| {
            std::thread::sleep(Duration::from_millis(50));
            DecodeOutcome::Segments(
                vec![segment(&job.committed_text)],
                "fake-qm".to_owned(),
            )
        });
        let lane = ConfirmLane::spawn(4, decoder);
        for id in 1..=6 {
            lane.submit(fake_job(id, "text"));
        }
        let events = collect(&lane, 8, 10);
        let dropped: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                ConfirmLaneEvent::DroppedOldest { utterance_id, .. } => Some(*utterance_id),
                _ => None,
            })
            .collect();
        assert_eq!(dropped, vec![1, 2], "oldest unconfirmed jobs drop first");
        let verdict_ids: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                ConfirmLaneEvent::Verdict(v) => Some(v.utterance_id),
                _ => None,
            })
            .collect();
        assert_eq!(verdict_ids, vec![3, 4, 5, 6]);
        let (_, abandoned) = lane.drain(0.05);
        assert_eq!(abandoned, 0);
    }

    #[test]
    fn quality_model_unavailable_disables_lane_without_killing_it() {
        let decoder: QualityDecoder = Box::new(move |_job, _prev, _abort| {
            DecodeOutcome::Unavailable("package not installed".to_owned())
        });
        let lane = ConfirmLane::spawn(4, decoder);
        lane.submit(fake_job(1, "one"));
        lane.submit(fake_job(2, "two"));
        let events = collect(&lane, 1, 5);
        let warnings: Vec<(&str, &serde_json::Value)> = events
            .iter()
            .filter_map(|e| match e {
                ConfirmLaneEvent::Warning { reason, detail } => Some((reason.as_str(), detail)),
                _ => None,
            })
            .collect();
        assert_eq!(warnings.len(), 1, "exactly one unavailable warning");
        assert_eq!(warnings[0].0, "quality_model_unavailable");
        // No verdicts may follow a load failure; queued jobs are skipped.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ConfirmLaneEvent::Verdict(_))),
            "lane disabled: no verdicts"
        );
        // The handle still drains cleanly (session keeps running fast-only).
        let (extra, abandoned) = lane.drain(0.05);
        assert_eq!(abandoned, 0);
        assert!(!extra
            .iter()
            .any(|e| matches!(e, ConfirmLaneEvent::Verdict(_))));
    }

    #[test]
    fn drain_abandons_stuck_decode_within_deadline() {
        // Decoder honors the abort flag only on its poll loop — mirrors the
        // real checkpoint behavior inside transcribe.
        let decoder: QualityDecoder = Box::new(move |_job, _prev, is_abort| loop {
            if is_abort() {
                return DecodeOutcome::Aborted;
            }
            std::thread::sleep(Duration::from_millis(20));
        });
        let lane = ConfirmLane::spawn(4, decoder);
        lane.submit(fake_job(1, "stuck"));
        let started = Instant::now();
        let (events, abandoned) = lane.drain(0.2);
        assert!(started.elapsed() < Duration::from_secs(3), "drain must be bounded");
        assert!(events.is_empty(), "no verdict from an abandoned decode");
        assert!(abandoned >= 1, "the in-flight job counts as abandoned");
    }

    #[test]
    fn confirm_lane_resolution_matrix_excludes_self_confirmation() {
        use QualityModelSetting::{Auto, Disabled, Explicit};
        // Explicit opt-out is always off.
        assert_eq!(resolve_confirm_lane(&Disabled, "tiny.en"), None);
        // Fast lane fell back to turbo (label says so): Auto must NOT
        // self-confirm against itself.
        assert_eq!(resolve_confirm_lane(&Auto, CONFIRM_QUALITY_MODEL_DEFAULT), None);
        // Explicit spec identical to the effective fast model: off.
        assert_eq!(
            resolve_confirm_lane(&Explicit("tiny.en".to_owned()), "tiny.en"),
            None
        );
        // Explicit spec that cannot resolve anywhere: off (never downloads).
        assert_eq!(
            resolve_confirm_lane(
                &Explicit("no-such-model-anywhere".to_owned()),
                "tiny.en"
            ),
            None
        );
        // NOTE: the `Auto` + turbo-installed branch is intentionally NOT
        // asserted here — it depends on whether the release package is
        // cached on this machine; the model-gated e2e covers that shape.
    }

    #[test]
    fn min_tail_survives_aggressive_trims_after_long_silence() {
        let mut b = SessionBuffer::new(cfg());
        b.push(&vec![0.0; SAMPLE_RATE * 10]);
        // Watermark at the very end (everything committed / silence).
        b.set_committed_through(10.0);
        b.trim_to_committed();
        // At least min_tail (0.2 s) must remain for VAD pre-pad lookback.
        assert!(
            b.buffer_sec() >= 0.2 - 1e-9,
            "tail {} below min retention",
            b.buffer_sec()
        );
    }

    #[test]
    fn watermark_is_monotonic_and_clamped() {
        let mut b = SessionBuffer::new(cfg());
        b.push(&vec![0.0; SAMPLE_RATE * 2]);
        b.set_committed_through(1.5);
        b.set_committed_through(1.0); // earlier: ignored
        b.trim_to_committed();
        assert!((b.session_offset_sec() - 1.3).abs() < 1e-9);
        // Beyond buffered audio: clamped to end.
        b.set_committed_through(99.0);
        b.trim_to_committed();
        assert!(b.buffer_sec() >= 0.2 - 1e-9);
    }

    #[test]
    fn chunk_size_invariance_metamorphic() {
        // Same PCM in different push sizes -> identical state at equal times.
        let pcm: Vec<f32> = (0..SAMPLE_RATE * 4).map(|i| (i as f32).sin()).collect();
        let build = |chunks: &[usize]| {
            let mut b = SessionBuffer::new(cfg());
            let mut pos = 0;
            for &n in chunks {
                let end = (pos + n).min(pcm.len());
                b.push(&pcm[pos..end]);
                pos = end;
            }
            b.push(&pcm[pos..]);
            b.set_committed_through(2.0);
            b.trim_to_committed();
            b
        };
        let a = build(&[160; 100]);
        let c = build(&[1024, 3, 7777, 1]);
        assert_eq!(a.window(), c.window());
        assert_eq!(a.session_offset_sec(), c.session_offset_sec());
    }

    #[test]
    fn long_session_through_cap_keeps_offsets_exact() {
        let mut b = SessionBuffer::new(cfg());
        for _ in 0..60 {
            b.push(&vec![0.0; SAMPLE_RATE]); // 1 s at a time
            let _ = b.enforce_cap();
            assert!(b.buffer_sec() <= 12.0 + 1e-9);
        }
        assert!((b.session_duration_sec() - 60.0).abs() < 1e-9);
        assert!((b.session_offset_sec() - 48.0).abs() < 1e-9);
    }

    #[test]
    fn allocation_stability_after_warmup() {
        let mut b = SessionBuffer::new(cfg());
        b.push(&vec![0.0; SAMPLE_RATE * 12]);
        let _ = b.enforce_cap();
        let cap_before = b.samples.capacity();
        for _ in 0..50 {
            b.push(&vec![0.0; SAMPLE_RATE / 2]);
            b.set_committed_through(b.session_duration_sec() - 0.5);
            b.trim_to_committed();
            let _ = b.enforce_cap();
        }
        assert_eq!(
            b.samples.capacity(),
            cap_before,
            "steady-state operation must not reallocate"
        );
    }

    // -- prompt carry -------------------------------------------------------

    #[test]
    fn prompt_assembles_current_utterance_text() {
        let mut b = SessionBuffer::new(cfg());
        b.append_committed_text("hello ");
        b.append_committed_text("world");
        assert_eq!(b.prompt().as_deref(), Some("hello world"));
    }

    #[test]
    fn prompt_carries_previous_utterance_tail_across_end_utterance() {
        let mut b = SessionBuffer::new(cfg());
        b.append_committed_text("first utterance text");
        b.end_utterance();
        assert_eq!(b.prompt().as_deref(), Some("first utterance text"));
        b.append_committed_text("second begins");
        let p = b.prompt().expect("prompt");
        assert!(p.starts_with("first utterance text"), "got: {p}");
        assert!(p.ends_with("second begins"), "got: {p}");
    }

    #[test]
    fn prompt_front_truncates_at_word_boundary() {
        let small = SessionBufferConfig {
            prompt_cap_chars: 12,
            ..SessionBufferConfig::default()
        };
        let mut b = SessionBuffer::new(small);
        b.append_committed_text("alpha beta gamma delta");
        let p = b.prompt().expect("prompt");
        assert!(p.chars().count() <= 12, "cap exceeded: {p:?}");
        // Never starts mid-word: the surviving text starts at a word start.
        assert!(
            "alpha beta gamma delta".contains(&format!(" {p}"))
                || "alpha beta gamma delta".starts_with(&p),
            "mid-word start: {p:?}"
        );
    }

    #[test]
    fn cross_utterance_tail_respects_its_own_cap() {
        let c = SessionBufferConfig {
            cross_utterance_tail_chars: 10,
            ..SessionBufferConfig::default()
        };
        let mut b = SessionBuffer::new(c);
        b.append_committed_text("one two three four five six");
        b.end_utterance();
        let p = b.prompt().expect("prompt");
        assert!(p.chars().count() <= 10, "tail cap exceeded: {p:?}");
    }

    #[test]
    fn no_context_disables_all_carry() {
        let c = SessionBufferConfig {
            prompt_carry: false,
            ..SessionBufferConfig::default()
        };
        let mut b = SessionBuffer::new(c);
        b.append_committed_text("ignored");
        b.end_utterance();
        b.append_committed_text("also ignored");
        assert_eq!(b.prompt(), None);
    }

    #[test]
    fn empty_prompt_is_none_not_empty_string() {
        let b = SessionBuffer::new(cfg());
        assert_eq!(b.prompt(), None);
    }

    #[test]
    fn multibyte_text_never_splits_in_prompt_paths() {
        let c = SessionBufferConfig {
            prompt_cap_chars: 5,
            cross_utterance_tail_chars: 5,
            ..SessionBufferConfig::default()
        };
        let mut b = SessionBuffer::new(c);
        b.append_committed_text("héllo wörld émoji");
        b.end_utterance();
        // Must not panic on char boundaries; result respects the char cap.
        if let Some(p) = b.prompt() {
            assert!(p.chars().count() <= 5, "cap exceeded: {p:?}");
        }
    }

    // -----------------------------------------------------------------------
    // bd-rt-vad-stream-ulp5: StreamingVad
    // -----------------------------------------------------------------------

    /// Frames of loudness `db` dBFS as raw samples (constant amplitude).
    fn frames_at_db(db: f64, frames: usize) -> Vec<f32> {
        let amp = 10f64.powf(db / 20.0) as f32;
        vec![amp; frames * VAD_FRAME_SAMPLES]
    }

    fn edges_of(vad: &mut StreamingVad, audio: &[f32]) -> Vec<VadEdge> {
        vad.push(audio)
    }

    #[test]
    fn vad_opens_after_min_speech_and_closes_after_endpoint_silence() {
        let mut vad = StreamingVad::new(StreamingVadConfig::default());
        // 1 s of quiet establishes the floor (~-60 dB).
        assert!(edges_of(&mut vad, &frames_at_db(-60.0, 50)).is_empty());
        // 300 ms of speech-level audio (well above floor + 9 dB gate).
        let edges = edges_of(&mut vad, &frames_at_db(-20.0, 15));
        assert_eq!(edges.len(), 1, "expected SpeechStarted, got {edges:?}");
        let VadEdge::SpeechStarted { t_sec } = edges[0] else {
            panic!("expected SpeechStarted, got {edges:?}");
        };
        // Onset = first voiced frame (t=1.0s) minus 200 ms pre-pad.
        assert!((t_sec - 0.8).abs() < 0.021, "onset {t_sec}");
        assert!(vad.in_speech());
        // 700 ms of silence closes it (endpoint at 600 ms).
        let edges = edges_of(&mut vad, &frames_at_db(-60.0, 35));
        assert_eq!(edges.len(), 1, "expected Endpoint, got {edges:?}");
        let VadEdge::Endpoint { t_sec } = edges[0] else {
            panic!("expected Endpoint, got {edges:?}");
        };
        // Last voiced frame ended at 1.0 + 0.3 = 1.3 s.
        assert!((t_sec - 1.3).abs() < 0.021, "endpoint {t_sec}");
        assert!(!vad.in_speech());
    }

    #[test]
    fn vad_short_blip_below_min_speech_never_opens() {
        let mut vad = StreamingVad::new(StreamingVadConfig::default());
        let _ = edges_of(&mut vad, &frames_at_db(-60.0, 50));
        // 100 ms blip (5 frames) < 250 ms min-speech.
        let mut edges = edges_of(&mut vad, &frames_at_db(-20.0, 5));
        edges.extend(edges_of(&mut vad, &frames_at_db(-60.0, 50)));
        assert!(
            edges.is_empty(),
            "blip must not open an utterance: {edges:?}"
        );
    }

    #[test]
    fn vad_grace_frames_bridge_glottal_gaps() {
        let mut vad = StreamingVad::new(StreamingVadConfig::default());
        let _ = edges_of(&mut vad, &frames_at_db(-60.0, 50));
        // Voiced run with single-frame dips every 4 frames: grace (2) must
        // bridge them so the run still accumulates to min-speech.
        let mut audio = Vec::new();
        for _ in 0..5 {
            audio.extend(frames_at_db(-20.0, 4));
            audio.extend(frames_at_db(-60.0, 1));
        }
        let edges = edges_of(&mut vad, &audio);
        assert!(
            edges
                .iter()
                .any(|e| matches!(e, VadEdge::SpeechStarted { .. })),
            "gapped speech must still open: {edges:?}"
        );
    }

    #[test]
    fn vad_noise_floor_readapts_to_louder_environment_within_2s() {
        let mut vad = StreamingVad::new(StreamingVadConfig::default());
        // Quiet room.
        let _ = edges_of(&mut vad, &frames_at_db(-60.0, 50));
        // Environment jumps to a steady -35 dB hum. Initially this reads as
        // voiced (it clears -60+9), but the floor rises at 10 dB/s in
        // silence-adjacent states; after ~3 s of sustained hum with no
        // dynamics the machine must NOT be reporting speech anymore.
        let mut edges = Vec::new();
        for _ in 0..150 {
            edges.extend(vad.push(&frames_at_db(-35.0, 1)));
        }
        // The hum may have transiently opened an utterance; it must have
        // closed again (floor caught up, frames stopped being voiced).
        // Wait out an endpoint window at the hum level.
        for _ in 0..40 {
            edges.extend(vad.push(&frames_at_db(-35.0, 1)));
        }
        assert!(
            !vad.in_speech(),
            "floor must adapt to steady hum: {edges:?}"
        );
        assert!(
            vad.noise_floor_db() > -45.0,
            "floor {} too low",
            vad.noise_floor_db()
        );
    }

    #[test]
    fn vad_digital_silence_never_voiced_despite_low_floor() {
        let mut vad = StreamingVad::new(StreamingVadConfig::default());
        // Pure zeros push the floor to the -90 clamp; a -58 dB whisper-of-a
        // -signal clears floor+gate but sits under min_voice_dbfs (-55):
        // must NOT count as voice.
        let _ = edges_of(&mut vad, &vec![0.0f32; VAD_FRAME_SAMPLES * 50]);
        let edges = edges_of(&mut vad, &frames_at_db(-58.0, 20));
        assert!(edges.is_empty(), "sub-threshold audio opened: {edges:?}");
    }

    #[test]
    fn vad_chunk_size_invariance_metamorphic() {
        let mut audio = frames_at_db(-60.0, 50);
        audio.extend(frames_at_db(-20.0, 20));
        audio.extend(frames_at_db(-60.0, 40));
        audio.extend(frames_at_db(-18.0, 25));
        audio.extend(frames_at_db(-60.0, 40));

        let run = |chunk: usize| {
            let mut vad = StreamingVad::new(StreamingVadConfig::default());
            let mut edges = Vec::new();
            for c in audio.chunks(chunk) {
                edges.extend(vad.push(c));
            }
            edges
        };
        let a = run(VAD_FRAME_SAMPLES); // exact frames
        let b = run(7); // pathological tiny
        let c = run(4096); // large
        assert_eq!(a, b, "chunk-size variance (320 vs 7)");
        assert_eq!(a, c, "chunk-size variance (320 vs 4096)");
        assert_eq!(
            a.iter()
                .filter(|e| matches!(e, VadEdge::SpeechStarted { .. }))
                .count(),
            2,
            "expected two utterances: {a:?}"
        );
    }

    #[test]
    fn vad_pre_pad_clamps_to_previous_endpoint() {
        let cfg = StreamingVadConfig {
            endpoint_ms: 200,
            pre_pad_ms: 400,
            ..StreamingVadConfig::default()
        };
        let mut vad = StreamingVad::new(cfg);
        let mut edges = Vec::new();
        edges.extend(vad.push(&frames_at_db(-60.0, 25)));
        edges.extend(vad.push(&frames_at_db(-20.0, 15)));
        edges.extend(vad.push(&frames_at_db(-60.0, 12))); // 240ms silence -> endpoint
        edges.extend(vad.push(&frames_at_db(-20.0, 15)));
        let starts: Vec<f64> = edges
            .iter()
            .filter_map(|e| match e {
                VadEdge::SpeechStarted { t_sec } => Some(*t_sec),
                VadEdge::Endpoint { .. } => None,
            })
            .collect();
        let ends: Vec<f64> = edges
            .iter()
            .filter_map(|e| match e {
                VadEdge::Endpoint { t_sec } => Some(*t_sec),
                VadEdge::SpeechStarted { .. } => None,
            })
            .collect();
        assert_eq!(starts.len(), 2, "{edges:?}");
        assert_eq!(ends.len(), 1, "{edges:?}");
        // Second utterance's 400 ms pre-pad would reach into the first
        // utterance; it must clamp at the endpoint time.
        assert!(
            starts[1] >= ends[0] - 1e-9,
            "pre-pad crossed the previous endpoint: start {} < end {}",
            starts[1],
            ends[0]
        );
    }

    #[test]
    fn vad_classifier_veto_blocks_energy_passing_frames() {
        struct RejectAll;
        impl VoiceClassifier for RejectAll {
            fn is_voice(&mut self, _frame: &[f32]) -> bool {
                false
            }
        }
        let mut vad =
            StreamingVad::new(StreamingVadConfig::default()).with_classifier(Box::new(RejectAll));
        let _ = vad.push(&frames_at_db(-60.0, 50));
        let edges = vad.push(&frames_at_db(-20.0, 30));
        assert!(edges.is_empty(), "vetoed frames opened an utterance");
        assert!(vad.classifier_vetoes() > 0);
    }

    #[test]
    fn vad_cost_is_trivial_per_step() {
        // Not a ledger claim — an order-of-magnitude guard: 30 s of audio
        // through the VAD must be far under a driver step budget.
        let audio = frames_at_db(-30.0, 1500); // 30 s
        let mut vad = StreamingVad::new(StreamingVadConfig::default());
        let started = Instant::now();
        let _ = vad.push(&audio);
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "VAD too slow: {:?}",
            started.elapsed()
        );
    }

    // -----------------------------------------------------------------------
    // bd-rt-vad-stream-ulp5: earshot second-tier EVALUATION (VERDICT: NO)
    //
    // Decision rule (locked in the bead): ship energy pre-gate + earshot
    // only if the classifier measurably vetoes loud non-speech without
    // hurting speech recall. Measured result (2026-08-22, release build):
    // harmonic music at -14 dBFS scores mean 0.886 / max 0.958 — 99.5% of
    // frames PASS the 0.5 voice threshold and still 80% pass at 0.85;
    // raising the cutoff to 0.95 finally vetoes the music but kills half
    // the real speech (jfk.wav mask-recall 0.50). No separating threshold
    // exists. Hum/noise rejection IS excellent (0.00 / 0.04) and cost is
    // trivial (4.2 us per 16 ms frame, ~RTF 0.00026) — but loud music is
    // exactly the case the relative-energy gate cannot veto.
    // Energy-only v1 ships; the tests below document the negative result
    // and stay runnable (`cargo test --lib earshot_eval -- --ignored
    // --nocapture`) for re-evaluation against future earshot versions;
    // full numbers live in the bead close comment.
    // -----------------------------------------------------------------------

    /// earshot's native input frame: 256 samples @ 16 kHz = 16 ms.
    const EARSHOT_FRAME_SAMPLES: usize = 256;

    /// Contiguous [-1, 1] stream -> per-16 ms-frame voice scores.
    fn earshot_scores(samples: &[f32]) -> Vec<f32> {
        let mut det = earshot::Detector::default_boxed();
        samples
            .as_chunks::<EARSHOT_FRAME_SAMPLES>()
            .0
            .iter()
            .map(|f| det.predict_f32(f))
            .collect()
    }

    /// Deterministic LCG white noise at `db` dBFS (reproducible across runs).
    fn lcg_noise(len: usize, seed: u64, db: f64) -> Vec<f32> {
        let amp = 10f64.powf(db / 20.0);
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (s >> 33) as f64 / ((u64::MAX >> 33) as f64) * 2.0 - 1.0;
                (u * amp) as f32
            })
            .collect()
    }

    /// Deterministic music-like signal: C-major arpeggio, harmonic-rich
    /// plucked notes with vibrato and note envelopes, at `db` dBFS. Loud
    /// music passes any relative-energy gate — the exact case tier-2
    /// exists for.
    fn music_signal(len: usize, db: f64) -> Vec<f32> {
        const PI: f64 = std::f64::consts::PI;
        let amp = 10f64.powf(db / 20.0);
        let notes = [261.63_f64, 329.63, 392.0, 523.25];
        let note_len = 0.15_f64;
        (0..len)
            .map(|i| {
                let t = i as f64 / SAMPLE_RATE as f64;
                let idx = (t / note_len) as usize % notes.len();
                let tt = t % note_len;
                let env =
                    ((tt / 0.01).min(1.0) * ((note_len - tt) / 0.03).min(1.0)).clamp(0.0, 1.0);
                let vib = 1.0 + 0.004 * (2.0 * PI * 5.5 * t).sin();
                let f = notes[idx] * vib;
                let s = (2.0 * PI * f * t).sin()
                    + 0.35 * (2.0 * PI * 2.0 * f * t).sin()
                    + 0.12 * (2.0 * PI * 3.0 * f * t).sin();
                (amp * env * s * 0.5) as f32
            })
            .collect()
    }

    /// Deterministic mains hum: 60 Hz + harmonics, slow AM, at `db` dBFS.
    fn hum_signal(len: usize, db: f64) -> Vec<f32> {
        const PI: f64 = std::f64::consts::PI;
        let amp = 10f64.powf(db / 20.0);
        (0..len)
            .map(|i| {
                let t = i as f64 / SAMPLE_RATE as f64;
                let am = 0.85 + 0.15 * (2.0 * PI * 0.7 * t).sin();
                let s = (2.0 * PI * 60.0 * t).sin()
                    + 0.5 * (2.0 * PI * 120.0 * t).sin()
                    + 0.25 * (2.0 * PI * 180.0 * t).sin()
                    + 0.12 * (2.0 * PI * 240.0 * t).sin();
                (amp * am * s * 0.53) as f32
            })
            .collect()
    }

    fn jfk_samples() -> Vec<f32> {
        let mut reader =
            hound::WavReader::open("tests/fixtures/native/jfk.wav").expect("open jfk.wav");
        assert_eq!(
            reader.spec().sample_rate,
            u32::try_from(SAMPLE_RATE).unwrap()
        );
        assert_eq!(reader.spec().channels, 1);
        reader
            .samples::<i16>()
            .map(|s| f32::from(s.expect("pcm")) / 32_768.0)
            .collect()
    }

    #[ignore = "bd-rt-vad-stream-ulp5 verdict: earshot REJECTED — music -14 dB passes at every usable threshold (0.995@0.5, 0.80@0.85); kept as a re-evaluation harness"]
    #[test]
    fn earshot_eval_raw_scores_on_labeled_fixtures() {
        // clip the reference mask is a BATCH energy mask (p75 of frame RMS)
        // — eval-only ground truth approximation, never runtime logic.
        // Every fixture is scored and printed BEFORE any assertion so one
        // manual run records the complete matrix.
        let dur = |v: &[f32]| v.len() as f64 / SAMPLE_RATE as f64;

        let music = music_signal(SAMPLE_RATE * 6, -14.0);
        let noise = lcg_noise(SAMPLE_RATE * 4, 0xC0FFEE, -20.0);
        let hum = hum_signal(SAMPLE_RATE * 4, -18.0);
        let mut violations: Vec<String> = Vec::new();
        let row = |name: &str, samples: &[f32], scores: &[f32], violations: &mut Vec<String>| {
            let voiced = scores.iter().filter(|&&s| s >= 0.5).count();
            let frac = f64::from(voiced as u32) / scores.len() as f64;
            println!(
                "eval {name}: {voiced}/{} frames voiced ({frac:.4}), mean {:.3}, max {:.3} [{:.2}s]",
                scores.len(),
                scores.iter().sum::<f32>() / scores.len() as f32,
                scores.iter().cloned().fold(f32::MIN, f32::max),
                dur(samples)
            );
            if frac > 0.05 {
                violations.push(format!("{name}: {frac:.3} of loud non-speech passed"));
            }
        };

        let music_scores = earshot_scores(&music);
        row("music -14 dB", &music, &music_scores, &mut violations);
        row(
            "white noise -20 dB",
            &noise,
            &earshot_scores(&noise),
            &mut violations,
        );
        row(
            "mains hum -18 dB",
            &hum,
            &earshot_scores(&hum),
            &mut violations,
        );

        let jfk = jfk_samples();
        let speech_scores = earshot_scores(&jfk);
        let rms: Vec<f32> = jfk
            .as_chunks::<EARSHOT_FRAME_SAMPLES>()
            .0
            .iter()
            .map(|f| {
                let sq: f32 = f.iter().map(|v| v * v).sum();
                (sq / EARSHOT_FRAME_SAMPLES as f32).sqrt()
            })
            .collect();
        let mut sorted = rms.clone();
        sorted.sort_by(f32::total_cmp);
        let p75 = sorted[sorted.len() * 3 / 4];
        let speech_frames = rms.iter().filter(|&&r| r > p75).count();
        let voiced_total = speech_scores.iter().filter(|&&s| s >= 0.5).count();
        println!(
            "eval jfk.wav {:.2}s: {} frames, speech-mask {speech_frames}, earshot voiced \
             {voiced_total} ({:.3} of all)",
            dur(&jfk),
            speech_scores.len(),
            voiced_total as f64 / speech_scores.len() as f64,
        );

        // Threshold separation sweep: is there ANY cutoff that vetoes the
        // music while still catching speech?
        println!("threshold sweep [t]: music-pass-rate | jfk-mask-recall");
        for t in [0.30_f32, 0.50, 0.70, 0.85, 0.95] {
            let mp =
                music_scores.iter().filter(|&&s| s >= t).count() as f64 / music_scores.len() as f64;
            let rec = rms
                .iter()
                .zip(&speech_scores)
                .filter(|&(&r, &s)| r > p75 && s >= t)
                .count() as f64
                / speech_frames.max(1) as f64;
            println!("  [{t:.2}] {mp:.4} | {rec:.4}");
        }

        assert!(
            violations.is_empty(),
            "earshot failed the labeled-fixture evaluation: {violations:?}"
        );
    }

    #[ignore = "accuracy verdict is negative (see raw-scores harness); cost itself is fine — 4.2 us/16ms frame release; debug timing on shared rch workers is not decision-grade"]
    #[test]
    fn earshot_eval_cost_per_frame_is_far_under_step_budget() {
        // may cost at most ~1 ms per step => <=200 us per 16 ms predict.
        let audio = music_signal(SAMPLE_RATE * 10, -14.0);
        let mut det = earshot::Detector::default_boxed();
        let frames: Vec<&[f32]> = audio
            .as_chunks::<EARSHOT_FRAME_SAMPLES>()
            .0
            .iter()
            .map(|chunk| chunk.as_slice())
            .collect();
        for f in &frames[..32] {
            let _ = det.predict_f32(f); // warmup
        }
        let started = Instant::now();
        let mut sink = 0.0_f32;
        for f in &frames {
            sink += det.predict_f32(f);
        }
        let elapsed = started.elapsed();
        let per_frame_us = elapsed.as_nanos() as f64 / 1_000.0 / frames.len() as f64;
        println!(
            "eval earshot cost: {:.1} us per 16 ms frame over {} frames ({sink:.0} sink)",
            per_frame_us,
            frames.len()
        );
        assert!(
            per_frame_us < 200.0,
            "tier-2 too slow: {per_frame_us:.1} us/frame"
        );
    }
}

#[cfg(test)]
mod alignatt_tests {
    use super::*;
    use crate::model::TranscriptionSegment;
    use crate::native_engine::decode::{DecodeOutput, TokenAttn, WindowStats};

    /// Build a DecodeOutput with the given segments and one window whose
    /// token_attn carries the given attention frames (in decode order).
    fn out(segments: &[(f64, f64, &str)], attn_frames: &[u32]) -> DecodeOutput {
        DecodeOutput {
            segments: segments
                .iter()
                .map(|(start, end, text)| TranscriptionSegment {
                    start_sec: Some(*start),
                    end_sec: Some(*end),
                    text: (*text).to_owned(),
                    speaker: None,
                    confidence: Some(0.9),
                })
                .collect(),
            language: Some("en".to_owned()),
            windows: vec![WindowStats {
                avg_logprob: -0.1,
                no_speech_prob: 0.01,
                tokens: attn_frames.len(),
                window_offset_sec: 0.0,
                token_attn: attn_frames
                    .iter()
                    .enumerate()
                    .map(|(i, f)| TokenAttn {
                        token_index: i as u32,
                        attn_frame: *f,
                    })
                    .collect(),
            }],
            dropped_windows: Vec::new(),
            work: crate::native_engine::decode::DecodeWorkStats::default(),
            word_timings: None,
        }
    }

    #[test]
    fn commits_segments_wholly_behind_the_attention_boundary() {
        // 10 s slice, 200 ms holdback -> limit frame 490. Attention prefix
        // reaches frame 480 -> safe boundary 9.6 s. "alpha" and "bravo" end
        // behind it; "charlie" is the slice's final segment and is barred by
        // the closure guard regardless of attention.
        let mut p = AlignAttPolicy::new(200);
        let o = out(
            &[
                (0.0, 4.0, "alpha"),
                (4.0, 9.0, "bravo"),
                (9.0, 9.8, "charlie"),
            ],
            &[100, 200, 480],
        );
        let d = EmissionPolicy::step(&mut p, &o, 10.0);
        assert_eq!(d.commit_text, "alpha bravo");
        assert_eq!(d.commit_tokens, 2);
        assert_eq!(d.commit_through_sec, Some(9.0));
        assert_eq!(d.partial_tail.as_deref(), Some("charlie"));
        assert!(d.commit_confidence.is_some());
    }

    #[test]
    fn attention_prefix_rule_never_commits_past_an_unsafe_token() {
        // Middle token attends INSIDE the holdback zone (frame 495 > 490):
        // the prefix stops there; the trailing "safe" token must not
        // resurrect the boundary. Safe frame stays 100 -> 2.0 s -> nothing
        // commits.
        let mut p = AlignAttPolicy::new(200);
        let o = out(
            &[(0.0, 4.0, "alpha"), (4.0, 9.0, "bravo")],
            &[100, 495, 200],
        );
        let d = EmissionPolicy::step(&mut p, &o, 10.0);
        assert_eq!(d.commit_text, "");
        assert_eq!(d.commit_tokens, 0);
        assert_eq!(d.partial_tail.as_deref(), Some("alpha bravo"));
    }

    #[test]
    fn closure_guard_never_commits_a_slice_final_segment_mid_utterance() {
        // First-campaign regression: a truncated slice decoded a confident,
        // WRONG sentence close ("And so am I fellow Americans.") whose
        // attention sat safely behind the boundary, and committing it
        // dropped the utterance's real text. The slice's final segment
        // never commits mid-utterance, however safe its attention looks.
        let mut p = AlignAttPolicy::new(200);
        let o = out(&[(0.0, 2.0, "hallucinated close")], &[80]);
        let d = EmissionPolicy::step(&mut p, &o, 10.0);
        assert_eq!(d.commit_text, "");
        assert_eq!(d.partial_tail.as_deref(), Some("hallucinated close"));
        // finalize (audio complete) lifts the guard.
        let d = EmissionPolicy::finalize(&mut p, &o, 10.0);
        assert_eq!(d.commit_text, "hallucinated close");
        assert!(d.partial_tail.is_none());
    }

    #[test]
    fn slices_are_fresh_no_cross_decode_bookkeeping() {
        // The driver advances the slice origin past committed audio, so a
        // later step sees only uncommitted segments and the policy commits
        // from the front without any memory of earlier decodes.
        let mut p = AlignAttPolicy::new(200);
        let d1 = EmissionPolicy::step(
            &mut p,
            &out(&[(0.0, 2.0, "alpha"), (2.0, 4.5, "bravo")], &[150]),
            5.0,
        );
        assert_eq!(d1.commit_text, "alpha");
        assert_eq!(d1.commit_through_sec, Some(2.0));
        // Next slice starts where the commit ended: times re-zero, "bravo"
        // is now first and commits once a segment follows it.
        let d2 = EmissionPolicy::step(
            &mut p,
            &out(&[(0.0, 2.5, "bravo"), (2.5, 5.8, "charlie")], &[140]),
            6.0,
        );
        assert_eq!(d2.commit_text, "bravo");
        assert_eq!(d2.commit_through_sec, Some(2.5));
        assert_eq!(d2.partial_tail.as_deref(), Some("charlie"));
    }

    #[test]
    fn finalize_commits_everything_including_the_final_segment() {
        let mut p = AlignAttPolicy::new(200);
        let o = out(&[(0.0, 2.0, "alpha"), (2.0, 3.9, "bravo")], &[120]);
        let d = EmissionPolicy::finalize(&mut p, &o, 4.0);
        assert_eq!(d.commit_text, "alpha bravo");
        assert_eq!(d.commit_through_sec, Some(3.9));
        assert!(d.partial_tail.is_none());
    }

    #[test]
    fn empty_decode_yields_an_empty_decision() {
        let mut p = AlignAttPolicy::new(200);
        let d = EmissionPolicy::step(&mut p, &out(&[], &[]), 1.0);
        assert_eq!(d.commit_text, "");
        assert!(d.partial_tail.is_none());
        assert!(d.commit_through_sec.is_none());
    }

    #[test]
    fn hallucination_gate_holds_commits_on_low_confidence_slices() {
        // Campaign receipt: a step decode over pure tone produced
        // confident-looking segments; committing them puts invented text
        // in the append-only stream. High no-speech probability (or very
        // low avg logprob) must hold ALL commits for that step.
        let mut p = AlignAttPolicy::new(200);
        let mut o = out(&[(0.0, 2.0, "ghost"), (2.0, 4.0, "words")], &[80, 150]);
        o.windows[0].no_speech_prob = 0.9;
        let d = EmissionPolicy::step(&mut p, &o, 10.0);
        assert_eq!(d.commit_text, "");
        assert!(d.holdback);
        assert_eq!(d.partial_tail.as_deref(), Some("ghost words"));
        // Same segments with healthy stats commit normally.
        let o2 = out(&[(0.0, 2.0, "real"), (2.0, 4.0, "words")], &[80, 150]);
        let d2 = EmissionPolicy::step(&mut p, &o2, 10.0);
        assert_eq!(d2.commit_text, "real");
        assert!(!d2.holdback);
        // Very low avg_logprob trips the gate too.
        let mut o3 = out(&[(0.0, 2.0, "mush"), (2.0, 4.0, "tail")], &[80, 150]);
        o3.windows[0].avg_logprob = -1.5;
        assert!(EmissionPolicy::step(&mut p, &o3, 10.0).holdback);
    }

    #[test]
    fn holdback_floors_at_one_frame_and_policies_declare_attn_needs() {
        // 0 ms holdback still keeps one frame of guard.
        let p = AlignAttPolicy::new(0);
        assert_eq!(p.holdback_frames, 1);
        assert!(EmissionPolicy::needs_token_attn(&p));
        assert!(!EmissionPolicy::needs_token_attn(&EndpointCommitPolicy));
        assert_eq!(EmissionPolicy::name(&p), "alignatt");
    }
}
