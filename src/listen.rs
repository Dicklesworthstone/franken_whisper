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
        if self.cfg.prompt_carry {
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
}

/// The emission-policy seam (bd-rt-listen-cmd + policy beads): decides,
/// per step decode over the rolling buffer, what becomes committed
/// (append-only `transcript.delta`) versus mutable preview.
pub trait EmissionPolicy: Send {
    /// A mid-utterance step decode (buffer edge is LIVE — audio continues).
    fn step(
        &mut self,
        out: &crate::native_engine::decode::DecodeOutput,
        buffer: &SessionBuffer,
    ) -> PolicyDecision;
    /// The endpoint decode (audio for this utterance is COMPLETE — no
    /// holdback; commit everything not yet committed).
    fn finalize(
        &mut self,
        out: &crate::native_engine::decode::DecodeOutput,
        buffer: &SessionBuffer,
    ) -> PolicyDecision;
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
        _buffer: &SessionBuffer,
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
        buffer: &SessionBuffer,
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
            commit_through_sec: end.map(|t| buffer.session_time_of(t)),
            commit_tokens: out.windows.iter().map(|w| w.tokens as u64).sum(),
            commit_confidence,
            commit_text: text,
            partial_tail: None,
        }
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "endpoint-commit"
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

fn open_capture_source(
    config: &ListenConfig,
) -> FwResult<(
    Box<dyn crate::capture::CaptureSource>,
    &'static str,
    String,
    Option<String>,
)> {
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

/// Run one live session, emitting NDJSON event values through `emit`.
/// Returns `Ok(cancelled)` — `true` when the session ended on Ctrl-C (the
/// caller maps that to exit code 130). Fatal errors propagate; the CLI
/// wrapper converts them to the terminal `run_error` event.
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
    let mut policy = EndpointCommitPolicy;
    let confirm_disabled_by_fallback = fallback_warning.is_some();

    let info = ListenSessionInfo {
        source: source_label.to_owned(),
        capture_backend: capture_backend.to_owned(),
        device: device_label,
        sample_rate_hz: capture.sample_rate(),
        fast_model: fast_model_label.clone(),
        quality_model: None, // confirm lane lands with bd-rt-confirm-lane-3okr
        policy: EmissionPolicy::name(&policy).to_owned(),
        step_ms: config.step_ms,
        partials: config.emit_partials,
        vad: serde_json::json!({
            "enabled": config.vad_enabled,
            "min_speech_ms": config.vad.min_speech_ms,
            "endpoint_ms": config.vad.endpoint_ms,
            "gate_db": config.vad.gate_db,
        }),
    };
    emit(robot::listen_session_start_value(
        &run_id,
        seq,
        &now_ts(),
        &info,
    ))?;
    seq += 1;
    if let Some(message) = &fallback_warning {
        emit(robot::listen_warning_value(
            &run_id,
            seq,
            &now_ts(),
            "fast_model_fallback",
            serde_json::json!({"detail": message, "confirm_lane_disabled": confirm_disabled_by_fallback}),
        ))?;
        seq += 1;
    }
    if let Some(message) = capture_warning {
        emit(robot::listen_warning_value(
            &run_id,
            seq,
            &now_ts(),
            "fallback_capture_backend",
            serde_json::json!({"detail": message}),
        ))?;
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

    let mut buffer = SessionBuffer::new(config.buffer.clone());
    let mut vad = StreamingVad::new(config.vad.clone());
    let mut watchdog = crate::capture::SilentInputWatchdog::new(
        3.0,
        crate::native_engine::mel::SAMPLE_RATE as u32,
    );
    let mut pinned_language = config.language.clone();

    // Session state.
    let mut utterance_id: u32 = 0;
    let mut in_speech = false;
    let mut utterance_started_at: f64 = 0.0;
    let mut utterance_t0: f64 = 0.0;
    let mut committed_text = String::new();
    let mut delta_count: u64 = 0;
    let mut partial_generation: u64 = 0;
    let mut stats = ListenSessionStats::default();
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

    macro_rules! emit_seq {
        ($value:expr) => {{
            emit($value)?;
            seq += 1;
        }};
    }

    // One decode over the current buffer.
    let mut decode_buffer = |buffer: &SessionBuffer,
                             pinned_language: &Option<String>|
     -> FwResult<crate::native_engine::decode::DecodeOutput> {
        let params = DecodeParams {
            language: pinned_language.clone(),
            timestamps: true,
            audio_ctx: AudioCtxPolicy::Auto,
            bypass_transcript_cache: true,
            initial_prompt: buffer.prompt(),
            n_threads: 4,
            ..DecodeParams::default()
        };
        model.transcribe(buffer.window(), &params, &checkpoint)
    };

    'session: loop {
        // -- termination checks ------------------------------------------
        if is_cancelled() {
            cancelled = true;
        }
        let out_of_time = config.max_seconds > 0.0
            && session_started.elapsed().as_secs_f64() >= config.max_seconds;
        if cancelled || out_of_time || source_ended {
            // Final flush when speech is open.
            if in_speech {
                let out = decode_buffer(&buffer, &pinned_language)?;
                let decision = EmissionPolicy::finalize(&mut policy, &out, &buffer);
                let t1 = buffer.session_duration_sec();
                if !decision.commit_text.is_empty() {
                    committed_text.push_str(&decision.commit_text);
                    delta_count += 1;
                    stats.deltas += 1;
                    emit_seq!(robot::transcript_delta_value(
                        &run_id,
                        seq,
                        &now_ts(),
                        utterance_id,
                        &decision.commit_text,
                        utterance_t0,
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
                            delta_count = 0;
                            partial_generation = 0;
                            EmissionPolicy::reset(&mut policy);
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
                            let out = decode_buffer(&buffer, &pinned_language)?;
                            if pinned_language.is_none() {
                                pinned_language.clone_from(&out.language);
                            }
                            let decision = EmissionPolicy::finalize(&mut policy, &out, &buffer);
                            if !decision.commit_text.is_empty() {
                                committed_text.push_str(&decision.commit_text);
                                delta_count += 1;
                                stats.deltas += 1;
                                if stats.ttft_ms.is_none() {
                                    stats.ttft_ms =
                                        Some(session_started.elapsed().as_secs_f64() * 1000.0);
                                }
                                emit_seq!(robot::transcript_delta_value(
                                    &run_id,
                                    seq,
                                    &now_ts(),
                                    utterance_id,
                                    &decision.commit_text,
                                    utterance_t0,
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
                            // Trim + prompt carry.
                            buffer.append_committed_text(&committed_text);
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
            let out = decode_buffer(&buffer, &pinned_language)?;
            let decision = EmissionPolicy::finalize(&mut policy, &out, &buffer);
            let t1 = buffer.session_duration_sec();
            if !decision.commit_text.is_empty() {
                committed_text.push_str(&decision.commit_text);
                delta_count += 1;
                stats.deltas += 1;
                emit_seq!(robot::transcript_delta_value(
                    &run_id,
                    seq,
                    &now_ts(),
                    utterance_id,
                    &decision.commit_text,
                    utterance_t0,
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
            buffer.append_committed_text(&committed_text);
            buffer.set_committed_through(t1);
            buffer.trim_to_committed();
            buffer.end_utterance();
            // Forced end mid-speech: next utterance opens immediately.
            utterance_id += 1;
            utterance_started_at = session_started.elapsed().as_secs_f64();
            utterance_t0 = t1;
            committed_text.clear();
            delta_count = 0;
            partial_generation = 0;
            EmissionPolicy::reset(&mut policy);
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
            let out = decode_buffer(&buffer, &pinned_language)?;
            if pinned_language.is_none() {
                pinned_language.clone_from(&out.language);
            }
            let decision = EmissionPolicy::step(&mut policy, &out, &buffer);
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

    // Terminal: final stats (success path; run_error is the fatal terminal
    // and is emitted by the CLI wrapper when this function errors).
    fill_step_stats(&mut stats, &step_latencies_ms, session_started, &buffer);
    emit(robot::listen_session_stats_value(
        &run_id,
        seq,
        &now_ts(),
        &stats,
        true,
    ))?;
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
