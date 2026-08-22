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

#[cfg(test)]
mod tests {
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
        let mut small = SessionBufferConfig::default();
        small.prompt_cap_chars = 12;
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
        let mut c = SessionBufferConfig::default();
        c.cross_utterance_tail_chars = 10;
        let mut b = SessionBuffer::new(c);
        b.append_committed_text("one two three four five six");
        b.end_utterance();
        let p = b.prompt().expect("prompt");
        assert!(p.chars().count() <= 10, "tail cap exceeded: {p:?}");
    }

    #[test]
    fn no_context_disables_all_carry() {
        let mut c = SessionBufferConfig::default();
        c.prompt_carry = false;
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
        let mut c = SessionBufferConfig::default();
        c.prompt_cap_chars = 5;
        c.cross_utterance_tail_chars = 5;
        let mut b = SessionBuffer::new(c);
        b.append_committed_text("héllo wörld émoji");
        b.end_utterance();
        // Must not panic on char boundaries; result respects the char cap.
        if let Some(p) = b.prompt() {
            assert!(p.chars().count() <= 5, "cap exceeded: {p:?}");
        }
    }
}
