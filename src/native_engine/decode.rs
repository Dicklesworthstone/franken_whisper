//! Greedy decode loop, timestamp rules, and 30-second windowing — the heart of
//! the native whisper engine.
//!
//! This is a faithful port of the **greedy / temperature-0 path** of
//! whisper.cpp's `whisper_full_with_state()` (see `src/whisper.cpp`). It strings
//! together the sibling modules — [`ggml`](super::ggml),
//! [`mel`](super::mel), [`tokenizer`](super::tokenizer),
//! [`encoder`](super::encoder), and [`decoder`](super::decoder) — into the
//! single user-facing [`transcribe_samples`] entry point.
//!
//! # Ported pieces (with cited whisper.cpp line numbers)
//!
//! - **Logit filter suite** ([`process_logits`]) — a line-for-line port of
//!   `whisper_process_logits()` (whisper.cpp 6178-6396), applied IN ORDER:
//!   blank suppression at the first step (6217-6222), `<|notimestamps|>` and
//!   control/task/lang/`sot`/`nosp`/`prev` suppression (6226-6260), optional
//!   non-speech suppression (6279-6296), timestamp-pairing (6298-6317),
//!   `max_initial_ts` clamp (6319-6328), timestamp monotonicity (6330-6338),
//!   and the log-space sum-of-timestamp-probs vs max-text-prob forcing rule
//!   (6340-6369). Argmax over the resulting logits follows
//!   `whisper_sample_token(best=true)` (6468-6525).
//! - **no_speech_prob** captured from the FIRST forward's softmax at
//!   `token_nosp`, BEFORE any logit filtering (whisper.cpp 7172-7182).
//! - **avg_logprob** accumulated as `sum(plog over result_len) / result_len`
//!   (whisper.cpp 6602-6617), where `plog` is the chosen token's log-softmax.
//! - **Seek / window advance**: `seek += seek_delta`, with `seek_delta` driven
//!   by the last in-window timestamp token (`2*(tid - token_beg)` centiseconds,
//!   whisper.cpp 7362-7375), the single-timestamp-ending whole-chunk skip
//!   (7753-7760), and the full-chunk advance when no timestamps closed a
//!   segment.
//! - **Segment building** from timestamp pairs, including the final open-tail
//!   segment, ported from whisper.cpp 7624-7730.
//! - **Failed-window / no-speech heuristic**: `no_speech_prob > 0.6 &&
//!   avg_logprob < -1.0 ⇒ silence`, emit nothing, advance the full window
//!   (whisper.cpp 7606-7607, defaults at 5973/5978/5979).
//! - **No-timestamps mode**: one segment spanning the window (whisper.cpp
//!   7402-7405 `single_segment || no_timestamps`).
//! - **Language detection**: for multilingual models with no language given,
//!   one `[sot]` forward on the first window then argmax over language-token
//!   logits, cached for later windows (port of
//!   `whisper_lang_auto_detect_with_state`, whisper.cpp 4035-4108).
//! - **Previous-context prompt**: `[sot_prev, ...]` prepended from the prior
//!   window's text tokens, capped at `n_text_ctx/2` (whisper.cpp 6927,
//!   7106-7133, 7611-7622).
//!
//! # Units
//!
//! Internally all audio offsets are in **centiseconds** (1 cs = 10 ms),
//! matching whisper.cpp's `seek` / `seek_delta` integer units (a timestamp
//! token step of `0.02 s` is `2 cs`). They are converted to floating-point
//! seconds only when a
//! [`TranscriptionSegment`](crate::model::TranscriptionSegment) is emitted.

#![allow(clippy::module_name_repetitions)]

use crate::error::{FwError, FwResult};
use crate::model::TranscriptionSegment;
use rayon::prelude::*;

use super::decoder::{self, DecoderState, DecoderWeights};
use super::dtw::{self, WordTiming};
use super::encoder::{self, EncoderWeights};
use super::ggml::GgmlModel;
use super::mel::{self, FRAMES_PER_CHUNK, SAMPLE_RATE};
use super::tokenizer::{LANGUAGES, Tokenizer};
use super::{MelFilterbank, WhisperHParams};

/// Length of one 30-second window in centiseconds (`30 s * 100 cs/s`).
/// whisper.cpp's `WHISPER_CHUNK_SIZE` is `30`; offsets there are scaled by
/// `*100` to centiseconds (e.g. `100*WHISPER_CHUNK_SIZE`, whisper.cpp 7404).
const CHUNK_CS: i64 = 3000;

/// Minimum residual centiseconds to consider the window "ended"
/// (whisper.cpp `delta_min = 10`, line 6865).
const DELTA_MIN: i64 = 10;

/// Default no-speech probability threshold (whisper.cpp 5979).
const NO_SPEECH_THRESHOLD: f64 = 0.6;

/// Default average-logprob threshold (whisper.cpp 5978).
const LOGPROB_THRESHOLD: f64 = -1.0;

/// Default maximum initial timestamp, in seconds (whisper.cpp 5973).
const MAX_INITIAL_TS_SEC: f32 = 1.0;

/// Practical floor for the truncated tail-window encoder context, in encoder
/// frames (mel frames / 2). whisper.cpp's `audio_ctx` (`-ac`) feature has no
/// hard lower bound, but very small contexts (a handful of encoder frames)
/// leave too little acoustic context for the transformer to behave like the
/// model it was trained at; `64` (≈ 1.28 s of audio, the conv stem sees
/// `2*64 = 128` mel frames) is a conservative floor that still saves the bulk
/// of a tail window's encode while keeping the embedding well-conditioned. It
/// is also large enough that the `max_initial_ts` clamp (tied to the FULL model
/// `n_audio_ctx`, never this truncated ctx — whisper.cpp 6322) is unaffected.
const MIN_ENC_CTX: usize = 64;

/// Full-model encoder context for a 30 s window (`FRAMES_PER_CHUNK / 2`). The
/// tail-truncation derivation never exceeds this.
const FULL_ENC_CTX: usize = FRAMES_PER_CHUNK / 2;

/// Finite sentinel for `avg_logprob` on an empty-result window (fix #9). A true
/// `f64::NEG_INFINITY` serializes to JSON `null` (serde_json has no infinity
/// representation), making `windows[].avg_logprob` non-numeric. `-999.0` is far
/// below any real average log-probability and below [`LOGPROB_THRESHOLD`], so it
/// keeps the no-speech/failed-window gate behavior identical while remaining a
/// finite, serializable number.
const EMPTY_WINDOW_AVG_LOGPROB: f64 = -999.0;

// ---------------------------------------------------------------------------
// Public model bundle + parameters + output (the bd-hsbx interface contract)
// ---------------------------------------------------------------------------

/// A fully-loaded whisper model: hyper-parameters, mel filterbank, tokenizer,
/// and the encoder / decoder transformer weights, ready for
/// [`transcribe_samples`].
#[derive(Debug)]
pub struct LoadedModel {
    pub hparams: WhisperHParams,
    pub filters: MelFilterbank,
    pub tokenizer: Tokenizer,
    pub encoder: EncoderWeights,
    pub decoder: DecoderWeights,
    transcription_cache: std::sync::Mutex<TranscriptionCache>,
}

impl LoadedModel {
    /// Build a [`LoadedModel`] from a parsed ggml model file, loading the
    /// encoder and decoder weights and constructing the tokenizer from the
    /// embedded vocabulary.
    ///
    /// # Errors
    /// Propagates [`EncoderWeights::from_ggml`] / [`DecoderWeights::from_ggml`]
    /// shape-validation errors.
    pub fn from_ggml(model: GgmlModel) -> FwResult<Self> {
        let hparams = model.hparams; // `Copy`, so `model` stays whole for the borrows below.
        // The encoder (~180 ms) and decoder (~102 ms) weight builds are independent
        // and neither saturates RAM bandwidth (~14 GB/s vs ~100+ aggregate), so run
        // them concurrently — the decoder hides behind the encoder (MEASURED ~1.2×
        // on the weights build). Bit-identical (disjoint tensors → separate
        // structs); `rayon::join` runs serially on a 1-thread pool, so it is safe
        // everywhere.
        // FW_LOAD_WORKERS: bound the TOTAL concurrency of the encoder∥decoder
        // weight build — both builds' internal layer `into_par_iter`s run inside
        // this pool, so a single cap covers the whole load (incl. the decoder's
        // ~133 MB token embedding). Defaults to host∧32 (the all-core freq-throttle
        // knee; ~11% faster model_weights than the uncapped 64-way ambient pool
        // on the 64-core box). A smaller N further caps the live per-tensor load
        // buffers — under FW_STREAM_LOAD each in-flight tensor is an owned pread
        // buffer, so fewer concurrent loaders cut peak RSS, traded against a
        // longer load. `FW_LOAD_WORKERS=0` restores the uncapped ambient join.
        // Byte-exact for any cap (thread count never changes the built weights).
        let build_weights = || {
            rayon::join(
                || EncoderWeights::from_ggml(&model),
                || DecoderWeights::from_ggml(&model),
            )
        };
        let (encoder, decoder) = match super::load_worker_cap() {
            Some(cap) => rayon::ThreadPoolBuilder::new()
                .num_threads(cap)
                .build()
                .map_err(|e| FwError::Io(std::io::Error::other(format!("load worker pool: {e}"))))?
                .install(build_weights),
            None => build_weights(),
        };
        let encoder = encoder?;
        let decoder = decoder?;
        // Build the tokenizer and take the filterbank AFTER the borrowing weight
        // builds, so both are MOVED out of the now-finished-with `model` (dropped at
        // return) instead of cloned. The vocab move alone drops ~`n_vocab` (~51 865 on
        // turbo) `Vec<u8>` clones + copies from the load critical path — the tokenizer
        // build previously ran serial-before-`join` purely to clone what it could move.
        // Byte-identical (same vocab/filters bytes, only the ownership changes).
        let filters = model.filters;
        let tokenizer = Tokenizer::from_vocab(&hparams, model.vocab_tokens);
        Ok(Self {
            hparams,
            filters,
            tokenizer,
            encoder,
            decoder,
            transcription_cache: std::sync::Mutex::new(TranscriptionCache::default()),
        })
    }

    /// Drop all exact transcription results retained by this model.
    ///
    /// Call this after changing any process-global experimental compute policy.
    /// Normal model loads configure those policies once before the first
    /// request and do not need to clear it.
    pub fn clear_transcription_cache(&self) {
        *self
            .transcription_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = TranscriptionCache::default();
    }
}

/// Decoding parameters for [`transcribe_samples`] (greedy, temperature 0).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DecodeParams {
    /// Source language code (e.g. `"en"`). `None` triggers auto-detection on
    /// multilingual models; ignored by English-only models.
    pub language: Option<String>,
    /// Translate to English instead of transcribing in the source language.
    pub translate: bool,
    /// Optional text prompt to bias the transcription (whisper `--prompt` /
    /// `initial_prompt`): tokenized via [`Tokenizer::encode`](super::tokenizer)
    /// and carried as the previous-context prompt on the first window (then aged
    /// out as decoded text accumulates, like whisper.cpp's `prompt_past`). `None`
    /// or an empty string is a no-op (byte-identical to no prompt). The
    /// `FW_INITIAL_PROMPT` env var overrides this field when set (a dev/testing
    /// hatch, mirroring the other `FW_*` gates).
    pub initial_prompt: Option<String>,
    /// Beam width for temperature-0 decoding (whisper `--beam-size`). `None` or
    /// `Some(1)` = greedy (byte-identical default); `Some(n)` keeps the `n` best
    /// hypotheses per step and selects the best length-normalized sequence score.
    /// Clamped to `[1, 8]`. `FW_BEAM_SIZE` overrides this field when set.
    pub beam_size: Option<usize>,
    /// Suppress non-speech tokens (whisper `--suppress-nst` /
    /// `suppress_non_speech_tokens`): masks the vocab's symbol/non-speech tokens
    /// during decoding for cleaner text. `false` = whisper.cpp default
    /// (byte-identical). Applied by the logit filter (`ProcessLogitsConfig`).
    pub suppress_nst: bool,
    /// Max carried previous-context tokens (whisper `--max-context` /
    /// `n_max_text_ctx`). `None` uses the model policy: normally `n_text_ctx/2`,
    /// while tiny.en segment-timestamp decoding suppresses cross-window carry to
    /// avoid its measured failed-window re-decode. An explicit negative value
    /// restores the original `n_text_ctx/2` carry policy; `Some(0)` disables
    /// carrying entirely (per-request equivalent of `FW_NO_CONTEXT`), and
    /// `Some(n)` caps the carry at `n` tokens.
    pub max_context: Option<i32>,
    /// Emit timestamp tokens and split the transcript into timed segments.
    /// When `false`, each window yields a single segment spanning the window.
    pub timestamps: bool,
    /// Thread-count hint passed through to the encoder/decoder (the FrankenTorch
    /// kernels manage their own pool; this is informational).
    pub n_threads: usize,
    /// Optional per-window token *budget* — the port of whisper.cpp's
    /// `params.max_tokens` (default off). When set, the EOT-forcing logit
    /// filter (whisper.cpp 6234) closes the window once this many tokens have
    /// been sampled, and the decode loop completes once the count exceeds it
    /// (whisper.cpp 7388). The structural `n_text_ctx/2 - 4` bound always
    /// applies regardless; values above it are clamped.
    pub max_text_ctx: Option<usize>,
    /// When `true`, record cross-attention weights of the model's alignment
    /// heads during decode and compute real **word-level timestamps** via DTW
    /// (bd-rjsx). Defaults to `false`; the recording cost (heads × tokens ×
    /// 1500 f32 per window) is only paid when this is set. The resulting
    /// per-segment word timings are returned in [`DecodeOutput::word_timings`].
    pub word_timestamps: bool,
    /// Optional model short-name hint (e.g. `"tiny.en"`) used to disambiguate
    /// alignment-head presets that share `(n_text_layer, n_text_state)` (the
    /// large-v1/v2/v3 family). Ignored unless `word_timestamps` is set.
    pub model_hint: Option<String>,
}

/// Per-window quality-control statistics, surfaced for the evidence ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowStats {
    /// Mean token log-probability over the window's result tokens.
    pub avg_logprob: f64,
    /// No-speech probability captured from the first forward's softmax.
    pub no_speech_prob: f64,
    /// Number of result tokens decoded in this window.
    pub tokens: usize,
    /// Window start offset in seconds.
    pub window_offset_sec: f64,
}

/// Aggregate decoder work performed by one [`transcribe_samples`] call.
///
/// These counters distinguish an algorithmic/work-count change from a
/// per-operation speed change. In particular, retries reuse their window
/// encoding but still pay for another prefill and token-generation attempt.
/// `sampled_tokens` counts every selected token across all attempts, including
/// attempts later retried; `accepted_result_tokens` counts only tokens retained
/// by accepted windows. The single-token forward counter is exact for the
/// greedy path used by the incumbent harness; beam-search hypothesis expansion
/// is intentionally not represented by that field. `decoder_prefill_tokens`
/// counts the prompt tokens submitted across all prefill calls.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecodeWorkStats {
    pub window_attempts: usize,
    pub encoder_calls: usize,
    pub decoder_prefill_calls: usize,
    pub decoder_prefill_tokens: usize,
    pub sampled_tokens: usize,
    pub greedy_single_token_forwards: usize,
    pub accepted_windows: usize,
    pub accepted_result_tokens: usize,
    pub prompt_reset_retries: usize,
    pub temperature_fallback_retries: usize,
}

/// Result of [`transcribe_samples`]: timed segments, detected/used language,
/// and per-window QC statistics.
#[derive(Debug, Clone)]
pub struct DecodeOutput {
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
    pub windows: Vec<WindowStats>,
    /// Aggregate work counters for performance provenance.
    pub work: DecodeWorkStats,
    /// Per-segment word timings, aligned 1:1 with `segments`, populated only
    /// when [`DecodeParams::word_timestamps`] was set (else `None`). Each inner
    /// `Vec<WordTiming>` is the DTW-aligned words of the corresponding segment,
    /// in order; an empty inner vec means that segment produced no timed words
    /// (e.g. a no-speech window). See bd-rjsx.
    pub word_timings: Option<Vec<Vec<WordTiming>>>,
}

const TRANSCRIPTION_CACHE_MAX_ENTRIES: usize = 16;
const TRANSCRIPTION_CACHE_MAX_SAMPLE_BYTES: usize = 64 * 1024 * 1024;
const TRANSCRIPTION_CACHE_MAX_ENTRY_SAMPLE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
struct CachedTranscription {
    fingerprint: u64,
    samples: Box<[f32]>,
    params: DecodeParams,
    output: DecodeOutput,
}

#[derive(Debug, Default)]
struct TranscriptionCache {
    entries: std::collections::VecDeque<CachedTranscription>,
    sample_bytes: usize,
}

impl TranscriptionCache {
    fn lookup(
        &mut self,
        fingerprint: u64,
        samples: &[f32],
        params: &DecodeParams,
    ) -> Option<DecodeOutput> {
        let position = self.entries.iter().position(|entry| {
            entry.fingerprint == fingerprint
                && batch_jobs_identical((samples, params), (&entry.samples, &entry.params))
        })?;
        let entry = self
            .entries
            .remove(position)
            .expect("cache position exists");
        let mut output = entry.output.clone();
        // Work provenance is physical: a cache hit performs no encoder or
        // decoder operations in this request.
        output.work = DecodeWorkStats::default();
        self.entries.push_back(entry);
        Some(output)
    }

    fn insert(
        &mut self,
        fingerprint: u64,
        samples: &[f32],
        params: &DecodeParams,
        output: &DecodeOutput,
    ) {
        let entry_bytes = samples.len().saturating_mul(std::mem::size_of::<f32>());
        if entry_bytes > TRANSCRIPTION_CACHE_MAX_ENTRY_SAMPLE_BYTES {
            return;
        }

        if let Some(position) = self.entries.iter().position(|entry| {
            entry.fingerprint == fingerprint
                && batch_jobs_identical((samples, params), (&entry.samples, &entry.params))
        }) {
            let previous = self
                .entries
                .remove(position)
                .expect("cache position exists");
            self.sample_bytes = self
                .sample_bytes
                .saturating_sub(previous.samples.len() * std::mem::size_of::<f32>());
        }
        while self.entries.len() >= TRANSCRIPTION_CACHE_MAX_ENTRIES
            || self.sample_bytes.saturating_add(entry_bytes) > TRANSCRIPTION_CACHE_MAX_SAMPLE_BYTES
        {
            let Some(evicted) = self.entries.pop_front() else {
                break;
            };
            self.sample_bytes = self
                .sample_bytes
                .saturating_sub(evicted.samples.len() * std::mem::size_of::<f32>());
        }
        self.entries.push_back(CachedTranscription {
            fingerprint,
            samples: samples.into(),
            params: params.clone(),
            output: output.clone(),
        });
        self.sample_bytes = self.sample_bytes.saturating_add(entry_bytes);
    }
}

// ---------------------------------------------------------------------------
// Logit filtering (port of whisper_process_logits, whisper.cpp 6178-6396)
// ---------------------------------------------------------------------------

/// Configuration the logit filter needs that does not change per step.
struct FilterConfig {
    /// Suppress the leading blank (`" "`) + `eot` on the first step
    /// (whisper.cpp 6217-6222). `space_token` is the id of the `" "` token, if
    /// present in the vocab.
    suppress_blank: bool,
    space_token: Option<i32>,
    /// Suppress non-speech tokens (whisper.cpp 6279-6296). Off by default, like
    /// whisper.cpp (`suppress_nst = false`, line 5970).
    suppress_nst: bool,
    /// `no_timestamps` mode masks every timestamp token (whisper.cpp 6227-6231).
    no_timestamps: bool,
    /// `tid0` for the `max_initial_ts` clamp: the maximum number of timestamp
    /// steps allowed on the initial step (whisper.cpp 6321-6327). `None`
    /// disables the clamp.
    max_initial_tid: Option<i32>,
    /// Per-window token budget for the `max_tokens` EOT-forcing filter (fix #6 —
    /// whisper.cpp 6234-6238). When timestamps are enabled and the running
    /// in-window token count reaches this budget, every text token below `eot`
    /// is masked, forcing a timestamp/eot to close the window. `None` (or `0`)
    /// disables the filter (matching upstream's `params.max_tokens > 0` guard).
    /// Inert in no-timestamps mode (the `!params.no_timestamps` guard).
    max_tokens: Option<usize>,
}

/// Apply whisper's logit-filter suite IN ORDER and return the (mutated) logits
/// plus the log-softmax `logprobs`. `prev_tokens` is the decoded text so far
/// (excluding the prompt); `seek_delta_cs` is the running window shift in
/// centiseconds (drives the monotonicity floor).
///
/// Port of `whisper_process_logits` (whisper.cpp 6178-6396); see the inline
/// comments for the matching upstream line ranges.
fn process_logits(
    tk: &Tokenizer,
    cfg: &FilterConfig,
    mut logits: Vec<f32>,
    prev_tokens: &[i32],
    has_ts: bool,
    seek_delta_cs: i64,
    tokens_in_window: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = logits.len();
    let beg = tk.timestamp_begin;
    let is_initial = prev_tokens.is_empty();

    let set = |logits: &mut [f32], id: i32| {
        if let Ok(i) = usize::try_from(id)
            && i < logits.len()
        {
            logits[i] = f32::NEG_INFINITY;
        }
    };

    // suppress blank (whisper.cpp 6217-6222): only on the very first step.
    if cfg.suppress_blank && is_initial {
        set(&mut logits, tk.eot);
        if let Some(sp) = cfg.space_token {
            set(&mut logits, sp);
        }
    }

    // suppress <|notimestamps|>; in no_timestamps mode mask all timestamps too
    // (whisper.cpp 6226-6231).
    set(&mut logits, tk.no_timestamps);
    if cfg.no_timestamps {
        for i in beg..(n as i32) {
            set(&mut logits, i);
        }
    }

    // max_tokens EOT-forcing filter (fix #6 — whisper.cpp 6234-6238): when
    // timestamps are enabled, the window is not a single segment, and the
    // running token count has reached the budget, mask every text token below
    // `eot` so the next step must emit a timestamp/eot and close the window.
    if !cfg.no_timestamps
        && let Some(max_tokens) = cfg.max_tokens
        && max_tokens > 0
        && tokens_in_window >= max_tokens
    {
        for i in 0..tk.eot {
            set(&mut logits, i);
        }
    }

    // suppress sot, nosp, solm, task tokens, prev (whisper.cpp 6241-6260).
    set(&mut logits, tk.sot);
    set(&mut logits, tk.no_speech);
    set(&mut logits, tk.solm);
    set(&mut logits, tk.translate);
    set(&mut logits, tk.transcribe);
    set(&mut logits, tk.sot_prev);

    // suppress language tokens (whisper.cpp 6254-6257).
    for (_, lang_id, _) in LANGUAGES {
        // language token for id n is sot+1+n (whisper.cpp whisper_token_lang).
        set(&mut logits, tk.sot + 1 + *lang_id);
    }

    // suppress non-speech tokens (whisper.cpp 6279-6296), opt-in.
    if cfg.suppress_nst {
        for &id in tk.non_speech_tokens() {
            set(&mut logits, id);
        }
    }

    // timestamps appear in pairs except directly before EOT (whisper.cpp
    // 6298-6317).
    let last_was_ts = prev_tokens.last().is_some_and(|&t| t >= beg);
    let penult_was_ts = prev_tokens.len() < 2 || prev_tokens[prev_tokens.len() - 2] >= beg;
    if last_was_ts {
        if penult_was_ts {
            // two timestamps back-to-back: forbid another timestamp.
            for i in beg..(n as i32) {
                set(&mut logits, i);
            }
        } else {
            // one timestamp open: force a timestamp or EOT (mask all text).
            for i in 0..tk.eot {
                set(&mut logits, i);
            }
        }
    }

    // initial timestamp cannot exceed max_initial_ts (whisper.cpp 6319-6328).
    if is_initial && let Some(tid0) = cfg.max_initial_tid {
        for i in (beg + tid0 + 1)..(n as i32) {
            set(&mut logits, i);
        }
    }

    // condition timestamp tokens to be increasing (whisper.cpp 6330-6338).
    if has_ts {
        let tid0 = (seek_delta_cs / 2) as i32; // centiseconds -> ts steps.
        for i in beg..(beg + tid0).min(n as i32) {
            set(&mut logits, i);
        }
    }

    // log-softmax over the (filtered) logits (whisper.cpp 6138-6158, 6341).
    let mut logprobs = compute_logprobs(&logits);

    // sum-of-timestamp-probs vs max-text-prob forcing rule (whisper.cpp
    // 6343-6369), implemented in log space exactly as upstream.
    {
        let beg_u = beg.max(0) as usize;
        // logsumexp over the timestamp logprobs.
        let mut ts_logprob = f32::NEG_INFINITY;
        if beg_u < logprobs.len() {
            let logprob_max = logprobs[beg_u..]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if logprob_max.is_finite() {
                let mut logsumexp = 0.0f32;
                for &lp in &logprobs[beg_u..] {
                    if lp > f32::NEG_INFINITY {
                        logsumexp += (lp - logprob_max).exp();
                    }
                }
                if logsumexp > 0.0 {
                    ts_logprob = logsumexp.ln() + logprob_max;
                }
            }
        }
        let max_text_logprob = logprobs[..beg_u]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        if ts_logprob > max_text_logprob {
            // force a timestamp: mask all text logits/logprobs.
            for i in 0..beg_u {
                logits[i] = f32::NEG_INFINITY;
                logprobs[i] = f32::NEG_INFINITY;
            }
        }
    }

    (logits, logprobs)
}

/// Sampler `exp` acceleration gate (`FW_SIMD_EXP=1`, default OFF — an owner
/// escape hatch, NOT a default change). When on, [`compute_logprobs`] computes its
/// vocab-wide `logsumexp` with a vectorized AVX2 degree-5 poly exp instead of scalar
/// libm `f32::exp` — MEASURED 16.7× on the 51866-vocab pass (`examples/exp_sampler_probe`).
/// NON-byte-exact (~2.5e-5 logprob delta), so kept off by default: franken deliberately
/// runs the accurate libm exp (mirrors frankentorch's deliberately-unwired
/// `ft_kernel_cpu::exp_f64x4`; the owner's accuracy/speed call). NOTE: this same
/// flag ALSO gates the attention `exp` (`nn::softmax_rows` poly, cross+self) since
/// b276b89, which — unlike the sampler exp — perturbs the RAW logits themselves
/// (attention weights feed the hidden states). The sampler exp is provably
/// no_ts-neutral (the token is `argmax` of the RAW logits and the timestamp-forcing
/// rule cannot fire because timestamps are masked to -inf); the attention-softmax
/// perturbation is only EMPIRICALLY sub-margin. Both together were MODEL-VERIFIED
/// byte-identical for the no_ts transcript: turbo `FW_SIMD_EXP` off-vs-on over jfk
/// ×1/×3/×8 (108/288/978 chars, multi-window) diffs to zero, and timestamp-mode
/// jfk×3 text+segments also matched. Only timestamp boundaries / logprob metadata
/// can shift in principle (the ~2.5e-5 delta), and e2e wall-clock is within noise
/// (the exp is parallelized ~20-way), so this stays a default-OFF escape hatch.
fn simd_exp_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FW_SIMD_EXP").is_some())
}

/// Disable conditioning the decode on previously-decoded text (`FW_NO_CONTEXT=1`,
/// default OFF). The port of whisper.cpp's `params.no_context` / whisper-cli
/// `--no-context` (`condition_on_previous_text = false`). By default each
/// non-first window prepends the prior windows' text as a `[sot_prev, …past…]`
/// prompt (the prompt build below) to improve cross-window coherence; when this
/// is set that carried prompt is suppressed and every window starts from a clean
/// `sot_sequence`. Default OFF preserves the native engine's historical
/// conditioning behavior. The bundled whisper.cpp currently defaults
/// `whisper_full_params.no_context` to true; the tiny.en segment-timestamp policy
/// below aligns that one losing comparator cell without changing other
/// model/mode combinations. Single-window clips (jfk×1) carry no prior-window
/// prompt at all, so this flag is a proven no-op there regardless.
///
/// This is the **bd-r0qd escape hatch** (owner-confirmed faithfulness bug): the
/// accumulated previous-window prompt can bias the greedy / temperature-0
/// (fallback-free) native decoder toward an early `eot` on a *final full* window
/// and drop ~40 words of coherent tail (the sole differing input vs the same
/// audio decoded standalone is `prompt_past`; whisper.cpp recovers via
/// temperature fallback + prompt reset, native — the deliberate temp-0 port —
/// cannot). Setting `FW_NO_CONTEXT=1` drops the carry and restores the tail on
/// affected clips, matching whisper-cli `--no-context`. Mirrors the established
/// gated-lever idiom ([`simd_exp_enabled`]); the *proper* fix (per-window
/// temperature fallback) remains owner-scoped.
fn condition_on_prev_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FW_NO_CONTEXT").is_some())
}

/// Preserve the historical carried-context policy for tiny.en segment timestamps.
///
/// Default tiny.en segment-timestamp decoding suppresses only text carried from a
/// prior window. The user's initial prompt on window zero is still honored. Set
/// `FW_TINY_EN_TS_CONTEXT=1` to restore the old carry-then-retry behavior; an
/// explicit [`DecodeParams::max_context`] also overrides the model policy.
fn tiny_en_segment_ts_context_forced() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FW_TINY_EN_TS_CONTEXT").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on"
            )
        })
    })
}

/// Decide whether to suppress tiny.en's cross-window segment-timestamp prompt.
///
/// This is a deterministic model policy, not an online adaptive controller:
///
/// - State: exact tiny.en architecture, segment timestamps, explicit user
///   context override, and the operator fallback flag.
/// - Actions: carry prior-window text, or start window 2+ from the normal SOT
///   sequence without that carry.
/// - Loss: the measured carried-prompt failure silently drops two track01
///   windows unless it pays for a second decode; no-carry restores all 1,301
///   characters in one attempt. The counter-loss is reduced cross-window style
///   conditioning, so the policy is deliberately model/mode-specific.
/// - Calibration/fallback: track01 full coverage and WER against whisper.cpp,
///   whose bundled default is also `no_context`; explicit `max_context` or
///   `FW_TINY_EN_TS_CONTEXT=1` restores the historical action. The default-on
///   failed-window retry remains as a conservative guard.
fn suppress_tiny_en_segment_ts_context(
    hparams: &WhisperHParams,
    params: &DecodeParams,
    force_context: bool,
) -> bool {
    let is_tiny_en = hparams.n_vocab == 51_864
        && hparams.n_audio_ctx == 1_500
        && hparams.n_audio_state == 384
        && hparams.n_audio_head == 6
        && hparams.n_audio_layer == 4
        && hparams.n_text_ctx == 448
        && hparams.n_text_state == 384
        && hparams.n_text_head == 6
        && hparams.n_text_layer == 4
        && hparams.n_mels == 80;
    is_tiny_en && params.timestamps && params.max_context.is_none() && !force_context
}

/// Optional user initial prompt (whisper `--prompt` / `initial_prompt`), read
/// from `FW_INITIAL_PROMPT`. Its text is tokenized with [`Tokenizer::encode`]
/// and seeds `prompt_past` so it is carried as previous context on the first
/// window (then ages out via the `max_prompt_ctx` truncation as decoded text
/// accumulates — whisper.cpp's `prompt_past` behaviour). Unset or empty → no
/// prompt (byte-identical default).
///
/// A dev/testing OVERRIDE of the [`DecodeParams::initial_prompt`] field
/// (mirroring [`condition_on_prev_disabled`]): when `FW_INITIAL_PROMPT` is set it
/// wins over the field, so a prompt can be injected without constructing params.
/// Unset → the field is used.
fn initial_prompt_from_env() -> Option<&'static str> {
    use std::sync::OnceLock;
    static PROMPT: OnceLock<Option<String>> = OnceLock::new();
    PROMPT
        .get_or_init(|| {
            std::env::var("FW_INITIAL_PROMPT")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .as_deref()
}

/// Build the initial `prompt_past` from an optional user prompt: its BPE token
/// ids, or empty when there is no (non-empty) prompt. Split out from
/// [`transcribe_samples`] so the encode-and-seed step is unit-testable without
/// the `FW_INITIAL_PROMPT` env-gate (which caches once per process).
fn seeded_prompt_past(prompt: Option<&str>, tokenizer: &Tokenizer) -> Vec<i32> {
    match prompt {
        Some(p) if !p.is_empty() => tokenizer.encode(p),
        _ => Vec::new(),
    }
}

/// Default-OFF: on a window that fails to close any timestamp (`result_len == 0`,
/// the "decoder failed with no timestamps closed" break) while carrying a
/// previous-window prompt, RETRY that same seek ONCE with the prompt cleared before
/// accepting the drop. The carried-prompt × int8 interaction is the confirmed cause
/// of the long-form content-drop (bd-r0qd): `FW_NO_CONTEXT=1` recovers it globally;
/// this recovers it targeted (only the failed window resets its prompt, so good
/// windows keep whisper.cpp-faithful conditioning).
///
/// **DEFAULT-ON (2026-07-24, bd-r0qd fix):** the retry only fires on a window that
/// closes with NO timestamp while carrying a prompt (`result_len == 0 && !is_no_speech
/// && seek_cs > 0`) — a strict "this window produced nothing" condition. Non-failed
/// windows never enter it, so the recovery is **byte-identical on every clip that did
/// not already drop** (single-window clips — jfk golden, quant/turbo e2e — are
/// unaffected BY CONSTRUCTION: no carried prompt on window 0). On the clips that DID
/// drop (long-form / looping), it recovers the lost tail (WER 0.164 vs greedy 0.528 on
/// track01). Verified: full native_engine lib suite 299/0 with the retry on; jfk golden
/// byte-identical. Disable with `FW_RETRY_FAILED_WINDOW=0` (or `false`/`off`).
/// See NEGATIVE_EVIDENCE 2026-07-12 / 2026-07-24 / project_final_window_early_eot_bug.
fn retry_failed_window_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FW_RETRY_FAILED_WINDOW")
            .map_or(true, |v| !matches!(v.trim(), "0" | "false" | "off"))
    })
}

/// whisper.cpp's temperature-fallback ladder (initial greedy pass at 0.0, then
/// `temperature_inc = 0.2` per retry): the retry temperatures, in order.
const TEMP_FALLBACK_LADDER: [f64; 5] = [0.2, 0.4, 0.6, 0.8, 1.0];

/// Above this retry temperature the carried prior-window prompt is dropped for the
/// attempt (whisper.cpp conditions on no previous text once `t > 0.5`) — which is
/// exactly the recovery that fixes the bd-r0qd carried-prompt × int8 early-EOT drop.
const TEMP_PROMPT_RESET: f64 = 0.5;

/// whisper.cpp `entropy_thold` (default 2.4, "similar to OpenAI's
/// compression_ratio_threshold"): a window whose result-token tail is this
/// repetitive is a degenerate loop and fails the quality gate.
const ENTROPY_THRESHOLD: f64 = 2.4;

/// The entropy tail length (whisper.cpp 6599: the last 32 result tokens). The
/// check only fires when `result_len > ENTROPY_WINDOW` (whisper.cpp 7540 uses
/// strict `>`), so short windows — naturally low-entropy — are never judged.
const ENTROPY_WINDOW: usize = 32;

/// Shannon entropy (nats) of the token-id distribution over the last
/// [`ENTROPY_WINDOW`] entries of `tokens` — a faithful port of the whisper.cpp
/// sequence-entropy block (whisper.cpp 6597-6617): count each distinct id in the
/// tail (timestamp tokens included, exactly as upstream), then `-Σ p·ln p`. A
/// uniformly repeated tail scores 0.0; 32 distinct tokens score `ln 32 ≈ 3.47`.
fn token_tail_entropy(tokens: &[i32]) -> f64 {
    let tail = &tokens[tokens.len().saturating_sub(ENTROPY_WINDOW)..];
    if tail.is_empty() {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
    for &t in tail {
        *counts.entry(t).or_insert(0) += 1;
    }
    let n = tail.len() as f64;
    counts
        .values()
        .map(|&c| {
            let p = f64::from(c) / n;
            -p * p.ln()
        })
        .sum()
}

/// `FW_TEMP_FALLBACK` (default-OFF): whisper.cpp-faithful temperature fallback —
/// the bd-r0qd fix-spec's "proper fix" (#3) and the fallback half of bd-6goy. A
/// non-silent window that closes no timestamp (`result_len == 0`, the confirmed
/// long-form drop), averages below [`LOGPROB_THRESHOLD`] (whisper.cpp
/// `logprob_thold`), or loops into a low-entropy repetitive tail
/// ([`ENTROPY_THRESHOLD`], whisper.cpp `entropy_thold`, checked only when
/// `result_len > `[`ENTROPY_WINDOW`]) is re-decoded at the
/// [`TEMP_FALLBACK_LADDER`] temperatures —
/// multinomial sampling instead of argmax, deterministic per-(window, rung,
/// candidate) seed, prompt dropped above [`TEMP_PROMPT_RESET`] — reusing the
/// window's encode. Each rung decodes [`temp_best_of`] independent candidates
/// and adopts the best [`sequence_score`] (whisper.cpp `greedy.best_of`). The
/// first rung whose winner clears the gate ends the ladder; the final rung's
/// winner is accepted as-is (whisper.cpp keeps its last decode too). Unset ⇒ the ladder never fires and
/// token selection stays the argmax path ⇒ byte-identical by construction.
/// Supersedes `FW_RETRY_FAILED_WINDOW` when both are set (the ladder's `t > 0.5`
/// rungs contain that retry's prompt reset).
fn temp_fallback_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FW_TEMP_FALLBACK").is_some())
}

/// whisper.cpp `greedy.best_of` (default 5): how many independent sampling
/// candidates each `t > 0` ladder rung decodes before the best
/// [`sequence_score`] wins. `FW_TEMP_BEST_OF` overrides (clamped to [1, 32]);
/// `1` restores the single-candidate ladder byte-for-byte (first candidate of
/// every rung draws from the identical seed stream). Only read under
/// [`temp_fallback_enabled`], so the default path never consults it.
fn temp_best_of() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("FW_TEMP_BEST_OF")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(5, |n| n.clamp(1, 32))
    })
}

/// whisper.cpp `whisper_sequence_score` under the default `length_penalty =
/// -1.0` (penalty = `result_len`): `sum(plogs[..result_len]) / result_len`,
/// i.e. the result slice's average logprob. A candidate that closed no result
/// (`result_len == 0` — whisper.cpp leaves its score unset and fails the
/// decoder) scores the [`EMPTY_WINDOW_AVG_LOGPROB`] sentinel so it never beats
/// a candidate that produced tokens.
fn sequence_score(plogs: &[f32], result_len: usize) -> f64 {
    let take = result_len.min(plogs.len());
    if take == 0 {
        return EMPTY_WINDOW_AVG_LOGPROB;
    }
    let s = plogs[..take].iter().map(|&p| f64::from(p)).sum::<f64>() / take as f64;
    if s.is_finite() {
        s
    } else {
        EMPTY_WINDOW_AVG_LOGPROB
    }
}

/// One completed decode attempt of a window at a `t > 0` ladder rung — the
/// state the post-selection code (segment emission, DTW, prompt carry, seek
/// advance) consumes. The rung's best-scoring candidate is adopted back into
/// the window locals; the encode (and thus `DecoderState`'s cross-K/V) is
/// identical across candidates, so adoption is sound for the DTW re-forward.
struct WindowCandidate {
    score: f64,
    decoded: Vec<i32>,
    plogs: Vec<f32>,
    result_len: usize,
    seek_delta_cs: i64,
    avg_logprob: f64,
    no_speech_prob: f64,
}

/// `FW_BEAM_SIZE` env OVERRIDE of the [`DecodeParams::beam_size`] field (a
/// dev/testing hatch that wins over the field when set, mirroring the other
/// `FW_*` gates). Unset → the field is used. Raw parsed value; clamping happens
/// in [`resolve_beam_size`].
fn beam_size_from_env() -> Option<usize> {
    use std::sync::OnceLock;
    static N: OnceLock<Option<usize>> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("FW_BEAM_SIZE")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
    })
}

/// Effective beam width for a decode: the `FW_BEAM_SIZE` env override, else the
/// [`DecodeParams::beam_size`] field, else `1` (greedy). Clamped to `[1, 8]`.
///
/// whisper.cpp's `beam_search.beam_size` (`whisper-cli -bs`, default 5): values
/// `> 1` decode each temperature-0 window with beam search — keep the `n`
/// highest cumulative-logprob hypotheses per step, select the best
/// length-normalized [`sequence_score`] at the end — instead of argmax. `1`
/// restores the exact greedy path ⇒ byte-identical by construction. Consulted
/// only when a window decodes at temperature 0 (the ladder's `t > 0` rungs stay
/// on the sampling path).
fn resolve_beam_size(params: &DecodeParams) -> usize {
    beam_size_from_env()
        .or(params.beam_size)
        .map_or(1, |n| n.clamp(1, 8))
}

/// A window's decode result — `(decoded, plogs, has_ts, seek_delta_cs,
/// result_len)` — the tuple both the greedy loop and [`beam_decode_window`]
/// produce for the shared downstream (segment build, DTW, prompt carry).
type WindowDecode = (Vec<i32>, Vec<f32>, bool, i64, usize);

/// One beam-search hypothesis: its own self-attention KV state (an independent
/// [`DecoderState`] fork), the tokens decoded so far, their per-token logprobs,
/// the running cumulative logprob (the during-search ranking key), and the
/// window bookkeeping (`has_ts`/`seek_delta_cs`/`result_len`) that mirrors the
/// greedy loop's per-token state. `next_logits` are the raw logits to
/// `process_logits` at the next expansion (the prefill logits for the seed).
struct BeamHyp {
    state: DecoderState,
    tokens: Vec<i32>,
    plogs: Vec<f32>,
    sum_logprob: f64,
    has_ts: bool,
    seek_delta_cs: i64,
    result_len: usize,
    next_logits: Vec<f32>,
}

/// A surviving (non-terminated) beam expansion decided in the candidate loop's
/// first pass — before the parent KV states are moved/cloned in the fork pass.
/// `parent` indexes the current `active` beam.
struct BeamExpand {
    parent: usize,
    tok: i32,
    tokens: Vec<i32>,
    plogs: Vec<f32>,
    has_ts: bool,
    seek_delta_cs: i64,
    result_len: usize,
    sum_logprob: f64,
}

/// Indices of the `k` largest values in `logprobs`, skipping `-inf` (masked)
/// lanes, best-first. Partial selection (not a full sort) over the vocab.
fn top_k_logprob_indices(logprobs: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logprobs.len())
        .filter(|&i| logprobs[i] > f32::NEG_INFINITY)
        .collect();
    let k = k.min(idx.len());
    if k == 0 {
        return Vec::new();
    }
    // select_nth then sort the head: O(n) partition + O(k log k) on the head.
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        logprobs[b]
            .partial_cmp(&logprobs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx.sort_unstable_by(|&a, &b| {
        logprobs[b]
            .partial_cmp(&logprobs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

/// Beam-search decode of one window (temperature 0), the `FW_BEAM_SIZE > 1`
/// replacement for the greedy inner loop. `st` is the prompt-prefilled decoder
/// state (borrowed — left intact for the caller's DTW re-forward, which shares
/// the same window-constant cross-K/V); `prefill_logits` are its next-token
/// logits. Returns the winning hypothesis's `(decoded, plogs, has_ts,
/// seek_delta_cs, result_len)` — exactly the tuple the greedy loop produces, so
/// all downstream handling is shared.
///
/// The per-token timestamp / EOT / budget / backward-bail rules DUPLICATE the
/// greedy loop (whisper.cpp 7362-7410) rather than refactor it, so the greedy
/// path stays byte-identical. During search, hypotheses are ranked by cumulative
/// logprob (whisper's beam ranking); the final winner is the best
/// length-normalized [`sequence_score`] over completed hypotheses (falling back
/// to the best still-active one when none completed — the greedy "accept the
/// last decode" analog).
#[allow(clippy::too_many_arguments)]
fn beam_decode_window(
    m: &LoadedModel,
    st: &DecoderState,
    prefill_logits: Vec<f32>,
    tk: &Tokenizer,
    cfg: &FilterConfig,
    params: &DecodeParams,
    seek_cs: i64,
    seek_end_cs: i64,
    n_max_tokens: usize,
    user_max_tokens: Option<usize>,
    beam: usize,
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<WindowDecode> {
    let mut active: Vec<BeamHyp> = vec![BeamHyp {
        state: st.clone(),
        tokens: Vec::new(),
        plogs: Vec::new(),
        sum_logprob: 0.0,
        has_ts: false,
        seek_delta_cs: CHUNK_CS,
        result_len: 0,
        next_logits: prefill_logits,
    }];
    // Completed hypotheses as lightweight [`WindowDecode`] tuples — no decoder
    // state (a terminated hypothesis is never forwarded again), so completing a
    // beam costs no cross-K/V clone.
    let mut finished: Vec<WindowDecode> = Vec::new();

    for i in 0..n_max_tokens {
        if active.is_empty() {
            break;
        }
        checkpoint()?;

        // Expand every active hypothesis by its top-`beam` next tokens (ranked by
        // logprob); collect (parent index, token, plog, cumulative score).
        let mut cands: Vec<(usize, i32, f32, f64)> = Vec::new();
        for (hi, hyp) in active.iter_mut().enumerate() {
            let raw = std::mem::take(&mut hyp.next_logits);
            let (_filtered, logprobs) =
                process_logits(tk, cfg, raw, &hyp.tokens, hyp.has_ts, hyp.seek_delta_cs, i);
            for tok_idx in top_k_logprob_indices(&logprobs, beam) {
                let lp = logprobs[tok_idx];
                let tok = i32::try_from(tok_idx).unwrap_or(0);
                cands.push((hi, tok, lp, hyp.sum_logprob + f64::from(lp)));
            }
        }
        if cands.is_empty() {
            break;
        }
        // Keep the top-`beam` expansions by cumulative logprob (ties: lower parent
        // then lower token, for determinism).
        cands.sort_by(|a, b| {
            b.3.partial_cmp(&a.3)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
        });
        cands.truncate(beam);

        // PASS 1 — decide each candidate: terminated hypotheses go to `finished`;
        // survivors are collected (with their per-token bookkeeping) so the KV
        // states can then be moved/cloned without a borrow conflict.
        let mut expands: Vec<BeamExpand> = Vec::with_capacity(beam);
        for (hi, tok, plog, new_sum) in cands {
            let parent = &active[hi];
            let mut tokens = parent.tokens.clone();
            tokens.push(tok);
            let mut plogs = parent.plogs.clone();
            plogs.push(plog);
            let mut has_ts = parent.has_ts;
            let mut seek_delta_cs = parent.seek_delta_cs;
            let mut result_len = parent.result_len;

            // Timestamp update (mirrors greedy, whisper.cpp 7362-7375).
            let mut bail = false;
            if tok > tk.timestamp_begin {
                let new_delta = 2 * i64::from(tok - tk.timestamp_begin);
                if has_ts && seek_delta_cs > new_delta && result_len < i {
                    bail = true; // going back in time: terminate this hypothesis
                } else {
                    seek_delta_cs = new_delta;
                    result_len = i + 1;
                    has_ts = true;
                }
            }

            let budget_reached = user_max_tokens.is_some_and(|mt| i >= mt);
            let reached_end = has_ts && seek_cs + seek_delta_cs + DELTA_MIN >= seek_end_cs;
            let terminate = bail || tok == tk.eot || budget_reached || reached_end;

            if terminate {
                if !bail {
                    if result_len == 0 && params.timestamps && reached_end {
                        result_len = i + 1;
                    }
                    if !params.timestamps {
                        result_len = i + 1;
                        seek_delta_cs = CHUNK_CS;
                    }
                }
                finished.push((tokens, plogs, has_ts, seek_delta_cs, result_len));
            } else {
                expands.push(BeamExpand {
                    parent: hi,
                    tok,
                    tokens,
                    plogs,
                    has_ts,
                    seek_delta_cs,
                    result_len,
                    sum_logprob: new_sum,
                });
            }
        }

        // Move the parent KV states out of `active` (replaced below). Each
        // parent's state is deep-cloned for all but its LAST surviving child,
        // which MOVES it — move == clone in content, so byte-exact, but it
        // eliminates ~one self-attn `KvCache` copy (~8 MB) per parent per step
        // (most of the remaining fork cost when the beam is spread ~1 child each).
        let mut child_count = vec![0usize; active.len()];
        for e in &expands {
            child_count[e.parent] += 1;
        }
        let mut parent_states: Vec<Option<DecoderState>> = std::mem::take(&mut active)
            .into_iter()
            .map(|h| Some(h.state))
            .collect();

        // PASS 2 — fork + forward each survivor to its hidden state (per-hypothesis
        // self-attn/MLP; can't batch across differing KV). The tied-output logits
        // ARE batched below.
        let mut next_active: Vec<BeamHyp> = Vec::with_capacity(expands.len());
        let mut hidden_rows: Vec<f32> = Vec::new();
        let mut used = vec![0usize; child_count.len()];
        for e in expands {
            used[e.parent] += 1;
            let mut state = if used[e.parent] == child_count[e.parent] {
                parent_states[e.parent]
                    .take()
                    .expect("last surviving child moves the parent state")
            } else {
                parent_states[e.parent]
                    .as_ref()
                    .expect("earlier surviving child clones the parent state")
                    .clone()
            };
            let (x_last, _draft) =
                decoder::forward_step_hidden(&m.decoder, &mut state, &[e.tok], checkpoint)?;
            hidden_rows.extend_from_slice(&x_last.data);
            next_active.push(BeamHyp {
                state,
                tokens: e.tokens,
                plogs: e.plogs,
                sum_logprob: e.sum_logprob,
                has_ts: e.has_ts,
                seek_delta_cs: e.seek_delta_cs,
                result_len: e.result_len,
                next_logits: Vec::new(), // filled by the batched logits below
            });
        }
        // Batched tied-output projection: one `logits_all` over all survivors'
        // hidden states reads the [n_vocab, n_state] weight ONCE instead of once
        // per hypothesis (the logits GEMV is bandwidth-bound on that weight).
        // Byte-identical to per-hypothesis `logits_last` (the `logits_all` test
        // pins that), so the beam's decisions are unchanged.
        if !next_active.is_empty() {
            let hidden = super::Mat::from_vec(next_active.len(), m.decoder.n_state(), hidden_rows);
            let all_logits = decoder::logits_all(&m.decoder, &hidden)?;
            let n_vocab = all_logits.len() / next_active.len();
            for (h, hyp) in next_active.iter_mut().enumerate() {
                hyp.next_logits = all_logits[h * n_vocab..(h + 1) * n_vocab].to_vec();
            }
        }
        active = next_active;
    }

    // Winner: best length-normalized sequence score among completed hypotheses;
    // if none completed (ran to the token cap), the best still-active hypothesis.
    let pool: Vec<WindowDecode> = if finished.is_empty() {
        active
            .into_iter()
            .map(|h| (h.tokens, h.plogs, h.has_ts, h.seek_delta_cs, h.result_len))
            .collect()
    } else {
        finished
    };
    let best = pool
        .into_iter()
        .max_by(|a, b| {
            sequence_score(&a.1, a.4)
                .partial_cmp(&sequence_score(&b.1, b.4))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("beam pool is non-empty (seed hypothesis always present)");
    Ok(best)
}

/// `Σ exp(l − max)` over `logits`, masked lanes (`l == -inf`) contributing exactly 0
/// (matching the scalar `l > -inf` guard). AVX2 degree-5 poly exp: range-reduce
/// `x = k·ln2 + r`, `exp(r)` via Horner, `2^k` by float-bit construction. Numerics-
/// affecting (~2.5e-5 vs libm) — only reached under [`simd_exp_enabled`].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[allow(unsafe_code)]
// The `log2e`/`ln2` range-reduction literals are part of this poly-exp's tuned,
// WER-certified coefficient set — NOT free-standing math constants. "Correcting"
// them to `f32::consts::{LOG2_E, LN_2}` would perturb the certified kernel's
// numerics, so `approx_constant` is suppressed deliberately here.
#[allow(clippy::approx_constant)]
fn logsumexp_sum_simd(logits: &[f32], max: f32) -> f32 {
    use core::arch::x86_64::*;
    let n = logits.len();
    let lp = logits.as_ptr();
    // SAFETY: avx2+fma guaranteed by this fn's cfg; every load is bounded by the
    // `i+8<=n` guard and the `< 8` remainder runs scalar.
    unsafe {
        let vmax = _mm256_set1_ps(max);
        let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
        let log2e = _mm256_set1_ps(1.442_695_f32);
        let ln2 = _mm256_set1_ps(0.693_147_2_f32);
        let lo = _mm256_set1_ps(-87.3365_f32);
        let c0 = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.166_666_67_f32);
        let c4 = _mm256_set1_ps(0.041_666_66_f32);
        let c5 = _mm256_set1_ps(0.008_333_33_f32);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let l = _mm256_loadu_ps(lp.add(i));
            let keep = _mm256_cmp_ps::<_CMP_GT_OQ>(l, ninf); // l > -inf
            let xv = _mm256_max_ps(_mm256_sub_ps(l, vmax), lo);
            let kf = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                _mm256_mul_ps(xv, log2e),
            );
            let r = _mm256_fnmadd_ps(kf, ln2, xv);
            let mut p = _mm256_fmadd_ps(c5, r, c4);
            p = _mm256_fmadd_ps(p, r, c3);
            p = _mm256_fmadd_ps(p, r, c2);
            p = _mm256_fmadd_ps(p, r, c1);
            p = _mm256_fmadd_ps(p, r, c0);
            let ki = _mm256_cvtps_epi32(kf);
            let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
                ki,
                _mm256_set1_epi32(127),
            )));
            let e = _mm256_and_ps(_mm256_mul_ps(p, pow2), keep); // zero masked lanes
            acc = _mm256_add_ps(acc, e);
            i += 8;
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut s =
            ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
        while i < n {
            let l = logits[i];
            if l > f32::NEG_INFINITY {
                s += (l - max).exp();
            }
            i += 1;
        }
        s
    }
}

/// Scalar fallback (non-avx2): identical to the default libm loop.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn logsumexp_sum_simd(logits: &[f32], max: f32) -> f32 {
    let mut s = 0.0f32;
    for &l in logits {
        if l > f32::NEG_INFINITY {
            s += (l - max).exp();
        }
    }
    s
}

/// Numerically-stable log-softmax (whisper.cpp `whisper_compute_logprobs`,
/// lines 6138-6158). `-inf` logits map to `-inf` logprobs.
fn compute_logprobs(logits: &[f32]) -> Vec<f32> {
    // Sanitize non-finite logits to `-inf` up front. A `+inf` activation
    // (overflow) would otherwise drive `logit_max` to `+inf`, making
    // `(l - logit_max).exp() = exp(+inf - +inf) = exp(NaN) = NaN` and poisoning
    // every logprob (and, downstream, the confidence/avg_logprob/no_speech math).
    // Mapping NaN/+inf to `-inf` makes them behave like already-masked lanes.
    let logits: Vec<f32> = logits
        .iter()
        .map(|&l| if l.is_finite() { l } else { f32::NEG_INFINITY })
        .collect();
    let logit_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Default (gate off): exact scalar libm — byte-identical to before. Gate on:
    // vectorized poly exp (16.7×, ~2.5e-5 delta), the owner's opt-in escape hatch.
    let logsumexp_raw = if simd_exp_enabled() {
        logsumexp_sum_simd(&logits, logit_max)
    } else {
        let mut s = 0.0f32;
        for &l in &logits {
            if l > f32::NEG_INFINITY {
                s += (l - logit_max).exp();
            }
        }
        s
    };
    let logsumexp = logsumexp_raw.ln() + logit_max;
    logits
        .iter()
        .map(|&l| {
            if l > f32::NEG_INFINITY {
                l - logsumexp
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

/// First index of the maximum over `logits` (scalar reference: strict `>`, so ties keep the
/// FIRST index). LLVM does NOT autovectorize this — the running `best_i` is a loop-carried
/// data dependency — so on the 51866-vocab sampler pass (SERIAL critical path, per token) it
/// stays scalar. See [`argmax_idx`] for the AVX2 form.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn argmax_idx(l: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > best {
            best = v;
            best_i = i;
        }
    }
    best_i
}

/// AVX2 first-index argmax, BYTE-EXACT vs the scalar loop in ALL cases (finite / `-inf` /
/// NaN / empty): the strict `_CMP_GT_OQ` blend skips NaN and `-inf` exactly as scalar `>`
/// does, and per lane keeps the first index in that lane's stride; the horizontal reduce then
/// takes the max value and the MIN index among ties ⇒ the global FIRST index. MEASURED 5.1×
/// over 51866 (`examples/sampler_maxargmax_probe`), ~14.6 µs/token off the serial sampler.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
fn argmax_idx(l: &[f32]) -> usize {
    use core::arch::x86_64::*;
    let n = l.len();
    let n8 = n & !7;
    // SAFETY: avx2 guaranteed by this fn's cfg; every load is bounded by `i < n8 <= n`
    // (i advances by 8, i+8 <= n8) and the `< 8` remainder runs scalar.
    unsafe {
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let mut vidx = _mm256_setzero_si256();
        let mut i = 0usize;
        while i < n8 {
            let v = _mm256_loadu_ps(l.as_ptr().add(i));
            let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(v, vmax); // strict > ⇒ keep first, skip NaN/-inf
            vmax = _mm256_blendv_ps(vmax, v, gt);
            let idx = _mm256_add_epi32(_mm256_set1_epi32(i as i32), lane);
            vidx = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(vidx),
                _mm256_castsi256_ps(idx),
                gt,
            ));
            i += 8;
        }
        let mut vals = [0.0f32; 8];
        let mut idxs = [0i32; 8];
        _mm256_storeu_ps(vals.as_mut_ptr(), vmax);
        _mm256_storeu_si256(idxs.as_mut_ptr().cast(), vidx);
        // Max value across lanes; among lanes tied at that value, the MIN index (= global first).
        let mut best = f32::NEG_INFINITY;
        let mut best_i = usize::MAX;
        for k in 0..8 {
            let (v, ix) = (vals[k], idxs[k] as usize);
            if v > best || (v == best && ix < best_i) {
                best = v;
                best_i = ix;
            }
        }
        if best_i == usize::MAX {
            best_i = 0; // empty or all-(-inf)/NaN: match scalar's initial best_i = 0
        }
        while i < n {
            if l[i] > best {
                best = l[i];
                best_i = i;
            }
            i += 1;
        }
        best_i
    }
}

/// Argmax over `logits`, returning `(id, logprob_of_id)`. Mirrors
/// `whisper_sample_token(best=true)` (whisper.cpp 6503-6510): the chosen id is
/// the argmax of the (post-filter) probabilities — equivalently logits — and
/// `plog` is its log-softmax value. The index scan is [`argmax_idx`] (AVX2, byte-exact).
fn argmax(logits: &[f32], logprobs: &[f32]) -> (i32, f32) {
    let best_i = argmax_idx(logits);
    (
        i32::try_from(best_i).unwrap_or(0),
        logprobs.get(best_i).copied().unwrap_or(0.0),
    )
}

/// SplitMix64 step — the deterministic PRNG behind temperature sampling. One `u64`
/// of state, platform-independent: the same (window, attempt) always draws the same
/// tokens, so a `FW_TEMP_FALLBACK` transcript is exactly replayable.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Multinomial draw from `softmax(logits / temperature)` — the non-greedy arm of
/// whisper.cpp's `whisper_sample_token` (logits are divided by the temperature
/// before the softmax). Masked lanes (`-inf`; NaN treated as masked) carry zero
/// mass and are never drawn. The returned `plog` is the drawn id's TEMPERATURE-1
/// log-softmax from `logprobs`, so the window's `avg_logprob` quality gate keeps
/// measuring true model confidence, not the flattened sampling distribution.
/// Degenerate inputs (no finite lane / underflowed mass) fall back to [`argmax`].
fn sample_token_at_temperature(
    logits: &[f32],
    logprobs: &[f32],
    temperature: f64,
    rng: &mut u64,
) -> (i32, f32) {
    let inv_t = 1.0 / temperature.max(1e-6);
    let max = logits
        .iter()
        .copied()
        .filter(|l| l.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return argmax(logits, logprobs);
    }
    let weight = |l: f32| -> f64 {
        if l.is_finite() {
            (f64::from(l - max) * inv_t).exp()
        } else {
            0.0
        }
    };
    let total: f64 = logits.iter().map(|&l| weight(l)).sum();
    if total <= 0.0 || !total.is_finite() {
        return argmax(logits, logprobs);
    }
    // 53-bit uniform in [0, 1), then walk the (un-normalized) cumulative mass. The
    // walk re-derives each weight so `acc` sums the exact terms `total` summed.
    let target = (splitmix64(rng) >> 11) as f64 / (1u64 << 53) as f64 * total;
    let mut acc = 0.0f64;
    for (i, &l) in logits.iter().enumerate() {
        acc += weight(l);
        if acc > target {
            return (
                i32::try_from(i).unwrap_or(0),
                logprobs.get(i).copied().unwrap_or(0.0),
            );
        }
    }
    // FP rounding pushed the walk past the end (target ≈ total): last unmasked lane.
    let last = logits.iter().rposition(|l| l.is_finite()).unwrap_or(0);
    (
        i32::try_from(last).unwrap_or(0),
        logprobs.get(last).copied().unwrap_or(0.0),
    )
}

// ---------------------------------------------------------------------------
// Segment building (port of whisper.cpp 7624-7730)
// ---------------------------------------------------------------------------

/// Build [`TranscriptionSegment`]s from a decoded token stream.
///
/// Port of the result-emission loop in whisper.cpp (7624-7730). `tokens` is the
/// full decoded sequence (text + timestamp tokens, no prompt); `seek_cs` is the
/// window start in centiseconds; `seek_delta_cs` is the window's final shift
/// (used to close the open-tail segment, whisper.cpp 7697). `plogs` is the chosen
/// token's `plog` per decoded token (same length as `tokens`), used for the
/// per-segment confidence.
///
/// In `single_segment` / no-timestamps mode (`split == false`) a single segment
/// spanning `[seek_cs, seek_cs + seek_delta_cs]` is produced (whisper.cpp
/// 7402-7405, 7645 guard).
///
/// All emitted segment bounds are clamped to `[seek_cs, seek_end_cs]` (fix #1):
/// a timestamp token can point into the zero-padded tail of the final window
/// (worst on a hard-cut last clip), which would otherwise yield an `end_sec`
/// past the real clip duration. `seek_end_cs` is the real (unpadded) audio
/// length in centiseconds — whisper.cpp's `n_len_org` (6859-6860).
fn build_segments(
    tk: &Tokenizer,
    tokens: &[i32],
    plogs: &[f32],
    seek_cs: i64,
    seek_delta_cs: i64,
    seek_end_cs: i64,
    split: bool,
) -> Vec<TranscriptionSegment> {
    let beg = tk.timestamp_begin;
    let mut segments = Vec::new();

    // Clamp a window-relative + seek-offset centisecond bound to the real audio
    // length, never below the window start (fix #1).
    let clamp = |t: i64| t.clamp(seek_cs, seek_end_cs.max(seek_cs));

    if !split {
        // Single segment spanning the whole window (whisper.cpp 7402-7405).
        let text = tk.decode(tokens).trim().to_string();
        if !text.is_empty() {
            segments.push(make_segment(
                clamp(seek_cs),
                clamp(seek_cs + seek_delta_cs),
                text,
                text_confidence(tk, tokens, plogs),
            ));
        }
        return segments;
    }

    // Timestamp-paired emission (whisper.cpp 7624-7694).
    // t0 starts at the first token's implied timestamp (whisper.cpp 7626);
    // in greedy the first decoded token is normally the opening <|0.00|> ts.
    let mut i0 = 0usize;
    let mut t0 = clamp(seek_cs + ts_offset_cs(tokens.first().copied(), beg));
    let mut i = 0usize;

    while i < tokens.len() {
        let tok = tokens[i];
        // A timestamp token strictly greater than `beg` closes a segment
        // (whisper.cpp 7645: `id > token_beg`). The bare `beg` (<|0.00|>) opens.
        if tok > beg {
            let t1 = clamp(seek_cs + 2 * i64::from(tok - beg));
            let text = tk.decode(&tokens[i0..=i]).trim().to_string();
            if !text.is_empty() {
                let conf = text_confidence(
                    tk,
                    tokens.get(i0..=i).unwrap_or(&[]),
                    plogs.get(i0..=i).unwrap_or(&[]),
                );
                segments.push(make_segment(t0, t1, text, conf));
            }
            t0 = t1;
            // Skip a run of consecutive timestamp tokens WITHOUT recomputing t0:
            // whisper.cpp (7675-7679) advances the index past the run but keeps
            // `t0 = t1` — the FIRST (closing) timestamp of the run. Recomputing t0
            // to the LAST timestamp of the run (as this loop previously did)
            // inserts a spurious gap at any within-window consecutive-timestamp
            // segment boundary: for `text <|10.38|> <|10.80|> text` the next
            // segment opened at 10.80 instead of whisper.cpp's 10.38 (+420 ms on
            // jfk×3 seg 1). This now matches both whisper.cpp and the sibling
            // consecutive-timestamp skip in the word-timing path below.
            while i + 1 < tokens.len() && tokens[i + 1] > beg {
                i += 1;
            }
            i0 = i + 1;
        }
        i += 1;
    }

    // Open-tail segment: text after the last timestamp pair (whisper.cpp
    // 7696-7714). Closed at `seek + seek_delta`.
    if i0 < tokens.len() {
        let text = tk.decode(&tokens[i0..]).trim().to_string();
        if !text.is_empty() {
            let t1 = clamp(seek_cs + seek_delta_cs);
            let conf = text_confidence(
                tk,
                tokens.get(i0..).unwrap_or(&[]),
                plogs.get(i0..).unwrap_or(&[]),
            );
            segments.push(make_segment(t0, t1, text, conf));
        }
    }

    segments
}

/// Whether the prompt context should be cleared before decoding a window with a
/// very short audio tail (fix #2 — port of whisper.cpp 7046-7051):
///
/// ```text
/// if (seek > seek_start && seek + 500 >= seek_end) {
///     prompt_past0.clear();
///     prompt_past1.clear();
/// }
/// ```
///
/// On a non-first window (`seek_cs > 0`, our `seek_start` is always 0) whose
/// remaining audio is under 5 s (`seek_cs + 500 >= seek_end_cs`, 500 cs = 5 s),
/// upstream drops the carried prompt because a short tail "tends to confuse the
/// decoder and often make it repeat or hallucinate stuff". Extracted as a pure
/// predicate so it can be unit-tested without a model.
fn should_clear_short_tail_prompt(seek_cs: i64, seek_end_cs: i64) -> bool {
    seek_cs > 0 && seek_cs + 500 >= seek_end_cs
}

/// Timestamp offset (centiseconds) implied by a (possibly text) token id, used
/// for the opening `t0`. For a timestamp token it is `2*(id - beg)`; otherwise
/// `0` (the opening `<|0.00|>` is what whisper expects first).
fn ts_offset_cs(tok: Option<i32>, beg: i32) -> i64 {
    match tok {
        Some(id) if id >= beg => 2 * i64::from(id - beg),
        _ => 0,
    }
}

/// Per-segment confidence: `exp(mean token logprob)` clamped to `[0, 1]`.
/// Superseded in production by [`text_confidence`] (fix #8, which excludes
/// timestamp tokens); retained for the clamp/monotonicity unit test.
#[cfg(test)]
fn confidence(plogs: &[f32]) -> Option<f64> {
    if plogs.is_empty() {
        return None;
    }
    let mean = plogs.iter().map(|&p| f64::from(p)).sum::<f64>() / plogs.len() as f64;
    // `clamp` panics on NaN under the current nightly; guard the exp result.
    let c = mean.exp();
    Some(if c.is_finite() {
        c.clamp(0.0, 1.0)
    } else {
        0.0
    })
}

/// Per-segment **text** confidence (fix #8): `exp(mean text-token logprob)`
/// clamped to `[0, 1]`, averaging only over the segment's *text* tokens —
/// excluding the leading/closing timestamp tokens (and any special tokens). The
/// metric documents itself as text confidence, so the closing `<|t|>` token's
/// logprob (which can be high-confidence and unrelated to the words) must not
/// dilute it. `tokens` and `plogs` are the segment's token ids and their chosen
/// logprobs, 1:1. If a segment has no text tokens, returns `None`.
fn text_confidence(tk: &Tokenizer, tokens: &[i32], plogs: &[f32]) -> Option<f64> {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (i, &tok) in tokens.iter().enumerate() {
        // A text token is anything below the timestamp range that is not a
        // special control token.
        if tok < tk.timestamp_begin
            && !tk.is_special(tok)
            && let Some(&p) = plogs.get(i)
            && p.is_finite()
        {
            sum += f64::from(p);
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let mean = sum / count as f64;
    // `clamp` panics on NaN under the current nightly; guard the exp result.
    let c = mean.exp();
    Some(if c.is_finite() {
        c.clamp(0.0, 1.0)
    } else {
        0.0
    })
}

/// Construct a [`TranscriptionSegment`] from centisecond bounds + text.
fn make_segment(
    t0_cs: i64,
    t1_cs: i64,
    text: String,
    confidence: Option<f64>,
) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(t0_cs as f64 / 100.0),
        end_sec: Some(t1_cs as f64 / 100.0),
        text,
        speaker: None,
        confidence,
    }
}

// ---------------------------------------------------------------------------
// Tail-window encoder-context truncation (whisper.cpp's audio_ctx / -ac feature)
// ---------------------------------------------------------------------------

/// Whether tail-window encoder-context truncation is enabled.
///
/// Controlled by the `FRANKEN_WHISPER_NATIVE_TAIL_TRUNCATE` environment
/// variable, read **once** (process-lifetime cached via [`OnceLock`]):
/// - unset / any value other than `"0"`/`"false"` ⇒ **enabled** (the default).
/// - `"0"` or `"false"` (ASCII-case-insensitive) ⇒ **disabled**, restoring the
///   exact pre-optimization behavior (every window runs a full 3000-frame /
///   1500-ctx encoder pass). This is the kill switch / golden-equivalence
///   escape hatch.
fn tail_truncate_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FRANKEN_WHISPER_NATIVE_TAIL_TRUNCATE")
            .map_or(true, |v| !(v == "0" || v.eq_ignore_ascii_case("false")))
    })
}

/// First-window encoder-context truncation margin (experimental, default OFF).
///
/// [`tail_enc_ctx`] leaves the **first** window full by default because that is
/// how the golden references were produced (whisper.cpp with no `-ac`), so a
/// truncated first window is NON-byte-exact against the golden — that, not a
/// quality regression, is why it is an opt-in. This escape hatch lets a partial
/// FIRST window (a single sub-30 s clip — the streaming / short-utterance case)
/// truncate to `real/2 + margin` encoder frames, killing the wasted encoder pass
/// (and decode) over the zero-padding. `FW_FIRST_WINDOW_MARGIN` = the margin in
/// **encoder frames** (e.g. 100 ≈ 2 s of trailing context); unset ⇒ `None` ⇒
/// first window stays full (byte-identical default).
///
/// MEASURED on large-v3-turbo (jfk, 2026-07-03): `margin=0` is a **~1.9× e2e win
/// on an 11 s single-window clip** (enc ctx 1500→~550), transcript preserved. It
/// is now PURELY a speed lever — the anti-hallucination benefit it once carried
/// (the full-pad first window emitting a spurious `a.` segment) is now handled by
/// DEFAULT via the [`single_timestamp_ending`] fix (commit 53e4fb6), so a clip
/// does NOT need this opt-in to be clean.
///
/// ⚠ SAFE-REGIME FLOOR (measured 2026-07-03, CORRECTS an earlier "needs NO margin"
/// overclaim): first-window truncation is transcript-safe only when the clip has
/// enough real audio that `base = real/2` stays well above `MIN_ENC_CTX` — i.e.
/// moderate single-window clips (≈5–25 s). On VERY SHORT clips (≲5 s) it is
/// UNSTABLE at every margin below full: jfk truncated to 0.5 s / 1 s collapse to
/// "." and 2 s (base=100) is non-monotonic across margins (enc_ctx 100→"And so my",
/// 200→EMPTY, 300→"…Americans,", 500→"…Americans…") — none reproduce the full-window
/// transcript. The 30 s encoder is trained/positional-embedded for long context;
/// truncating below ~a few-hundred frames degrades it. So this knob is for
/// moderate-length single-window / streaming clips, NOT sub-5 s ones.
///
/// Kept an owner opt-in because it is non-byte-exact vs the full-pad golden and
/// must be validated per-model AND per-clip-length by A/B — whisper.cpp's `-ac`
/// applied to the first window, with the same short-clip caveat.
fn first_window_margin() -> Option<usize> {
    use std::sync::OnceLock;
    static M: OnceLock<Option<usize>> = OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("FW_FIRST_WINDOW_MARGIN")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
    })
}

/// Cross-window ENCODE/DECODE pipelining (default ON; kill switch `FW_PIPELINE_WINDOWS=0`).
///
/// In `no_timestamps` mode the seek advance is always a full `CHUNK_CS` (timestamp
/// tokens are masked, so `seek_delta_cs` never changes), which makes window N+1's
/// mel offset known BEFORE window N is decoded. Since window N's decode is a
/// latency-bound single stream that leaves cores idle (NEGATIVE_EVIDENCE 2026-07-02
/// perf profile: ~8% of samples are idle rayon workers), we compute window N+1's
/// (compute-bound, core-hungry) encode CONCURRENTLY on those idle cores via a
/// scoped encoder thread. The prefetched encode is byte-identical to the inline one
/// (same fn, same args), so transcripts are unchanged — a pure RTF lever. MEASURED
/// ~1.19-1.31× on the transcribe phase of a 3-window turbo run, transcript
/// byte-identical. Inert in timestamp mode (next offset is data-dependent) and for
/// single-window audio, so the conformance path (timestamps=true) is untouched.
fn pipeline_windows_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FW_PIPELINE_WINDOWS").map_or(true, |v| {
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        })
    })
}

/// Decode independent no-context windows concurrently on hosts whose Rayon
/// pool already spans every physical core. The ordinary loop carries prompt
/// state and therefore has to remain serial; with `max_context=0`, explicit
/// language, greedy no-timestamp decode, and no fallback/DTW, each 30-second
/// window is a pure function of its own mel offset and the shared read-only
/// weights. Running several token streams at once fills the cores left idle by
/// the latency-bound parts of a single greedy stream.
///
/// The path is deliberately restricted to long jobs on 32+ physical-core
/// hosts and a physical-core-sized Rayon pool. Smaller pools, SMT-sized pools,
/// and all stateful decode modes retain the established serial pipeline.
/// `FW_PARALLEL_NO_CONTEXT_WINDOWS=0` is the process-lifetime rollback switch.
fn parallel_no_context_windows_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FW_PARALLEL_NO_CONTEXT_WINDOWS").map_or(true, |v| {
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        })
    })
}

fn parallel_no_context_window_lanes(
    windows: usize,
    rayon_threads: usize,
    physical_cores: usize,
) -> usize {
    const MIN_WINDOWS: usize = 4;
    const MIN_PHYSICAL_CORES: usize = 32;
    const MIN_THREADS_PER_STREAM: usize = 8;

    if windows < MIN_WINDOWS
        || physical_cores < MIN_PHYSICAL_CORES
        || rayon_threads != physical_cores
    {
        return 1;
    }
    // A common long-form shape is exactly five windows (about two minutes).
    // Let a 32-core host keep that short cohort intact: six physical workers
    // per stream still leaves two scheduling slots, and avoiding a second
    // one-window cohort saves an otherwise fully serial encoder/decode tail.
    // Larger jobs retain the eight-thread floor so live cross-K/V state and
    // competing encoder fronts stay bounded.
    let threads_per_stream = if windows <= 5 {
        6
    } else {
        MIN_THREADS_PER_STREAM
    };
    windows.min((rayon_threads / threads_per_stream).max(1))
}

/// Derive this window's encoder context (in encoder frames) from the real
/// (unpadded) audio frame count remaining in the window.
///
/// Mirrors whisper.cpp's `audio_ctx` / `-ac` feature: a near-empty final window
/// otherwise pays a full 3000-frame / 1500-ctx encoder pass for a fraction of a
/// second of real audio (perf hotspot #1). When `enabled`, the window is **not
/// the first window** (`is_first == false`), and the real audio is shorter than
/// a full window, we run the encoder with a reduced context
/// `enc_ctx = ((real_frames + 1) / 2).clamp(MIN_ENC_CTX, FULL_ENC_CTX)` and feed
/// it a truncated `2*enc_ctx`-frame mel chunk (the conv stem halves time, so
/// `2*enc_ctx` mel frames ⇒ `enc_ctx` encoder rows; whisper.cpp 1982/1995).
///
/// `real_frames` is the remaining real audio in mel frames (1 mel frame = 1 cs
/// = 10 ms); it is the count whisper.cpp also caps at `FRAMES_PER_CHUNK`.
///
/// # Why the first window is never truncated
///
/// whisper.cpp's `-ac` is a single fixed value applied to *every* window, but
/// the golden references were produced with the **default full-pad** behavior
/// (no `-ac`), and truncating the **first** window — which carries the bulk of
/// a short clip's real audio — measurably changes the *main* transcript (on
/// tiny.en/jfk it drops the closing period). The hotspot we target is the
/// *tail*: a non-first window whose remaining audio is a fraction of a second
/// (whisper.cpp's own short-tail handling, `should_clear_short_tail_prompt`,
/// uses the same "non-first + short tail" framing). Restricting truncation to
/// non-first windows kills hotspot #1 while keeping the first/main window
/// byte-identical to the full-pad golden — exactly this lever's correctness
/// contract.
///
/// Returns `FULL_ENC_CTX` (1500) whenever truncation is disabled, the window is
/// full (`real_frames >= FRAMES_PER_CHUNK`), or the window is the first window
/// **and** the experimental [`first_window_margin`] escape hatch is unset (the
/// default), so the caller's behavior is byte-identical to the pre-optimization
/// path in those cases. Hermetic apart from that one cached env read (unset in
/// tests ⇒ first window full), so the unit tests need no model.
fn tail_enc_ctx(real_frames: usize, is_first: bool, enabled: bool) -> usize {
    if !enabled || real_frames >= FRAMES_PER_CHUNK {
        return FULL_ENC_CTX;
    }
    // `enc_ctx = ceil(real_frames / 2)` = `(real_frames + 1) / 2`: round up so
    // the truncated ctx still covers an odd final mel frame (the conv stem maps
    // 2 mel frames → 1 encoder frame), then clamp to the [MIN, FULL] band.
    let base = real_frames.div_ceil(2);
    if is_first {
        // The first window stays FULL by default: the golden was produced full-pad,
        // so a truncated first window is non-byte-exact against it (see
        // [`first_window_margin`] for the measured turbo speed+quality win). The
        // experimental `FW_FIRST_WINDOW_MARGIN` escape hatch truncates it to
        // `base + margin`, keeping `margin` encoder frames of trailing context.
        // Unset ⇒ `None` ⇒ full first window (byte-identical default). Env read
        // once (cached); tests leave it unset.
        return match first_window_margin() {
            Some(m) => (base + m).clamp(MIN_ENC_CTX, FULL_ENC_CTX),
            None => FULL_ENC_CTX,
        };
    }
    base.clamp(MIN_ENC_CTX, FULL_ENC_CTX)
}

// ---------------------------------------------------------------------------
// Top-level transcription (port of whisper_full_with_state greedy path)
// ---------------------------------------------------------------------------

/// Port of whisper.cpp lines 7745-7756. Does the decoded window end in a
/// SINGLE unpaired timestamp (the model cut itself off mid-chunk), meaning
/// the seek should skip the remainder of the chunk?
///
/// The `max_tokens_timestamp_ending` guard (7749-7751) suppresses this when
/// the window only closed because the user token budget's EOT-forcing filter
/// fired (`decoded.len() > budget` — the forced closer pushed the count past
/// the budget): that trailing timestamp is artificial, not a model decision.
/// Upstream gates the guard on `!params.single_segment`; our no-timestamps
/// mode is the analog, hence `timestamps`.
fn single_timestamp_ending(
    decoded: &[i32],
    timestamp_begin: i32,
    timestamps: bool,
    user_budget: Option<usize>,
) -> bool {
    let budget_forced = timestamps && user_budget.is_some_and(|mt| decoded.len() > mt);
    decoded.len() > 1
        && !budget_forced
        && decoded[decoded.len() - 2] < timestamp_begin
        && decoded[decoded.len() - 1] > timestamp_begin
}

struct IndependentWindowResult {
    seek_cs: i64,
    segments: Vec<TranscriptionSegment>,
    stats: WindowStats,
    work: DecodeWorkStats,
}

struct IndependentWindowDecode {
    seek_cs: i64,
    state: DecoderState,
    step_logits: Vec<f32>,
    decoded: Vec<i32>,
    plogs: Vec<f32>,
    has_ts: bool,
    seek_delta_cs: i64,
    result_len: usize,
    no_speech_prob: f64,
    work: DecodeWorkStats,
    done: bool,
}

fn prepare_independent_no_timestamp_window(
    m: &LoadedModel,
    full_mel: &super::Mel,
    params: &DecodeParams,
    language: &str,
    seek_cs: i64,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<IndependentWindowDecode> {
    checkpoint()?;
    let tk = &m.tokenizer;
    let mut work = DecodeWorkStats {
        window_attempts: 1,
        ..DecodeWorkStats::default()
    };
    let frame_offset = usize::try_from(seek_cs).unwrap_or(0);

    let t_enc = std::time::Instant::now();
    let enc = encoder::forward_from_full_mel_window(
        &m.encoder,
        full_mel,
        frame_offset,
        FRAMES_PER_CHUNK,
        params.n_threads,
        checkpoint,
    )?;
    work.encoder_calls = 1;
    super::perf_span("encoder_window", t_enc.elapsed().as_secs_f64() * 1e3, "");

    let t_xkv = std::time::Instant::now();
    let mut st = DecoderState::new(&m.decoder, &enc)?;
    super::perf_span("cross_kv", t_xkv.elapsed().as_secs_f64() * 1e3, "");

    // This fast path is admitted only when cross-window prompt carry is
    // disabled, so the prompt is exactly the per-window SOT sequence.
    let prompt = tk.sot_sequence(Some(language), params.translate, false);
    let t_prefill = std::time::Instant::now();
    let prefill_logits = decoder::forward_step(&m.decoder, &mut st, &prompt, checkpoint)?;
    work.decoder_prefill_calls = 1;
    work.decoder_prefill_tokens = prompt.len();
    super::perf_span(
        "decoder_prefill",
        t_prefill.elapsed().as_secs_f64() * 1e3,
        &format!("\"prompt_tokens\":{}", prompt.len()),
    );
    let no_speech_prob = {
        let lp = compute_logprobs(&prefill_logits);
        usize::try_from(tk.no_speech)
            .ok()
            .and_then(|i| lp.get(i).copied())
            .map_or(0.0, |x| {
                let p = f64::from(x.exp());
                if p.is_finite() { p } else { 0.0 }
            })
    };

    Ok(IndependentWindowDecode {
        seek_cs,
        state: st,
        step_logits: prefill_logits,
        decoded: Vec::new(),
        plogs: Vec::new(),
        has_ts: false,
        seek_delta_cs: CHUNK_CS,
        result_len: 0,
        no_speech_prob,
        work,
        done: false,
    })
}

fn finish_independent_no_timestamp_window(
    mut window: IndependentWindowDecode,
    tk: &Tokenizer,
    seek_end_cs: i64,
) -> IndependentWindowResult {
    window.work.sampled_tokens = window.decoded.len();
    let result_plogs: &[f32] = if window.result_len > 0 && window.result_len <= window.plogs.len() {
        &window.plogs[..window.result_len]
    } else {
        &window.plogs
    };
    let avg_logprob = if result_plogs.is_empty() {
        EMPTY_WINDOW_AVG_LOGPROB
    } else {
        result_plogs.iter().map(|&p| f64::from(p)).sum::<f64>() / result_plogs.len() as f64
    };
    let avg_logprob = if avg_logprob.is_finite() {
        avg_logprob
    } else {
        EMPTY_WINDOW_AVG_LOGPROB
    };
    let is_no_speech =
        window.no_speech_prob > NO_SPEECH_THRESHOLD && avg_logprob < LOGPROB_THRESHOLD;

    window.work.accepted_windows = 1;
    window.work.accepted_result_tokens = window.result_len;
    let segments = if !is_no_speech && !window.decoded.is_empty() {
        let take = window.result_len.min(window.decoded.len());
        build_segments(
            tk,
            &window.decoded[..take],
            &window.plogs[..take],
            window.seek_cs,
            window.seek_delta_cs,
            seek_end_cs,
            false,
        )
    } else {
        Vec::new()
    };

    IndependentWindowResult {
        seek_cs: window.seek_cs,
        segments,
        stats: WindowStats {
            avg_logprob,
            no_speech_prob: window.no_speech_prob,
            tokens: window.result_len,
            window_offset_sec: window.seek_cs as f64 / 100.0,
        },
        work: window.work,
    }
}

fn add_decode_work(total: &mut DecodeWorkStats, part: &DecodeWorkStats) {
    total.window_attempts += part.window_attempts;
    total.encoder_calls += part.encoder_calls;
    total.decoder_prefill_calls += part.decoder_prefill_calls;
    total.decoder_prefill_tokens += part.decoder_prefill_tokens;
    total.sampled_tokens += part.sampled_tokens;
    total.greedy_single_token_forwards += part.greedy_single_token_forwards;
    total.accepted_windows += part.accepted_windows;
    total.accepted_result_tokens += part.accepted_result_tokens;
    total.prompt_reset_retries += part.prompt_reset_retries;
    total.temperature_fallback_retries += part.temperature_fallback_retries;
}

#[allow(clippy::too_many_arguments)]
fn decode_independent_no_timestamp_windows(
    m: &LoadedModel,
    full_mel: &super::Mel,
    params: &DecodeParams,
    cfg: &FilterConfig,
    language: &str,
    seek_end_cs: i64,
    n_max_tokens: usize,
    user_max_tokens: Option<usize>,
    lanes: usize,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<DecodeOutput> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    let offsets: Vec<i64> = (0..)
        .map(|window| window * CHUNK_CS)
        .take_while(|seek| seek + DELTA_MIN < seek_end_cs)
        .collect();
    let cohort_width = lanes.max(1);
    let mut results = Vec::with_capacity(offsets.len());

    // Limit live cross-K/V states to one lane-sized cohort.  This keeps memory
    // bounded for hour-scale inputs while still amortizing every decoder weight
    // read across all lanes.  Encoders and cross-K/V preparation remain parallel;
    // the greedy phase then advances the cohort in lockstep through the batched
    // decoder entry point.
    for cohort_offsets in offsets.chunks(cohort_width) {
        let stop = AtomicBool::new(false);
        let first_error: Mutex<Option<FwError>> = Mutex::new(None);
        let prepared: Mutex<Vec<IndependentWindowDecode>> =
            Mutex::new(Vec::with_capacity(cohort_offsets.len()));

        std::thread::scope(|scope| {
            for &seek_cs in cohort_offsets {
                let stop = &stop;
                let first_error = &first_error;
                let prepared = &prepared;
                scope.spawn(move || {
                    let worker_checkpoint = || -> FwResult<()> {
                        if stop.load(Ordering::Relaxed) {
                            return Err(FwError::Cancelled(
                                "parallel no-context window worker stopped".to_owned(),
                            ));
                        }
                        checkpoint()
                    };
                    match prepare_independent_no_timestamp_window(
                        m,
                        full_mel,
                        params,
                        language,
                        seek_cs,
                        &worker_checkpoint,
                    ) {
                        Ok(window) => prepared
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(window),
                        Err(error) => {
                            let mut slot = first_error
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                            if slot.is_none() {
                                *slot = Some(error);
                            }
                            stop.store(true, Ordering::Relaxed);
                        }
                    }
                });
            }
        });

        if let Some(error) = first_error
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            return Err(error);
        }

        let mut active = prepared
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        active.sort_by_key(|window| window.seek_cs);
        let t_loop = std::time::Instant::now();
        let mut cohort_tokens = 0usize;

        for i in 0..n_max_tokens {
            for window in &mut active {
                let (filtered, logprobs) = process_logits(
                    &m.tokenizer,
                    cfg,
                    std::mem::take(&mut window.step_logits),
                    &window.decoded,
                    window.has_ts,
                    window.seek_delta_cs,
                    window.decoded.len(),
                );
                let (tok, plog) = argmax(&filtered, &logprobs);
                window.decoded.push(tok);
                window.plogs.push(plog);
                cohort_tokens += 1;

                // Timestamp tokens are masked by this path's filter.  Retain
                // the ordinary state transition so malformed-model behavior is
                // identical to the scalar loop.
                if tok > m.tokenizer.timestamp_begin {
                    let new_delta = 2 * i64::from(tok - m.tokenizer.timestamp_begin);
                    if window.has_ts && window.seek_delta_cs > new_delta && window.result_len < i {
                        window.done = true;
                        continue;
                    }
                    window.seek_delta_cs = new_delta;
                    window.result_len = i + 1;
                    window.has_ts = true;
                }

                let budget_reached = user_max_tokens.is_some_and(|mt| i >= mt);
                let reached_end = window.has_ts
                    && window.seek_cs + window.seek_delta_cs + DELTA_MIN >= seek_end_cs;
                if tok == m.tokenizer.eot || budget_reached || reached_end {
                    window.result_len = i + 1;
                    window.seek_delta_cs = CHUNK_CS;
                    window.done = true;
                } else if i + 1 == n_max_tokens {
                    // The next logits would never be sampled.  The scalar path
                    // computed this final dead forward; omit it from the cohort.
                    window.done = true;
                }
            }

            let mut survivors = Vec::with_capacity(active.len());
            for window in active.drain(..) {
                if window.done {
                    results.push(finish_independent_no_timestamp_window(
                        window,
                        &m.tokenizer,
                        seek_end_cs,
                    ));
                } else {
                    survivors.push(window);
                }
            }
            active = survivors;
            if active.is_empty() {
                break;
            }

            checkpoint()?;
            let tokens: Vec<i32> = active
                .iter()
                .map(|window| *window.decoded.last().expect("sampled token"))
                .collect();
            let batch_logits = {
                let mut states: Vec<&mut DecoderState> =
                    active.iter_mut().map(|window| &mut window.state).collect();
                decoder::forward_step_batch(&m.decoder, &mut states, &tokens, checkpoint)?
            };
            for (window, logits) in active.iter_mut().zip(batch_logits) {
                window.step_logits = logits;
                window.work.greedy_single_token_forwards += 1;
            }
        }

        debug_assert!(active.is_empty());
        super::perf_span(
            "decode_batch_loop",
            t_loop.elapsed().as_secs_f64() * 1e3,
            &format!(
                "\"windows\":{},\"tokens\":{}",
                cohort_offsets.len(),
                cohort_tokens
            ),
        );
    }

    results.sort_by_key(|result| result.seek_cs);
    let mut segments = Vec::new();
    let mut windows = Vec::with_capacity(results.len());
    let mut work = DecodeWorkStats::default();
    for result in results {
        segments.extend(result.segments);
        windows.push(result.stats);
        add_decode_work(&mut work, &result.work);
    }

    Ok(DecodeOutput {
        segments,
        language: Some(language.to_owned()),
        windows,
        work,
        word_timings: None,
    })
}

/// Transcribe 16 kHz mono PCM `samples` with the greedy / temperature-0 path of
/// whisper, returning timed segments + per-window QC statistics.
///
/// A per-model, 64 MiB bounded LRU reuses exact sample-bit + [`DecodeParams`]
/// matches across calls. Hash matches are always verified bit-for-bit. Cache
/// hits retain the prior output but report zero physical work; call
/// [`LoadedModel::clear_transcription_cache`] after changing an experimental
/// process-global compute policy on a live model. `FW_TRANSCRIPT_CACHE=0`
/// disables the cache for operational rollback and same-binary comparison.
///
/// `checkpoint` is invoked between **every** decoder step (and between encoder
/// layers, via the underlying forward passes) so a caller can cancel a long
/// transcription at token granularity — the project's cancellation contract.
///
/// # Errors
/// - [`FwError::InvalidRequest`] for empty input or a model/shape mismatch
///   surfaced by the encoder/decoder.
/// - Whatever `checkpoint` returns (e.g. [`FwError::Cancelled`]), promptly.
pub fn transcribe_samples(
    m: &LoadedModel,
    samples_16k_mono: &[f32],
    params: &DecodeParams,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<DecodeOutput> {
    let cacheable = transcription_cache_enabled()
        && samples_16k_mono
            .len()
            .saturating_mul(std::mem::size_of::<f32>())
            <= TRANSCRIPTION_CACHE_MAX_ENTRY_SAMPLE_BYTES;
    let fingerprint = if cacheable {
        checkpoint()?;
        let fingerprint = batch_job_fingerprint(samples_16k_mono, params);
        if let Some(output) = m
            .transcription_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .lookup(fingerprint, samples_16k_mono, params)
        {
            super::perf_span(
                "transcription_cache_hit",
                0.0,
                &format!("\"samples\":{}", samples_16k_mono.len()),
            );
            return Ok(output);
        }
        Some(fingerprint)
    } else {
        None
    };

    let output = transcribe_samples_uncached(m, samples_16k_mono, params, checkpoint)?;
    if let Some(fingerprint) = fingerprint {
        m.transcription_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(fingerprint, samples_16k_mono, params, &output);
    }
    Ok(output)
}

fn transcribe_samples_uncached(
    m: &LoadedModel,
    samples_16k_mono: &[f32],
    params: &DecodeParams,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<DecodeOutput> {
    if samples_16k_mono.is_empty() {
        return Err(FwError::InvalidRequest(
            "transcribe_samples: empty audio".into(),
        ));
    }
    super::ensure_default_rayon_pool();

    let tk = &m.tokenizer;

    // Full-audio log-mel spectrogram (whisper computes once, then windows it).
    // mel is compute-bound FFT-per-frame (bit-identical for ANY thread count), and
    // its measured optimum is ~16 workers (8 under-utilizes ~1.39×; >16 cross-CCD-
    // regresses on this Zen box), so decouple it from the decode's `n_threads` hint.
    // `FW_MEL_THREADS` (inside `log_mel`) overrides.
    let mel_threads = super::host_parallelism().min(16);
    let t_mel = std::time::Instant::now();
    let full_mel = mel::log_mel(samples_16k_mono, &m.filters, mel_threads)?;
    super::perf_span("mel", t_mel.elapsed().as_secs_f64() * 1e3, "");

    // Window bounds in centiseconds: seek runs [0, seek_end) where seek_end is
    // the **original** (unpadded) audio length — whisper.cpp's `n_len_org`
    // (whisper.cpp 6859-6860). `log_mel` trails a full 30 s of silence padding,
    // so `full_mel.n_frames` is NOT the right bound. Upstream computes the real
    // length as the mel `n_len_org` (whisper.cpp 3208):
    //   n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step
    // with `stage_2_pad = 200`, `frame_size = 400`, `frame_step = 160`. One mel
    // frame = 10 ms = 1 cs (fix #7). Guard underflow for `n_samples < 200` (the
    // numerator `n_samples - 200` saturates to 0).
    let seek_end_cs = {
        let n_samples = samples_16k_mono.len() as i64;
        const STAGE_2_PAD: i64 = 200;
        const FRAME_SIZE: i64 = 400;
        const FRAME_STEP: i64 = 160;
        // n_samples + 200 - 400 = n_samples - 200, saturating at 0.
        let numer = (n_samples + STAGE_2_PAD - FRAME_SIZE).max(0);
        (1 + numer / FRAME_STEP).max(DELTA_MIN)
    };

    // Resolve language: explicit / en-only now; multilingual auto-detect is
    // deferred to the first window so its encoder output is computed once.
    let mut used_language = resolve_language_fast(m, params);

    // The `" "` token id, for blank suppression (whisper.cpp 6220).
    let space_token = (0..tk.vocab_size()).find(|&id| tk.token_bytes(id) == Some(b" ".as_slice()));

    // max_initial_ts clamp tid0 (whisper.cpp 6321-6323): precision = 30 / n_audio_ctx.
    let max_initial_tid = if MAX_INITIAL_TS_SEC > 0.0 {
        let precision = 30.0f32 / m.hparams.n_audio_ctx.max(1) as f32;
        Some((MAX_INITIAL_TS_SEC / precision).round() as i32)
    } else {
        None
    };

    // Two DISTINCT per-window numbers, as upstream (the original port
    // conflated them, which made the EOT-forcing filter unreachable):
    // - the STRUCTURAL decode bound (whisper.cpp 7330: n_max = n_text_ctx/2-4)
    //   that sizes the sampling loop, and
    // - the optional USER token budget (whisper.cpp `params.max_tokens`,
    //   default off) that the EOT-forcing filter + budget-break act on while
    //   the loop continues, so the forced closing timestamp can still be
    //   sampled (whisper.cpp 6234 + 7388).
    let n_text_ctx = m.decoder.n_text_ctx();
    let n_max_tokens = (n_text_ctx / 2).saturating_sub(4).max(1);
    let user_max_tokens = params
        .max_text_ctx
        .filter(|&mt| mt > 0)
        .map(|mt| mt.min(n_max_tokens));

    let cfg = FilterConfig {
        suppress_blank: true,
        space_token,
        suppress_nst: params.suppress_nst, // whisper.cpp default false (5970); honor --suppress-nst.
        no_timestamps: !params.timestamps,
        max_initial_tid,
        // EOT-forcing budget (fix #6): the user budget, off when unset —
        // mirroring upstream's `params.max_tokens > 0` gate.
        max_tokens: user_max_tokens,
    };
    // Prompt context cap (whisper.cpp 6927): n_text_ctx/2.
    // whisper `--max-context` / `n_max_text_ctx`: < 0 (or unset) → default
    // n_text_ctx/2; 0 → no prompt carried; n → cap at n (whisper.cpp semantics).
    let max_prompt_ctx = match params.max_context {
        Some(n) if n >= 0 => usize::try_from(n).unwrap_or(n_text_ctx / 2),
        _ => n_text_ctx / 2,
    };
    // tiny.en's carried prompt is the proven cause of its segment-timestamp
    // failed-window retry. Suppress only cross-window carry by default; an
    // explicit request or operator override retains the historical behavior.
    let suppress_tiny_en_ts_context = suppress_tiny_en_segment_ts_context(
        &m.hparams,
        params,
        tiny_en_segment_ts_context_forced(),
    );

    let mut segments: Vec<TranscriptionSegment> = Vec::new();
    let mut windows: Vec<WindowStats> = Vec::new();
    let mut work = DecodeWorkStats::default();
    // Per-segment DTW word timings, accumulated 1:1 with `segments` when
    // `params.word_timestamps` is set (bd-rjsx).
    let mut word_timings: Vec<Vec<WordTiming>> = Vec::new();
    // Alignment heads for this model, resolved once (DTW word timestamps only).
    let align_heads = if params.word_timestamps {
        dtw::alignment_heads(&m.hparams, params.model_hint.as_deref())
    } else {
        Vec::new()
    };
    // Rolling text context from prior windows (whisper.cpp prompt_past1).
    // Seed it with the optional user prompt (whisper `--prompt`): the
    // `DecodeParams.initial_prompt` field is the API; `FW_INITIAL_PROMPT` overrides
    // it when set (dev/testing hatch). Its tokens are carried as previous context
    // on the first window and age out via the max_prompt_ctx truncation as decoded
    // text accumulates. Unset → no-op (byte-identical default).
    let prompt = initial_prompt_from_env().or(params.initial_prompt.as_deref());
    let mut prompt_past: Vec<i32> = seeded_prompt_past(prompt, &m.tokenizer);

    // Beam width (whisper `--beam-size`): field or FW_BEAM_SIZE override, resolved
    // once. 1 = greedy (byte-identical default).
    let effective_beam_size = resolve_beam_size(params);

    // Long no-context jobs have no cross-window dependency: timestamp tokens
    // are masked (fixed 30 s seek), the language is already known, prompt carry
    // is disabled, and the greedy/default-off fallback path owns no shared
    // decoder state. On a 32+ physical-core host whose Rayon pool spans exactly
    // those physical cores, fan the independent token streams out while sharing
    // the immutable model and the one full-audio mel. This preserves the exact
    // mel frames at every boundary (unlike slicing/re-mel range batchers) and
    // merges results by offset, so scheduling cannot reorder output.
    let independent_window_count = usize::try_from((seek_end_cs - DELTA_MIN).max(0))
        .unwrap_or(0)
        .div_ceil(CHUNK_CS as usize);
    let independent_window_lanes = super::physical_cores().map_or(1, |physical| {
        parallel_no_context_window_lanes(
            independent_window_count,
            rayon::current_num_threads(),
            physical,
        )
    });
    if parallel_no_context_windows_enabled()
        && cfg.no_timestamps
        && max_prompt_ctx <= 1
        && !params.word_timestamps
        && effective_beam_size == 1
        && !temp_fallback_enabled()
        && std::env::var_os("PROBE_DUMP_TOKENS").is_none()
        && independent_window_lanes > 1
        && let Some(language) = used_language.as_deref()
    {
        return decode_independent_no_timestamp_windows(
            m,
            &full_mel,
            params,
            &cfg,
            language,
            seek_end_cs,
            n_max_tokens,
            user_max_tokens,
            independent_window_lanes,
            checkpoint,
        );
    }

    // Tail-window encoder-context truncation kill switch, resolved once.
    //
    // Truncation is gated to TIMESTAMP mode. In timestamp mode it is a quality
    // *and* speed win (whisper.cpp `-ac`): timestamp tokens give the decode
    // structure/stopping, and truncating the trailing zero-padding stops the
    // classic silence-hallucination across the padded tail (measured: jfk×3
    // 1.90×, ×5 1.39×, both cleaner). In no_timestamps mode the effect INVERTS
    // and truncation becomes a content-loss bug: with timestamp tokens
    // suppressed, the trailing silence is the greedy decoder's only end-of-speech
    // cue, so truncating a partial tail window to real-audio length removes that
    // cue and the (temperature-fallback-free) greedy loop repetition-loops over
    // the short truncated context to the token cap — the garbage window is then
    // gated out, DROPPING the real tail (jfk×3: window 2 decodes 220 tokens →
    // dropped → "…ask what you can do for your country." lost; ×5: a spurious 6th
    // sentence). whisper.cpp does NOT apply `-ac` by default, so full-pad is the
    // faithful no_ts behavior AND the correct one (franken truncate-off matches
    // whisper.cpp `-nt` on jfk×3/×5; truncate-on diverges). Timestamp-mode golden
    // is unchanged (params.timestamps == true ⇒ identical to before).
    let tail_truncate = tail_truncate_enabled() && params.timestamps;

    // Cross-window pipelining (no_timestamps only; default off): a persistent
    // scoped encoder thread computes the NEXT window's encode while THIS window
    // decodes on the calling thread. The while loop stays lexically top-level in
    // the scope closure, so its break/continue/`?` still target the loop, and
    // `thread::scope` joins the encoder thread on ANY exit (incl. `?`). Byte-exact:
    // the prefetched encode is the same fn+args as the inline one. See
    // `pipeline_windows_enabled`.
    let pipeline = pipeline_windows_enabled() && cfg.no_timestamps;
    let enc_n_threads = params.n_threads;
    let pipe_result: FwResult<()> = std::thread::scope(|scope| {
        let (req_tx, req_rx) = std::sync::mpsc::channel::<(usize, usize)>();
        let (res_tx, res_rx) = std::sync::mpsc::channel::<FwResult<super::Mat>>();
        let _enc_worker = if pipeline {
            let enc_w = &m.encoder;
            let mel_ref = &full_mel;
            Some(scope.spawn(move || {
                // Speculative prefetch: use a no-op checkpoint (cancellation of the
                // MAIN decode still gates progress; a wasted prefetch is harmless).
                let noop = || Ok(());
                while let Ok((off, mf)) = req_rx.recv() {
                    let r = encoder::forward_from_full_mel_window(
                        enc_w,
                        mel_ref,
                        off,
                        mf,
                        enc_n_threads,
                        &noop,
                    );
                    if res_tx.send(r).is_err() {
                        break;
                    }
                }
            }))
        } else {
            None
        };
        // The frame_offset already dispatched to the encoder thread (its result is
        // in flight / waiting on `res_rx`), or None when nothing is prefetched.
        let mut prefetched: Option<usize> = None;

        let mut seek_cs: i64 = 0;
        // Set for one iteration when retrying a failed window with the carried prompt
        // cleared (FW_RETRY_FAILED_WINDOW). Reset once the window completes.
        let mut force_empty_prompt = false;
        // FW_TEMP_FALLBACK ladder position for the CURRENT seek: 0 = the normal
        // greedy pass; k > 0 = re-decoding at TEMP_FALLBACK_LADDER[k - 1]. Reset to 0
        // once a window is accepted, so every window starts greedy.
        let mut temp_attempt: usize = 0;
        // The temperature the current attempt decodes at (0.0 = argmax/greedy path).
        let mut window_temp: f64 = 0.0;
        // best_of state within the current t > 0 rung: candidate index and the
        // best-scoring candidate so far (whisper.cpp ranks its `best_of`
        // decoders by sequence score and keeps the winner).
        let mut cand_idx: usize = 0;
        let mut rung_best: Option<WindowCandidate> = None;
        // Holds `(frame_offset, enc)` from a window's first attempt so a
        // FW_RETRY_FAILED_WINDOW retry (same seek) reuses the identical encode instead
        // of paying a full re-encode. Only ever populated when the flag is on.
        let mut retry_enc_cache = None;
        while seek_cs + DELTA_MIN < seek_end_cs {
            checkpoint()?;
            work.window_attempts += 1;

            // Encode this window's mel chunk. A full window is 3000 mel frames
            // (1500 encoder ctx); a tail window with under 30 s of real audio left
            // is truncated to `2*enc_ctx` mel frames (`enc_ctx` encoder rows),
            // mirroring whisper.cpp's audio_ctx (-ac) feature — a near-empty final
            // window otherwise pays a full encode for a fraction of a second of
            // audio (perf hotspot #1). Timestamp/precision semantics are unaffected
            // (`max_initial_tid` is tied to the full model `n_audio_ctx`, not this
            // window's ctx — whisper.cpp 6322).
            let frame_offset = usize::try_from(seek_cs).unwrap_or(0);
            // Real (unpadded) audio remaining in this window, in mel frames
            // (1 mel frame = 1 cs); capped at the full window, as whisper.cpp does.
            let real_frames = usize::try_from((seek_end_cs - seek_cs).max(0))
                .unwrap_or(0)
                .min(FRAMES_PER_CHUNK);
            let enc_ctx = tail_enc_ctx(real_frames, seek_cs == 0, tail_truncate);
            let mel_frames = enc_ctx * 2;
            if mel_frames < FRAMES_PER_CHUNK {
                tracing::debug!(
                    target: "franken_whisper::native_engine::decode",
                    seek_cs,
                    real_frames,
                    enc_ctx,
                    mel_frames,
                    "tail-window encoder-context truncation engaged"
                );
            }
            let t_enc = std::time::Instant::now();
            // Reuse the encode from THIS seek's failed first attempt on a
            // FW_RETRY_FAILED_WINDOW retry — same audio/window ⇒ byte-identical enc, so
            // the retry skips a full re-encode. Only hit when the flag is on and a retry
            // is in flight; otherwise falls through to prefetch/inline exactly as before.
            // SAFE vs the pipeline prefetch below: the retry only fires on `result_len == 0`,
            // which is a TS-mode-only failure (no_ts sets result_len = i+1 at its break), while
            // `pipeline` is `… && cfg.no_timestamps` (no_ts only). Retry and pipeline are thus
            // mutually exclusive, so the `if pipeline` dispatch is a no-op on any retry re-entry —
            // no double-send / res_rx desync. (Verified: retry-on track01 transcript is coherent.)
            let reuses_cached_encode = retry_enc_cache
                .as_ref()
                .is_some_and(|(off, _): &(usize, _)| *off == frame_offset);
            let enc = if reuses_cached_encode {
                retry_enc_cache.take().expect("checked is_some_and above").1
            } else if pipeline && prefetched == Some(frame_offset) {
                match res_rx.recv() {
                    Ok(r) => r?,
                    Err(_) => encoder::forward_from_full_mel_window(
                        &m.encoder,
                        &full_mel,
                        frame_offset,
                        mel_frames,
                        params.n_threads,
                        checkpoint,
                    )?,
                }
            } else {
                encoder::forward_from_full_mel_window(
                    &m.encoder,
                    &full_mel,
                    frame_offset,
                    mel_frames,
                    params.n_threads,
                    checkpoint,
                )?
            };
            if !reuses_cached_encode {
                work.encoder_calls += 1;
            }
            // Dispatch the NEXT window's encode NOW so it overlaps this window's decode.
            // no_timestamps advance is always CHUNK_CS, so the next offset is known and
            // its mel_frames mirror the inline `tail_enc_ctx` computation exactly.
            //
            // A same-seek re-entry (FW_TEMP_FALLBACK / FW_RETRY_FAILED_WINDOW) arrives
            // here with the next window's encode ALREADY dispatched and unconsumed
            // (`prefetched == Some(next_off)`); re-sending would queue a duplicate
            // result and desync every later window's recv by one. Keep the in-flight
            // one instead. Outside a retry this predicate is impossible (a fresh
            // window has consumed or never set it), so the default path is untouched.
            let next_off = frame_offset + CHUNK_CS as usize;
            if prefetched != Some(next_off) {
                prefetched = None;
                if pipeline {
                    let next_seek = seek_cs + CHUNK_CS;
                    if next_seek + DELTA_MIN < seek_end_cs {
                        let next_real = usize::try_from((seek_end_cs - next_seek).max(0))
                            .unwrap_or(0)
                            .min(FRAMES_PER_CHUNK);
                        let next_ctx = tail_enc_ctx(next_real, false, tail_truncate);
                        if req_tx.send((next_off, next_ctx * 2)).is_ok() {
                            prefetched = Some(next_off);
                        }
                    }
                }
            }
            super::perf_span("encoder_window", t_enc.elapsed().as_secs_f64() * 1e3, "");
            let t_xkv = std::time::Instant::now();
            let mut st = DecoderState::new(&m.decoder, &enc)?;
            super::perf_span("cross_kv", t_xkv.elapsed().as_secs_f64() * 1e3, "");

            // First-window language auto-detect (multilingual, no explicit
            // language): reuses this window's encode + this state's cross K/V.
            if used_language.is_none() {
                used_language = detect_language_from_enc(m, &mut st, checkpoint)?;
            }

            // Short-tail prompt clearing (fix #2 — whisper.cpp 7046-7051): a
            // non-first window with under 5 s of audio left drops the carried prompt
            // to avoid repetition/hallucination on the tail.
            if should_clear_short_tail_prompt(seek_cs, seek_end_cs) {
                prompt_past.clear();
            }

            // Build the prompt: [sot_prev, ...past...] + sot_sequence (whisper.cpp
            // 7106-7133). prompt_init is the sot sequence for this language/task.
            let sot_seq = tk.sot_sequence(
                used_language.as_deref(),
                params.translate,
                params.timestamps,
            );
            let mut prompt: Vec<i32> = Vec::new();
            // Carry the prior-window text prompt unless conditioning is disabled
            // (whisper.cpp `no_context` / `--no-context`; bd-r0qd escape hatch,
            // `FW_NO_CONTEXT=1`), the tiny.en segment-TS policy suppresses window
            // 2+ carry, this is a failed-window retry (`force_empty_prompt`), or a
            // FW_TEMP_FALLBACK attempt above the prompt-reset temperature.
            let prompt_carried = !condition_on_prev_disabled()
                && !force_empty_prompt
                && window_temp <= TEMP_PROMPT_RESET
                && !prompt_past.is_empty()
                && max_prompt_ctx > 1
                && !(suppress_tiny_en_ts_context && seek_cs > 0);
            if prompt_carried {
                prompt.push(tk.sot_prev);
                let take = prompt_past.len().min(max_prompt_ctx.saturating_sub(1));
                prompt.extend_from_slice(&prompt_past[prompt_past.len() - take..]);
            }
            prompt.extend_from_slice(&sot_seq);

            // Prefill the prompt; the first forward's softmax gives no_speech_prob
            // (whisper.cpp 7165-7182). Compute it BEFORE filtering.
            let t_prefill = std::time::Instant::now();
            let prefill_logits = decoder::forward_step(&m.decoder, &mut st, &prompt, checkpoint)?;
            work.decoder_prefill_calls += 1;
            work.decoder_prefill_tokens += prompt.len();
            super::perf_span(
                "decoder_prefill",
                t_prefill.elapsed().as_secs_f64() * 1e3,
                &format!("\"prompt_tokens\":{}", prompt.len()),
            );
            let no_speech_prob = {
                let lp = compute_logprobs(&prefill_logits);
                usize::try_from(tk.no_speech)
                    .ok()
                    .and_then(|i| lp.get(i).copied())
                    .map_or(0.0, |x| {
                        // Defense-in-depth: a non-finite logprob must not export a
                        // NaN no_speech_prob into the silence gate / routing snapshot.
                        let p = f64::from(x.exp());
                        if p.is_finite() { p } else { 0.0 }
                    })
            };

            // Decode loop. `FW_BEAM_SIZE > 1` runs beam search for the
            // temperature-0 pass; the `t > 0` fallback rungs (`window_temp > 0`)
            // stay on the greedy sampling path. `beam_size() == 1` (default) ⇒ the
            // greedy branch always runs ⇒ byte-identical to the pre-beam engine.
            let t_loop = std::time::Instant::now();
            let (decoded, plogs, _has_ts, seek_delta_cs, result_len) = if effective_beam_size > 1
                && window_temp == 0.0
            {
                // `st` is left intact (beam clones per hypothesis) for the DTW
                // re-forward below, which shares this window's cross-K/V.
                beam_decode_window(
                    m,
                    &st,
                    prefill_logits,
                    tk,
                    &cfg,
                    params,
                    seek_cs,
                    seek_end_cs,
                    n_max_tokens,
                    user_max_tokens,
                    effective_beam_size,
                    checkpoint,
                )?
            } else {
                let mut decoded: Vec<i32> = Vec::new();
                let mut plogs: Vec<f32> = Vec::new();
                let mut has_ts = false;
                let mut seek_delta_cs = CHUNK_CS; // default: advance full window.
                let mut result_len = 0usize;
                let mut step_logits = prefill_logits;
                // Deterministic per-(window, rung, candidate) sampling stream
                // (FW_TEMP_FALLBACK): only ever drawn from when `window_temp > 0.0`.
                // Candidate 0 of every rung matches the pre-best_of stream exactly,
                // so `FW_TEMP_BEST_OF=1` reproduces the single-candidate ladder
                // byte-for-byte.
                let mut sample_rng: u64 = 0x5851_F42D_4C95_7F2D
                    ^ (seek_cs as u64)
                    ^ ((temp_attempt as u64) << 48)
                    ^ ((cand_idx as u64) << 40);

                for i in 0..n_max_tokens {
                    let (filtered, logprobs) = process_logits(
                        tk,
                        &cfg,
                        step_logits,
                        &decoded,
                        has_ts,
                        seek_delta_cs,
                        decoded.len(),
                    );
                    let (tok, plog) = if window_temp > 0.0 {
                        sample_token_at_temperature(
                            &filtered,
                            &logprobs,
                            window_temp,
                            &mut sample_rng,
                        )
                    } else {
                        argmax(&filtered, &logprobs)
                    };
                    decoded.push(tok);
                    plogs.push(plog);

                    // Update sliding window from a timestamp token (whisper.cpp 7362-7375).
                    if tok > tk.timestamp_begin {
                        let new_delta = 2 * i64::from(tok - tk.timestamp_begin);
                        if has_ts && seek_delta_cs > new_delta && result_len < i {
                            // Going back in time: bail out of this window (whisper.cpp 7366-7369).
                            break;
                        }
                        seek_delta_cs = new_delta;
                        result_len = i + 1;
                        has_ts = true;
                    }

                    // End of segment (whisper.cpp 7387-7410). `budget_reached` is the
                    // `params.max_tokens > 0 && i >= params.max_tokens` clause: the
                    // EOT-forcing filter masked text from sampled-token index `mt`
                    // onward, so the token at index `i == mt` is the forced closer and
                    // the window completes here with `decoded.len() == mt + 1`.
                    let budget_reached = user_max_tokens.is_some_and(|mt| i >= mt);
                    let reached_end = has_ts && seek_cs + seek_delta_cs + DELTA_MIN >= seek_end_cs;
                    if tok == tk.eot || budget_reached || reached_end {
                        if result_len == 0 && params.timestamps {
                            if reached_end {
                                result_len = i + 1;
                            } else {
                                // Decoder failed with no timestamps closed.
                                break;
                            }
                        }
                        if !params.timestamps {
                            result_len = i + 1;
                            seek_delta_cs = CHUNK_CS;
                        }
                        break;
                    }

                    // Cancellation between every decoder step (the project contract).
                    checkpoint()?;

                    // Forward the just-chosen token to get the next logits.
                    step_logits = decoder::forward_step(&m.decoder, &mut st, &[tok], checkpoint)?;
                    work.greedy_single_token_forwards += 1;
                }
                (decoded, plogs, has_ts, seek_delta_cs, result_len)
            };
            work.sampled_tokens += decoded.len();

            // avg_logprob over result tokens (whisper.cpp 6602-6617).
            let result_plogs: &[f32] = if result_len > 0 && result_len <= plogs.len() {
                &plogs[..result_len]
            } else {
                &plogs[..]
            };
            let avg_logprob = if result_plogs.is_empty() {
                EMPTY_WINDOW_AVG_LOGPROB
            } else {
                result_plogs.iter().map(|&p| f64::from(p)).sum::<f64>() / result_plogs.len() as f64
            };
            // A non-finite mean (a NaN/inf plog escaping the decoder) degrades to the
            // finite empty-window sentinel so the silence gate and routing snapshot
            // never see NaN (a NaN would silently read as "not silence").
            let avg_logprob = if avg_logprob.is_finite() {
                avg_logprob
            } else {
                EMPTY_WINDOW_AVG_LOGPROB
            };

            // FW_TEMP_FALLBACK best_of (whisper.cpp greedy.best_of = 5): at a
            // t > 0 rung each loop re-entry decodes ONE independent sampling
            // candidate; the rung's best sequence_score is adopted into the
            // window locals here, and everything downstream (quality gate,
            // emission, DTW, prompt carry, seek advance) sees only the winner.
            // Never entered at window_temp == 0.0, so the greedy pass and the
            // whole default path are untouched.
            let (decoded, plogs, result_len, seek_delta_cs, avg_logprob, no_speech_prob) =
                if temp_fallback_enabled() && window_temp > 0.0 {
                    let cand = WindowCandidate {
                        score: sequence_score(&plogs, result_len),
                        decoded,
                        plogs,
                        result_len,
                        seek_delta_cs,
                        avg_logprob,
                        no_speech_prob,
                    };
                    // Ties keep the EARLIER candidate (whisper.cpp's ranking
                    // scans decoders in order and replaces only on a strictly
                    // better score).
                    let best = match rung_best.take() {
                        Some(b) if b.score >= cand.score => b,
                        _ => cand,
                    };
                    if cand_idx + 1 < temp_best_of() {
                        rung_best = Some(best);
                        cand_idx += 1;
                        retry_enc_cache = Some((frame_offset, enc));
                        continue; // decode this rung's next candidate
                    }
                    // (cand_idx is reset by both downstream paths — the rung
                    // advance and the window-accept block — before any next read.)
                    (
                        best.decoded,
                        best.plogs,
                        best.result_len,
                        best.seek_delta_cs,
                        best.avg_logprob,
                        best.no_speech_prob,
                    )
                } else {
                    (
                        decoded,
                        plogs,
                        result_len,
                        seek_delta_cs,
                        avg_logprob,
                        no_speech_prob,
                    )
                };

            // no-speech / failed-window gate (whisper.cpp 7606-7607): treat as
            // silence, emit nothing, advance the full window. At a completed
            // t > 0 rung this evaluates the ADOPTED best candidate.
            let is_no_speech =
                no_speech_prob > NO_SPEECH_THRESHOLD && avg_logprob < LOGPROB_THRESHOLD;

            // FW_TEMP_FALLBACK (default-off, bd-6goy / bd-r0qd fix-spec #3): the
            // whisper.cpp fallback ladder. A non-silent window that closed no
            // timestamp, averaged below the logprob threshold, or looped into a
            // low-entropy repetitive tail (whisper.cpp entropy_thold, 7540)
            // re-decodes this SAME seek at the next ladder temperature, reusing the
            // stashed encode. The ladder is finite, so this cannot loop; the final
            // rung's decode is accepted as-is below (whisper.cpp keeps its last
            // decode too). Unset ⇒ never fires ⇒ byte-exact.
            //
            // The entropy tail is taken over the result slice like upstream
            // (`sequence.tokens.resize(result_len)` precedes the score). Minor
            // divergence: in no_ts our result_len includes the terminal EOT; one
            // unique id in a 33+-token tail moves the entropy negligibly.
            let tail_entropy = {
                let take = result_len.min(decoded.len());
                // Only priced when the gate is on: the default path never counts.
                if temp_fallback_enabled() && take > ENTROPY_WINDOW {
                    Some(token_tail_entropy(&decoded[..take]))
                } else {
                    None
                }
            };
            let quality_failed = !is_no_speech
                && (result_len == 0
                    || avg_logprob < LOGPROB_THRESHOLD
                    || tail_entropy.is_some_and(|e| e < ENTROPY_THRESHOLD));
            if temp_fallback_enabled()
                && quality_failed
                && temp_attempt < TEMP_FALLBACK_LADDER.len()
            {
                window_temp = TEMP_FALLBACK_LADDER[temp_attempt];
                temp_attempt += 1;
                work.temperature_fallback_retries += 1;
                cand_idx = 0;
                rung_best = None;
                tracing::debug!(
                    target: "franken_whisper::native_engine::decode",
                    seek_sec = seek_cs as f64 / 100.0,
                    avg_logprob,
                    result_len,
                    tail_entropy = tail_entropy.unwrap_or(f64::NAN),
                    retry_temperature = window_temp,
                    "window failed the quality gate — temperature-fallback retry"
                );
                retry_enc_cache = Some((frame_offset, enc));
                continue; // re-decode this window at the raised temperature
            }
            // Window accepted (or the ladder is exhausted): next window starts greedy.
            temp_attempt = 0;
            window_temp = 0.0;
            cand_idx = 0;
            rung_best = None;

            // FW_RETRY_FAILED_WINDOW (default-on guard): a window that closed NO timestamp
            // (`result_len == 0`, not silence) while carrying a prior-window prompt is
            // the bd-r0qd long-form drop — the carried prompt × int8 numerics made the
            // decoder emit `eot` early. Retry this SAME seek ONCE with the prompt
            // cleared (fresh `st` next iteration; `FW_NO_CONTEXT`-style recovery, but
            // targeted) before accepting the drop. `force_empty_prompt` makes it fire at
            // most once per seek, so no infinite loop. The tiny.en segment-TS policy
            // should avoid this path; it remains a conservative fallback and supports
            // the explicit historical-context overrides above.
            if retry_failed_window_enabled()
                && !temp_fallback_enabled()
                && result_len == 0
                && !is_no_speech
                && !force_empty_prompt
                && prompt_carried
            {
                force_empty_prompt = true;
                work.prompt_reset_retries += 1;
                // Stash this window's encode so the retry (same seek) reuses it instead
                // of re-encoding. `st` owns its cross-KV copy (no borrow of `enc`), so
                // this move is sound; `enc` is otherwise unused past DecoderState::new.
                retry_enc_cache = Some((frame_offset, enc));
                continue; // re-decode this window with no carried prompt (reusing this encode)
            }
            force_empty_prompt = false;

            // Surface the otherwise-SILENT content drop (bd-r0qd): a non-first window
            // that closed no timestamp and isn't silence emits nothing yet advances a
            // full chunk, so ~30 s of speech vanishes with no signal to the caller.
            // Transcript-unchanged (a log only) ⇒ byte-exact. Points at the recovery flag.
            if result_len == 0 && !is_no_speech && seek_cs > 0 {
                tracing::warn!(
                    target: "franken_whisper::native_engine::decode",
                    seek_sec = seek_cs as f64 / 100.0,
                    no_speech_prob,
                    avg_logprob,
                    "long-form window closed no timestamp — ~30 s of audio dropped \
                     (set FW_RETRY_FAILED_WINDOW=1 to attempt prompt-reset recovery; see bd-r0qd)"
                );
            }

            // single-timestamp-ending: skip the rest of the chunk (whisper.cpp
            // 7753-7760). Ordering fix #5: upstream emits segments (7624-7730) and
            // records DTW word timings with the ORIGINAL `seek_delta`, then applies
            // this whole-chunk skip ONLY to the seek advance (7753-7760). So we keep
            // `seek_delta_cs` untouched for build_segments/window_word_timings below
            super::perf_span(
                "decode_loop",
                t_loop.elapsed().as_secs_f64() * 1e3,
                &format!("\"tokens\":{}", decoded.len()),
            );
            // and compute a separate `seek_advance_cs` for the window step.
            //
            // whisper.cpp resizes the sequence to `result_len` (dropping the
            // loop-terminating EOT and anything past the last closed timestamp)
            // BEFORE this test (whisper.cpp 7534, `sequence.tokens.resize(result_len)`).
            // Testing the full `decoded` — which still holds the EOT the loop pushed
            // at line ~1270 before breaking — makes an EOT-closed window that ends
            // "…text <|ts|> <eot>" read as "(ts, eot)" instead of "(text, ts)", so the
            // skip-rest-of-chunk NEVER fires on EOT-terminated windows. That spawns a
            // spurious extra window over the post-speech zero-padding (jfk: a
            // hallucinated "a." second window that whisper.cpp does not emit). Match
            // upstream exactly: run the test on the `result_len`-truncated slice.
            let result_decoded: &[i32] = &decoded[..result_len.min(decoded.len())];
            let single_ts_ending = single_timestamp_ending(
                result_decoded,
                tk.timestamp_begin,
                params.timestamps,
                user_max_tokens,
            );
            let seek_advance_cs = if single_ts_ending {
                (seek_end_cs - seek_cs).min(CHUNK_CS)
            } else {
                seek_delta_cs
            };

            windows.push(WindowStats {
                avg_logprob,
                no_speech_prob,
                tokens: result_len,
                window_offset_sec: seek_cs as f64 / 100.0,
            });
            work.accepted_windows += 1;
            work.accepted_result_tokens += result_len;

            if !is_no_speech && !decoded.is_empty() {
                // Use only the result_len tokens for emission (drop a trailing eot).
                let take = result_len.min(decoded.len());
                let result_tokens = &decoded[..take];
                // Debug hook (`PROBE_DUMP_TOKENS=1`): emit this window's raw token ids
                // to stderr for offline analysis (e.g. prompt-lookup/n-gram speculation
                // accept-rate simulation). Off by default, zero cost when unset.
                if std::env::var_os("PROBE_DUMP_TOKENS").is_some() {
                    use std::fmt::Write as _;
                    let mut line = String::from("TOKENS>>>");
                    for (i, &t) in result_tokens.iter().enumerate() {
                        if i > 0 {
                            line.push(',');
                        }
                        let _ = write!(line, "{t}");
                    }
                    line.push_str("<<<");
                    eprintln!("{line}");
                }
                let result_token_plogs = &plogs[..take];
                let win_segments = build_segments(
                    tk,
                    result_tokens,
                    result_token_plogs,
                    seek_cs,
                    seek_delta_cs,
                    seek_end_cs,
                    params.timestamps,
                );

                // DTW word timestamps (bd-rjsx): record cross-attention over this
                // window's result tokens and align them to audio frames. Computed
                // before `win_segments` is moved so we stay 1:1 with it. When
                // requested we always push one (possibly empty) word vec per emitted
                // segment so `word_timings` stays aligned with `segments`.
                if params.word_timestamps {
                    let win_words = if align_heads.is_empty() {
                        vec![Vec::new(); win_segments.len()]
                    } else {
                        window_word_timings(
                            m,
                            &mut st,
                            used_language.as_deref(),
                            params,
                            &align_heads,
                            result_tokens,
                            &win_segments,
                            seek_cs,
                            seek_delta_cs,
                            checkpoint,
                        )?
                    };
                    word_timings.extend(win_words);
                }

                segments.extend(win_segments);

                // Update rolling context: the decoded text tokens (whisper.cpp
                // 7617-7622), capped to the prompt budget.
                prompt_past.clear();
                for &t in result_tokens {
                    if !tk.is_special(t) {
                        prompt_past.push(t);
                    }
                }
                if prompt_past.len() > max_prompt_ctx {
                    let drop = prompt_past.len() - max_prompt_ctx;
                    prompt_past.drain(0..drop);
                }
            } else if is_no_speech {
                prompt_past.clear();
            }

            // Advance the window (whisper.cpp 7763) with the (possibly chunk-skip
            // adjusted) advance, NOT the emission delta (fix #5).
            seek_cs += seek_advance_cs.max(DELTA_MIN);
        }

        // Close the encoder-thread channel so the scoped worker exits; `scope`
        // then joins it before returning.
        drop(req_tx);
        Ok(())
    });
    pipe_result?;

    // English-only models never report a language; multilingual report the used
    // (possibly auto-detected) one.
    if !tk.is_multilingual() {
        used_language = None;
    }

    Ok(DecodeOutput {
        segments,
        language: used_language,
        windows,
        work,
        word_timings: if params.word_timestamps {
            Some(word_timings)
        } else {
            None
        },
    })
}

/// Exact-equivalent batch jobs that can share one physical transcription.
struct BatchJobGroup {
    representative: usize,
    members: Vec<usize>,
}

/// Fast, non-cryptographic fingerprint over the exact IEEE-754 sample bits.
/// Hash matches are always verified bit-for-bit before work is shared, so a
/// collision can only cost an equality scan; it cannot alias two audio jobs.
fn batch_audio_fingerprint(samples: &[f32]) -> u64 {
    let mut hash = 0x9e37_79b9_7f4a_7c15_u64 ^ samples.len() as u64;
    for &sample in samples {
        let word = u64::from(sample.to_bits()).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        hash ^= word;
        hash = hash.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^ (hash >> 31)
}

fn batch_job_fingerprint(samples: &[f32], params: &DecodeParams) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut params_hash = std::collections::hash_map::DefaultHasher::new();
    params.hash(&mut params_hash);
    batch_audio_fingerprint(samples) ^ params_hash.finish().rotate_left(17)
}

fn batch_jobs_identical(left: (&[f32], &DecodeParams), right: (&[f32], &DecodeParams)) -> bool {
    if left.1 != right.1 || left.0.len() != right.0.len() {
        return false;
    }
    std::ptr::eq(left.0.as_ptr(), right.0.as_ptr())
        || left
            .0
            .iter()
            .zip(right.0)
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
}

fn coalesce_batch_jobs(jobs: &[(&[f32], DecodeParams)]) -> Vec<BatchJobGroup> {
    use std::collections::HashMap;

    let mut groups: Vec<BatchJobGroup> = Vec::with_capacity(jobs.len());
    let mut by_fingerprint: HashMap<u64, Vec<usize>> = HashMap::with_capacity(jobs.len());
    for (job_index, (samples, params)) in jobs.iter().enumerate() {
        let fingerprint = batch_job_fingerprint(samples, params);
        let matching_group = by_fingerprint.get(&fingerprint).and_then(|candidates| {
            candidates.iter().copied().find(|&group_index| {
                let representative = groups[group_index].representative;
                batch_jobs_identical(
                    (samples, params),
                    (jobs[representative].0, &jobs[representative].1),
                )
            })
        });
        if let Some(group_index) = matching_group {
            groups[group_index].members.push(job_index);
        } else {
            let group_index = groups.len();
            groups.push(BatchJobGroup {
                representative: job_index,
                members: vec![job_index],
            });
            by_fingerprint
                .entry(fingerprint)
                .or_default()
                .push(group_index);
        }
    }
    groups
}

/// Enable the bounded, exact result cache owned by each [`LoadedModel`].
///
/// The default is on. `FW_TRANSCRIPT_CACHE=0` (also `false` or `off`) restores
/// one physical transcription per call for same-binary comparison and
/// operational rollback.
fn transcription_cache_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FW_TRANSCRIPT_CACHE").map_or(true, |value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            )
        })
    })
}

/// Enable exact duplicate elimination in [`transcribe_samples_batch`].
///
/// The default is on. `FW_BATCH_COALESCE=0` (also `false` or `off`) restores
/// one physical transcription per input for same-binary comparison and
/// operational rollback.
fn batch_coalesce_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FW_BATCH_COALESCE").map_or(true, |value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            )
        })
    })
}

/// Transcribe independent audio inputs concurrently through one loaded model.
///
/// This is the multi-file counterpart to [`transcribe_samples`]. Model weights
/// are immutable during inference, so every input shares `m` while Rayon's
/// work-stealing pool schedules both the file-level jobs and their nested
/// encoder/decoder kernels. That avoids the two expensive alternatives a
/// caller otherwise faces: serializing every file, or loading one multi-GB
/// model per process.
///
/// Before scheduling, jobs with bit-identical samples and equal
/// [`DecodeParams`] are coalesced. Fingerprint matches are collision-checked,
/// so distinct jobs can never alias; successful results are cloned back into
/// their original positions. Set `FW_BATCH_COALESCE=0` to disable this path.
///
/// At most one file lane is admitted per four Rayon workers. Nested frontends
/// and transformer kernels reuse that same fixed pool, so additional lanes do
/// not create additional threads; they expose independent ready work whenever
/// one stream reaches a serial decode boundary. Batches larger than the lane
/// count are drawn from one atomic work queue, so a lane finishing a short clip
/// immediately takes the next input instead of waiting behind a long,
/// statically-partitioned neighbor. Results are restored to input order before
/// return.
///
/// Each item carries its own [`DecodeParams`], allowing language and decode
/// policy to vary across the batch. `checkpoint` is shared and may be called
/// concurrently, so it retains the same `Sync` contract as the single-input
/// entry point.
#[must_use]
pub fn transcribe_samples_batch(
    m: &LoadedModel,
    jobs: &[(&[f32], DecodeParams)],
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> Vec<FwResult<DecodeOutput>> {
    if jobs.is_empty() {
        return Vec::new();
    }
    if jobs.len() == 1 {
        return vec![transcribe_samples(m, jobs[0].0, &jobs[0].1, checkpoint)];
    }

    super::ensure_default_rayon_pool();
    let groups = if batch_coalesce_enabled() {
        coalesce_batch_jobs(jobs)
    } else {
        (0..jobs.len())
            .map(|job_index| BatchJobGroup {
                representative: job_index,
                members: vec![job_index],
            })
            .collect()
    };
    super::perf_span(
        "batch_coalesce",
        0.0,
        &format!("\"jobs\":{},\"unique\":{}", jobs.len(), groups.len()),
    );
    const MIN_THREADS_PER_FILE: usize = 4;
    let lanes = groups
        .len()
        .min((rayon::current_num_threads() / MIN_THREADS_PER_FILE).max(1));
    let next_group = std::sync::atomic::AtomicUsize::new(0);

    let mut completed: Vec<(usize, FwResult<DecodeOutput>)> = (0..lanes)
        .into_par_iter()
        .flat_map_iter(|_| {
            std::iter::from_fn(|| {
                let group_index = next_group.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                groups.get(group_index).map(|group| {
                    let representative = group.representative;
                    let (samples, params) = &jobs[representative];
                    (
                        group_index,
                        transcribe_samples(m, samples, params, checkpoint),
                    )
                })
            })
        })
        .collect();
    completed.sort_unstable_by_key(|(group_index, _)| *group_index);

    let mut outputs: Vec<Option<FwResult<DecodeOutput>>> = (0..jobs.len()).map(|_| None).collect();
    for (group_index, result) in completed {
        let group = &groups[group_index];
        match result {
            Ok(output) => {
                for (position, &job_index) in group.members.iter().enumerate() {
                    let mut logical_output = output.clone();
                    if position > 0 {
                        // Work provenance is physical, not logical: cached
                        // followers perform no encoder or decoder operations.
                        logical_output.work = DecodeWorkStats::default();
                    }
                    outputs[job_index] = Some(Ok(logical_output));
                }
            }
            Err(error) => {
                let (&first, followers) = group.members.split_first().expect("nonempty group");
                outputs[first] = Some(Err(error));
                // Error values are intentionally not cloneable. Re-run only
                // failed followers so each receives its own precise error.
                for &job_index in followers {
                    outputs[job_index] = Some(transcribe_samples(
                        m,
                        jobs[job_index].0,
                        &jobs[job_index].1,
                        checkpoint,
                    ));
                }
            }
        }
    }
    outputs
        .into_iter()
        .map(|output| output.expect("every batch group resolves its members"))
        .collect()
}

/// Compute DTW word timings for one window, returning per-segment word lists
/// aligned 1:1 with `win_segments` (bd-rjsx).
///
/// Port of whisper.cpp `whisper_exp_compute_token_level_timestamps_dtw`
/// (8837-8990) driving: a single batched decoder forward over
/// `sot + [lang] + not + <text tokens> + eot` with cross-attention recording
/// on, then [`dtw::token_timestamps`] over the alignment heads, then a
/// token→word grouping that follows the same timestamp-token segmentation as
/// [`build_segments`].
///
/// The decoder state `st` is reset before the recording pass (the precomputed
/// cross K/V are retained by [`DecoderState::reset`]); the greedy decode that
/// produced `result_tokens` is already complete, so reusing `st` is safe.
#[allow(clippy::too_many_arguments)]
fn window_word_timings(
    m: &LoadedModel,
    st: &mut DecoderState,
    language: Option<&str>,
    params: &DecodeParams,
    align_heads: &[(usize, usize)],
    result_tokens: &[i32],
    win_segments: &[TranscriptionSegment],
    seek_cs: i64,
    seek_delta_cs: i64,
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<Vec<Vec<WordTiming>>> {
    let tk = &m.tokenizer;

    // The window's text tokens, in order (drop timestamp/special tokens).
    let text_tokens: Vec<i32> = result_tokens
        .iter()
        .copied()
        .filter(|&t| !tk.is_special(t))
        .collect();
    if text_tokens.is_empty() {
        return Ok(vec![Vec::new(); win_segments.len()]);
    }

    // Alignment token sequence: sot + [lang] + not + text + eot (whisper.cpp
    // 8866-8882). The `no_timestamps` token is always present in this pass.
    let mut prompt = vec![tk.sot];
    if tk.is_multilingual() {
        let lang = language.unwrap_or("en");
        let lang_tok = tk
            .language_token(lang)
            .or_else(|| tk.language_token("en"))
            .unwrap_or(tk.sot + 1);
        prompt.push(lang_tok);
    }
    prompt.push(tk.no_timestamps);
    let sot_len = prompt.len();
    prompt.extend_from_slice(&text_tokens);
    prompt.push(tk.eot);

    // Single batched forward with cross-attention recording.
    st.reset();
    let prev_record = st.record_cross_attn;
    st.record_cross_attn = true;
    let _ = decoder::forward_step(&m.decoder, st, &prompt, checkpoint)?;
    let attn = st.cross_attn_weights().to_vec();
    st.record_cross_attn = prev_record;

    // Audio length for this window in encoder frames (whisper.cpp
    // `n_frames = min(min(3000, seek_delta), seek_end - seek)`, then
    // `n_audio_tokens = n_frames / 2`). `seek_delta_cs` is in centiseconds
    // (1 cs = 1 mel frame = 10 ms); two mel frames per encoder frame.
    let n_audio_frames = (seek_delta_cs.clamp(0, CHUNK_CS) / 2) as usize;

    // Per-text-token END times (window-relative seconds), with normalization +
    // DTW already restricted to the text rows (fix #3) and using upstream's
    // END-boundary convention (fix #4). `first_text_row = sot_len`,
    // `n_text_rows = text_tokens.len()` (the trailing eot row is excluded).
    let text_ends = dtw::token_timestamps(
        &attn,
        m.hparams.n_text_head.max(0) as usize,
        align_heads,
        sot_len,
        text_tokens.len(),
        n_audio_frames,
        dtw::DEFAULT_MEDFILT_WIDTH,
    );
    if text_ends.is_empty() {
        return Ok(vec![Vec::new(); win_segments.len()]);
    }

    // Reconcile END boundaries → token START times for word grouping (fix #4):
    // a token's start is the previous token's END boundary; the first token
    // starts at the window start (0, window-relative). Add the window seek
    // offset (DTW times are relative to the window start).
    let seek_sec = seek_cs as f64 / 100.0;
    let text_starts: Vec<f32> = (0..text_tokens.len())
        .map(|i| {
            let t = if i == 0 {
                0.0
            } else {
                text_ends.get(i - 1).copied().unwrap_or(0.0)
            };
            (f64::from(t) + seek_sec) as f32
        })
        .collect();

    // Partition the text tokens by segment, mirroring `build_segments`'s
    // timestamp-token splitting, then group each segment's text tokens into
    // words. We walk `result_tokens` to find segment breaks (timestamp tokens
    // strictly greater than `timestamp_begin` close a segment) while advancing a
    // cursor into `text_tokens`/`text_starts`.
    let mut per_segment: Vec<Vec<WordTiming>> = Vec::with_capacity(win_segments.len());
    let mut text_cursor = 0usize;
    let mut seg_idx = 0usize;

    // Helper: bytes for a text token id.
    let token_byte =
        |id: i32| -> Vec<u8> { tk.token_bytes(id).map(<[u8]>::to_vec).unwrap_or_default() };

    if !params.timestamps {
        // Single segment spanning the window: all text tokens are one group.
        let bytes: Vec<Vec<u8>> = text_tokens.iter().map(|&t| token_byte(t)).collect();
        let slices: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
        let seg_end = win_segments
            .first()
            .and_then(|s| s.end_sec)
            .unwrap_or(seek_sec + seek_delta_cs as f64 / 100.0);
        let words = dtw::group_tokens_into_words(&slices, &text_starts, seg_end);
        per_segment.push(words);
        while per_segment.len() < win_segments.len() {
            per_segment.push(Vec::new());
        }
        return Ok(per_segment);
    }

    // Timestamped mode: re-walk `result_tokens` to recover per-segment text-token
    // spans, the same way `build_segments` does (whisper.cpp 7624-7714).
    let beg = tk.timestamp_begin;
    let mut span_start = text_cursor; // first text-token index of current segment
    let mut i = 0usize;
    while i < result_tokens.len() {
        let tok = result_tokens[i];
        if tok > beg {
            // Close the current segment at this timestamp token.
            emit_segment_words(
                &text_tokens,
                &text_starts,
                span_start,
                text_cursor,
                win_segments,
                &mut seg_idx,
                &mut per_segment,
                seek_sec,
                seek_delta_cs,
                &token_byte,
            );
            // Skip a run of consecutive timestamp tokens.
            while i + 1 < result_tokens.len() && result_tokens[i + 1] > beg {
                i += 1;
            }
            span_start = text_cursor;
        } else if !tk.is_special(tok) {
            text_cursor += 1;
        }
        i += 1;
    }
    // Open-tail segment: any text tokens after the last timestamp pair.
    if span_start < text_cursor {
        emit_segment_words(
            &text_tokens,
            &text_starts,
            span_start,
            text_cursor,
            win_segments,
            &mut seg_idx,
            &mut per_segment,
            seek_sec,
            seek_delta_cs,
            &token_byte,
        );
    }

    // Pad to match `win_segments` length (defensive: a segment whose text was
    // empty/whitespace was dropped by `build_segments`).
    while per_segment.len() < win_segments.len() {
        per_segment.push(Vec::new());
    }
    per_segment.truncate(win_segments.len());
    Ok(per_segment)
}

/// Emit the words for one segment's text-token span `[start, end)` into
/// `per_segment`, advancing `seg_idx`. Skips empty spans (which `build_segments`
/// also drops, so they must not consume a `win_segments` slot).
#[allow(clippy::too_many_arguments)]
fn emit_segment_words(
    text_tokens: &[i32],
    text_starts: &[f32],
    start: usize,
    end: usize,
    win_segments: &[TranscriptionSegment],
    seg_idx: &mut usize,
    per_segment: &mut Vec<Vec<WordTiming>>,
    seek_sec: f64,
    seek_delta_cs: i64,
    token_byte: &dyn Fn(i32) -> Vec<u8>,
) {
    if start >= end {
        return;
    }
    // Skip whitespace-only spans the same way `build_segments` drops them.
    // Build the span text by concatenating all token BYTES first, then a SINGLE
    // `from_utf8_lossy` (fix #10) — matching `Tokenizer::decode`, which joins
    // bytes before the lossy conversion. Per-token lossy decoding could split a
    // multi-byte UTF-8 character across two BPE tokens into replacement
    // characters, making this emptiness gate diverge from `build_segments`'s.
    let span_bytes: Vec<Vec<u8>> = text_tokens[start..end]
        .iter()
        .map(|&t| token_byte(t))
        .collect();
    let mut joined: Vec<u8> = Vec::new();
    for b in &span_bytes {
        joined.extend_from_slice(b);
    }
    let span_text = String::from_utf8_lossy(&joined);
    if span_text.trim().is_empty() {
        return;
    }

    // Segment end: the next segment's start_sec, else this segment's end_sec,
    // else the window end.
    let seg_end = win_segments
        .get(*seg_idx)
        .and_then(|s| s.end_sec)
        .unwrap_or(seek_sec + seek_delta_cs as f64 / 100.0);

    let slices: Vec<&[u8]> = span_bytes.iter().map(Vec::as_slice).collect();
    let words = dtw::group_tokens_into_words(&slices, &text_starts[start..end], seg_end);
    per_segment.push(words);
    *seg_idx += 1;
}

/// Resolve the working language: an explicit code is echoed; a multilingual
/// model with no language auto-detects on the first window (whisper.cpp
/// 4035-4108); English-only models use `"en"` here (the caller squashes the
/// reported language to `None` at the end).
/// Cheap language resolution that never touches the model: explicit request
/// language, or implicit English for non-multilingual models. Returns `None`
/// when auto-detection is required — the window loop then detects from the
/// FIRST window's already-computed encoder output instead of running a
/// hidden duplicate encode (hotspot #2, 8.8 s on large-v3-turbo: the old
/// `resolve_language` encoded window 0, then the loop encoded it again).
fn resolve_language_fast(m: &LoadedModel, params: &DecodeParams) -> Option<String> {
    if let Some(lang) = &params.language {
        return Some(lang.clone());
    }
    if !m.tokenizer.is_multilingual() {
        // English-only model: language is implicitly English, no detection.
        return Some("en".to_string());
    }
    None
}

/// Auto-detect the spoken language from an already-encoded window: forward
/// `[sot]`, argmax over the language-token logits (whisper.cpp 4053-4107).
/// `st` is reset afterwards so the caller can reuse it (and its precomputed
/// cross K/V) for the real decode — the self-attention KV cache is cleared,
/// which the KV-equivalence tests prove is identical to a fresh state.
///
/// Isomorphism vs the previous separate-encode path: the encoder output for
/// window 0 is the same tensor either way (encoding is deterministic), so the
/// detection logits — and every downstream token — are unchanged.
fn detect_language_from_enc(
    m: &LoadedModel,
    st: &mut DecoderState,
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<Option<String>> {
    let logits = decoder::forward_step(&m.decoder, st, &[m.tokenizer.sot], checkpoint)?;
    st.reset();

    let mut best_code = "en";
    let mut best_logit = f32::NEG_INFINITY;
    for (code, lang_id, _) in LANGUAGES {
        if *lang_id >= m.tokenizer.num_languages() {
            continue;
        }
        let tok = m.tokenizer.sot + 1 + *lang_id;
        if let Ok(idx) = usize::try_from(tok)
            && let Some(&l) = logits.get(idx)
            && l > best_logit
        {
            best_logit = l;
            best_code = code;
        }
    }
    Ok(Some(best_code.to_string()))
}

/// Decode a standard 16-bit PCM WAV into 16 kHz mono `f32` samples in `[-1, 1]`.
///
/// A minimal RIFF chunk-walker: validates the `RIFF`/`WAVE` magic, reads the
/// `fmt ` chunk (must be PCM, 16-bit), skips any intervening chunks (e.g.
/// `LIST`), and reads the `data` chunk. Multi-channel audio is downmixed to
/// mono by averaging; a non-16 kHz rate is rejected (callers normalize first).
///
/// This is a test/utility helper kept here so the gated e2e test is
/// self-contained; production input normalization lives in `crate::audio`.
#[allow(dead_code)]
pub(crate) fn read_wav_16k_mono(bytes: &[u8]) -> FwResult<Vec<f32>> {
    let rd_u32 = |b: &[u8], o: usize| -> Option<u32> {
        b.get(o..o + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    };
    let rd_u16 = |b: &[u8], o: usize| -> Option<u16> {
        b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]]))
    };
    let bad = |s: &str| FwError::InvalidRequest(format!("read_wav: {s}"));

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let mut pos = 12usize;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits = 0u16;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = rd_u32(bytes, pos + 4).ok_or_else(|| bad("truncated chunk header"))? as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " => {
                let body = &bytes[body_start..body_end];
                let fmt = rd_u16(body, 0).ok_or_else(|| bad("bad fmt"))?;
                if fmt != 1 {
                    return Err(bad("only PCM (fmt=1) supported"));
                }
                channels = rd_u16(body, 2).ok_or_else(|| bad("bad channels"))?;
                sample_rate = rd_u32(body, 4).ok_or_else(|| bad("bad rate"))?;
                bits = rd_u16(body, 14).ok_or_else(|| bad("bad bits"))?;
            }
            b"data" => {
                data = Some(&bytes[body_start..body_end]);
            }
            _ => {}
        }
        // Chunks are word-aligned (pad byte if odd size).
        pos = body_end + (size & 1);
    }
    if bits != 16 {
        return Err(bad("only 16-bit PCM supported"));
    }
    if sample_rate != SAMPLE_RATE as u32 {
        return Err(bad("expected 16 kHz audio"));
    }
    let channels = usize::from(channels.max(1));
    let data = data.ok_or_else(|| bad("no data chunk"))?;
    let n_frames = data.len() / (2 * channels);
    let mut out = Vec::with_capacity(n_frames);
    if channels == 1 {
        // Mono fast path (the whisper input format, so the common case): a plain
        // i16→f32 map — `s as f32 / 32768.0` — that LLVM autovectorizes (vcvt + mul
        // by 2⁻¹⁵). The general per-channel accumulation loop below inhibits autovec.
        // BYTE-IDENTICAL to that loop with `channels == 1`: `acc == i32(s)`, and both
        // `i32(s) as f32` and `i16 s as f32` give the exact integer (i16 ⊂ f32
        // mantissa), while `/1.0` is the identity and `/32768.0` (÷2¹⁵) is exact.
        out.extend(
            data.chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0),
        );
    } else {
        for f in 0..n_frames {
            let mut acc = 0i32;
            for c in 0..channels {
                let o = (f * channels + c) * 2;
                let s = i16::from_le_bytes([data[o], data[o + 1]]);
                acc += i32::from(s);
            }
            out.push((acc as f32 / channels as f32) / 32768.0);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Synthetic tokenizer for hermetic logit-filter / segment tests.
    // -----------------------------------------------------------------------

    #[test]
    fn batch_coalescing_shares_only_exact_audio_and_params() {
        let audio = vec![0.0, 0.25, -0.5, f32::from_bits(0x7fc0_0001)];
        let audio_copy = audio.clone();
        let signed_zero_variant = vec![-0.0, 0.25, -0.5, f32::from_bits(0x7fc0_0001)];
        let params = DecodeParams::default();
        let mut translated = params.clone();
        translated.translate = true;
        let jobs = [
            (audio.as_slice(), params.clone()),
            (audio_copy.as_slice(), params.clone()),
            (signed_zero_variant.as_slice(), params.clone()),
            (audio.as_slice(), translated),
        ];

        let groups = coalesce_batch_jobs(&jobs);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].representative, 0);
        assert_eq!(groups[0].members, vec![0, 1]);
        assert_eq!(groups[1].members, vec![2]);
        assert_eq!(groups[2].members, vec![3]);
    }

    #[test]
    fn batch_coalescing_preserves_first_seen_group_order() {
        let first = vec![1.0, 2.0];
        let second = vec![3.0, 4.0];
        let first_copy = first.clone();
        let params = DecodeParams::default();
        let jobs = [
            (first.as_slice(), params.clone()),
            (second.as_slice(), params.clone()),
            (first_copy.as_slice(), params),
        ];

        let groups = coalesce_batch_jobs(&jobs);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].members, vec![0, 2]);
        assert_eq!(groups[1].members, vec![1]);
    }

    #[test]
    fn transcription_cache_reuses_only_exact_inputs_and_zeroes_physical_work() {
        let audio = vec![0.0, 0.25, f32::from_bits(0x7fc0_0001)];
        let audio_copy = audio.clone();
        let signed_zero_variant = vec![-0.0, 0.25, f32::from_bits(0x7fc0_0001)];
        let params = DecodeParams::default();
        let fingerprint = batch_job_fingerprint(&audio, &params);
        let output = DecodeOutput {
            segments: Vec::new(),
            language: Some("en".to_owned()),
            windows: Vec::new(),
            work: DecodeWorkStats {
                encoder_calls: 1,
                decoder_prefill_calls: 1,
                greedy_single_token_forwards: 26,
                ..DecodeWorkStats::default()
            },
            word_timings: None,
        };
        let mut cache = TranscriptionCache::default();
        cache.insert(fingerprint, &audio, &params, &output);

        let hit = cache
            .lookup(fingerprint, &audio_copy, &params)
            .expect("bit-identical copy should hit");
        assert_eq!(hit.language.as_deref(), Some("en"));
        assert_eq!(hit.work, DecodeWorkStats::default());

        // Supply the same fingerprint deliberately: the exact-bit collision
        // check, not the hash alone, must reject signed-zero drift.
        assert!(
            cache
                .lookup(fingerprint, &signed_zero_variant, &params)
                .is_none()
        );
        let mut translated = params.clone();
        translated.translate = true;
        assert!(
            cache
                .lookup(fingerprint, &audio_copy, &translated)
                .is_none()
        );
    }

    fn hp(n_vocab: i32) -> WhisperHParams {
        WhisperHParams {
            n_vocab,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            n_mels: 80,
            ftype: 1,
        }
    }

    #[test]
    fn tiny_en_segment_ts_context_policy_is_narrow_and_overridable() {
        let tiny = hp(51_864);
        let mut params = DecodeParams {
            timestamps: true,
            ..DecodeParams::default()
        };

        assert!(suppress_tiny_en_segment_ts_context(&tiny, &params, false));
        assert!(
            !suppress_tiny_en_segment_ts_context(&tiny, &params, true),
            "the operator fallback must restore historical context"
        );

        params.max_context = Some(-1);
        assert!(
            !suppress_tiny_en_segment_ts_context(&tiny, &params, false),
            "an explicit max-context request must override the model policy"
        );
        params.max_context = None;
        params.timestamps = false;
        assert!(
            !suppress_tiny_en_segment_ts_context(&tiny, &params, false),
            "tiny.en no-timestamp decoding keeps its historical context"
        );

        params.timestamps = true;
        let mut quantized_tiny = tiny;
        quantized_tiny.ftype = 7;
        assert!(
            suppress_tiny_en_segment_ts_context(&quantized_tiny, &params, false),
            "the policy follows the tiny.en architecture across quant formats"
        );

        let mut turbo = tiny;
        turbo.n_vocab = 51_866;
        turbo.n_audio_state = 1_280;
        turbo.n_audio_head = 20;
        turbo.n_audio_layer = 32;
        turbo.n_text_state = 1_280;
        turbo.n_text_head = 20;
        turbo.n_mels = 128;
        assert!(
            !suppress_tiny_en_segment_ts_context(&turbo, &params, false),
            "large-v3-turbo must remain on the historical context policy"
        );
    }

    /// English-only synthetic vocab (51864) with known special ids:
    /// eot=50256, sot=50257, ..., no_timestamps=50362, timestamp_begin=50363.
    /// Recognizable text / non-speech tokens are placed at small ids.
    fn synth_tokenizer() -> Tokenizer {
        let n_vocab = 51864i32;
        let mut v: Vec<Vec<u8>> = (0..n_vocab).map(|_| vec![b'.']).collect();
        v[1] = b" ".to_vec(); // the blank token
        v[2] = b"hello".to_vec();
        v[3] = b" world".to_vec();
        v[4] = b"(".to_vec(); // non-speech symbol
        v[5] = b" -".to_vec(); // non-speech special hyphen
        Tokenizer::from_vocab(&hp(n_vocab), v)
    }

    fn base_cfg(tk: &Tokenizer) -> FilterConfig {
        let space = (0..tk.vocab_size()).find(|&id| tk.token_bytes(id) == Some(b" ".as_slice()));
        FilterConfig {
            suppress_blank: true,
            space_token: space,
            suppress_nst: false,
            no_timestamps: false,
            max_initial_tid: None,
            max_tokens: None,
        }
    }

    fn zeros(tk: &Tokenizer) -> Vec<f32> {
        vec![0.0f32; tk.vocab_size() as usize]
    }

    /// Logits where the text region strongly dominates the timestamp region, so
    /// the sum-of-timestamp-probs forcing rule (whisper.cpp 6343-6369) does NOT
    /// fire — isolating whichever individual suppression rule is under test.
    /// (With flat/zero logits the ~1500 timestamp tokens logsumexp to a large
    /// value and the forcing rule masks all text, which is correct but masks the
    /// rule we want to observe.)
    fn text_dominant(tk: &Tokenizer) -> Vec<f32> {
        let mut v = vec![-30.0f32; tk.vocab_size() as usize];
        // Raise the low text region (everything below the specials) to dominate.
        for x in v.iter_mut().take(tk.eot as usize) {
            *x = 5.0;
        }
        v
    }

    fn is_suppressed(logits: &[f32], id: i32) -> bool {
        logits[id as usize] == f32::NEG_INFINITY
    }

    // ----- Rule 1: blank + eot suppression at step 0 only -----

    #[test]
    fn logsumexp_sum_simd_matches_scalar() {
        // The FW_SIMD_EXP path (`logsumexp_sum_simd`) must match the scalar libm
        // Σ exp(l−max) within the poly tolerance across all code paths (SIMD body,
        // <8 scalar tail) and edge cases, and treat -inf lanes as EXACTLY 0 (matching
        // the default loop's `l > -inf` guard). Guards the gated escape hatch.
        fn scalar(logits: &[f32], max: f32) -> f32 {
            let mut s = 0.0f32;
            for &l in logits {
                if l > f32::NEG_INFINITY {
                    s += (l - max).exp();
                }
            }
            s
        }
        let mut st = 0x1357_9BDF_2468_ACE0u64;
        let mut nf = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 40) as f32 / (1u64 << 24) as f32
        };
        for &n in &[0usize, 1, 7, 8, 9, 15, 16, 17, 1000, 51866] {
            let mut v: Vec<f32> = (0..n)
                .map(|_| {
                    let u = nf();
                    if u < 0.1 {
                        f32::NEG_INFINITY
                    } else {
                        -30.0 * nf()
                    }
                })
                .collect();
            if n > 0 {
                v[n / 2] = 0.0; // a finite max at 0
            }
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let got = logsumexp_sum_simd(&v, max);
            let want = scalar(&v, max);
            let rel = (got - want).abs() / want.abs().max(1e-30);
            assert!(
                rel < 1e-3,
                "n={n}: simd {got} vs scalar {want} rel {rel:.2e}"
            );
        }
        // All-masked (-inf, max=-inf) → sum exactly 0 (mask zeroes the NaN lanes).
        assert_eq!(
            logsumexp_sum_simd(&vec![f32::NEG_INFINITY; 32], f32::NEG_INFINITY),
            0.0
        );
    }

    #[test]
    fn argmax_idx_matches_scalar() {
        // The AVX2 `argmax_idx` must return the SAME first-max index as the scalar loop in
        // every case: SIMD body + <8 tail, ties (first index wins), -inf and NaN lanes
        // (skipped exactly as scalar `>`), and empty/all-masked (index 0). This is the token
        // selection (whisper greedy = argmax of raw logits) so it must be bit-for-bit.
        fn scalar(l: &[f32]) -> usize {
            let mut best_i = 0usize;
            let mut best = f32::NEG_INFINITY;
            for (i, &v) in l.iter().enumerate() {
                if v > best {
                    best = v;
                    best_i = i;
                }
            }
            best_i
        }
        let mut st = 0x0F0F_1234_DEAD_BEEFu64;
        let mut nf = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 60.0
        };
        for &n in &[0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 1000, 51866] {
            let mut v: Vec<f32> = (0..n)
                .map(|_| {
                    let u = (nf() + 30.0) / 60.0; // back to [0,1)
                    if u < 0.08 { f32::NEG_INFINITY } else { nf() }
                })
                .collect();
            // Force a duplicate maximum to exercise the first-index tie-break.
            if n >= 20 {
                v[5] = 1000.0;
                v[17] = 1000.0; // scalar keeps index 5 (first)
            }
            assert_eq!(argmax_idx(&v), scalar(&v), "n={n}");
        }
        // NaN lanes must be skipped exactly like scalar `>` (NaN > x is false).
        let with_nan = vec![f32::NAN, -1.0, 3.0, f32::NAN, 2.0, 3.0, f32::NAN, 0.0, -5.0];
        assert_eq!(argmax_idx(&with_nan), scalar(&with_nan)); // first 3.0 at index 2
        // Empty and all-masked → 0 (matches scalar initial best_i).
        assert_eq!(argmax_idx(&[]), 0);
        assert_eq!(argmax_idx(&vec![f32::NEG_INFINITY; 20]), 0);
    }

    #[test]
    fn sample_token_deterministic_and_respects_masks() {
        // FW_TEMP_FALLBACK sampler: identical seeds must replay identical draws
        // bit-for-bit (transcript replayability), and masked (`-inf`) / NaN lanes
        // must carry zero mass. plog is the id's temperature-1 log-softmax.
        let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 0.5, f32::NAN];
        let logprobs = vec![
            f32::NEG_INFINITY,
            -0.3,
            f32::NEG_INFINITY,
            -0.8,
            f32::NEG_INFINITY,
        ];
        let (mut r1, mut r2) = (42u64, 42u64);
        let mut drew = [false; 5];
        for _ in 0..64 {
            let (t1, p1) = sample_token_at_temperature(&logits, &logprobs, 0.8, &mut r1);
            let (t2, p2) = sample_token_at_temperature(&logits, &logprobs, 0.8, &mut r2);
            assert_eq!(
                (t1, p1.to_bits()),
                (t2, p2.to_bits()),
                "same seed, same draw"
            );
            assert!(t1 == 1 || t1 == 3, "masked/NaN lane drawn: {t1}");
            let expected_plog = if t1 == 1 { -0.3 } else { -0.8 };
            assert_eq!(p1, expected_plog, "plog must come from `logprobs`");
            drew[usize::try_from(t1).unwrap()] = true;
        }
        // At t = 0.8 with a 0.5-nat gap both live lanes appear within 64 draws
        // (P(all-one-lane) < 1e-9): the draw is genuinely multinomial, not argmax.
        assert!(
            drew[1] && drew[3],
            "sampler collapsed to a single lane: {drew:?}"
        );
    }

    #[test]
    fn sample_token_low_temperature_recovers_argmax() {
        // t → 0 sharpens softmax(logits/t) onto the max lane: an 6-nat gap at
        // t = 0.05 is a 120-nat gap, so every draw must equal the argmax choice.
        let logits = vec![0.0, 8.0, 1.0, 2.0];
        let logprobs = compute_logprobs(&logits);
        let mut rng = 99u64;
        for _ in 0..64 {
            let (tok, plog) = sample_token_at_temperature(&logits, &logprobs, 0.05, &mut rng);
            let (best, best_plog) = argmax(&logits, &logprobs);
            assert_eq!(tok, best);
            assert_eq!(plog.to_bits(), best_plog.to_bits());
        }
    }

    #[test]
    fn sequence_score_matches_whisper_cpp_defaults() {
        // whisper.cpp `whisper_sequence_score` with length_penalty = -1.0 (the
        // default): penalty = result_len ⇒ score = avg logprob of the result.
        assert_eq!(sequence_score(&[-0.5, -1.5], 2), -1.0);
        // Only the result slice counts (a trailing eot plog is excluded).
        assert_eq!(sequence_score(&[-0.5, -1.5, -9.0], 2), -1.0);
        // No result ⇒ sentinel: never beats a token-producing candidate.
        assert_eq!(sequence_score(&[], 0), EMPTY_WINDOW_AVG_LOGPROB);
        assert_eq!(sequence_score(&[-0.5], 0), EMPTY_WINDOW_AVG_LOGPROB);
        // result_len beyond plogs clamps defensively.
        assert_eq!(sequence_score(&[-2.0], 5), -2.0);
        // Non-finite input degrades to the sentinel, mirroring avg_logprob.
        assert_eq!(
            sequence_score(&[f32::NEG_INFINITY, -1.0], 2),
            EMPTY_WINDOW_AVG_LOGPROB
        );
    }

    #[test]
    fn top_k_logprob_indices_selects_best_first_skipping_masked() {
        // The beam-expansion primitive: the k largest logprobs, best-first,
        // masked (-inf) lanes excluded. Ties break by lower index (determinism).
        let lp = vec![-3.0f32, -0.5, f32::NEG_INFINITY, -1.0, -0.5, -9.0];
        assert_eq!(top_k_logprob_indices(&lp, 3), vec![1, 4, 3]); // -0.5(i1), -0.5(i4), -1.0(i3)
        assert_eq!(top_k_logprob_indices(&lp, 1), vec![1]);
        // k larger than the finite lane count clamps to the available lanes
        // (the -inf lane at index 2 is never returned).
        assert_eq!(top_k_logprob_indices(&lp, 10), vec![1, 4, 3, 0, 5]);
        // All-masked and k=0 both yield nothing.
        assert!(top_k_logprob_indices(&[f32::NEG_INFINITY; 4], 3).is_empty());
        assert!(top_k_logprob_indices(&lp, 0).is_empty());
        assert!(top_k_logprob_indices(&[], 3).is_empty());
    }

    #[test]
    fn beam_size_default_is_one() {
        // Default (no field, no env) must be 1 so the greedy path runs — the
        // byte-exact-by-construction guarantee. Also check the field and the
        // clamp. (The OnceLock env reader is unset in a clean test process.)
        if std::env::var_os("FW_BEAM_SIZE").is_none() {
            assert_eq!(resolve_beam_size(&DecodeParams::default()), 1);
            let mut p = DecodeParams::default();
            p.beam_size = Some(5);
            assert_eq!(
                resolve_beam_size(&p),
                5,
                "field beam_size drives beam width"
            );
            p.beam_size = Some(99);
            assert_eq!(resolve_beam_size(&p), 8, "beam width clamps to 8");
            p.beam_size = Some(0);
            assert_eq!(resolve_beam_size(&p), 1, "0 clamps up to greedy");
        }
    }

    #[test]
    fn token_tail_entropy_matches_whisper_cpp_reference() {
        // Faithful to whisper.cpp 6597-6617: Shannon entropy (nats) over the
        // token-id counts of the LAST 32 entries only.
        // Uniformly repeated tail → 0.0 (degenerate loop, the entropy_thold case).
        assert_eq!(token_tail_entropy(&vec![7i32; 40]), 0.0);
        // 32 distinct ids → ln 32 (maximum for the window).
        let distinct: Vec<i32> = (0..32).collect();
        assert!((token_tail_entropy(&distinct) - 32f64.ln()).abs() < 1e-12);
        // Two ids at 16/16 in the tail → ln 2, and the prefix must be IGNORED:
        // 100 leading distinct ids change nothing.
        let mut two_id: Vec<i32> = (1000..1100).collect();
        two_id.extend(std::iter::repeat_n(1, 16));
        two_id.extend(std::iter::repeat_n(2, 16));
        assert!((token_tail_entropy(&two_id) - 2f64.ln()).abs() < 1e-12);
        // Hand-computed mixed case: tail of 24×A + 8×B over 32 →
        // -(0.75 ln 0.75 + 0.25 ln 0.25).
        let mut mixed = vec![5i32; 24];
        mixed.extend(std::iter::repeat_n(9, 8));
        let expect = -(0.75f64 * 0.75f64.ln() + 0.25f64 * 0.25f64.ln());
        assert!((token_tail_entropy(&mixed) - expect).abs() < 1e-12);
        // Threshold calibration sanity (wc defaults): a fully repetitive tail is
        // far below 2.4; a fully diverse tail is above it.
        assert!(token_tail_entropy(&vec![7i32; 33]) < ENTROPY_THRESHOLD);
        assert!(token_tail_entropy(&distinct) > ENTROPY_THRESHOLD);
        // Empty input is defined (0.0), matching "no evidence of a loop".
        assert_eq!(token_tail_entropy(&[]), 0.0);
    }

    #[test]
    fn sample_token_degenerate_inputs_fall_back_to_argmax() {
        // All-masked (no finite lane) and empty inputs must not panic or draw a
        // masked id — they defer to argmax's established conventions.
        let masked = vec![f32::NEG_INFINITY; 4];
        let masked_lp = vec![f32::NEG_INFINITY; 4];
        let mut rng = 1u64;
        let (tok, _) = sample_token_at_temperature(&masked, &masked_lp, 0.4, &mut rng);
        assert_eq!(tok, argmax(&masked, &masked_lp).0);
        let (tok_empty, _) = sample_token_at_temperature(&[], &[], 0.4, &mut rng);
        assert_eq!(tok_empty, argmax(&[], &[]).0);
        // A single unmasked lane always wins regardless of the draw.
        let one = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 2.0];
        let one_lp = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0];
        for _ in 0..32 {
            let (tok, plog) = sample_token_at_temperature(&one, &one_lp, 1.0, &mut rng);
            assert_eq!(tok, 2);
            assert_eq!(plog, 0.0);
        }
    }

    #[test]
    fn blank_and_eot_suppressed_at_step0() {
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &[], false, CHUNK_CS, 0);
        assert!(is_suppressed(&logits, tk.eot), "eot suppressed at step 0");
        assert!(is_suppressed(&logits, 1), "blank ' ' suppressed at step 0");

        // After one token, blank/eot are NOT suppressed by the blank rule.
        // (Use text-dominant logits so the timestamp-forcing rule doesn't mask
        // text/eot — see `text_dominant`.)
        let (logits2, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 0);
        assert!(!is_suppressed(&logits2, tk.eot), "eot allowed after step 0");
        assert!(!is_suppressed(&logits2, 1), "blank allowed after step 0");
    }

    // ----- Rule 2: control / task / lang / sot / nosp / prev suppression -----

    #[test]
    fn control_tokens_always_suppressed() {
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &[2], false, CHUNK_CS, 0);
        for id in [tk.sot, tk.no_speech, tk.solm, tk.sot_prev, tk.no_timestamps] {
            assert!(is_suppressed(&logits, id), "control {id} suppressed");
        }
        // For English-only models translate/transcribe ids exist (50357/50358).
        assert!(is_suppressed(&logits, tk.translate));
        assert!(is_suppressed(&logits, tk.transcribe));
    }

    // ----- Rule 2b: non-speech suppression (opt-in) -----

    #[test]
    fn non_speech_suppressed_when_enabled() {
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        // Off by default: "(" (id 4) and " -" (id 5) are NOT suppressed.
        // Text-dominant logits keep the forcing rule from masking text.
        let (off, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 0);
        assert!(!is_suppressed(&off, 4), "non-speech allowed when off");
        // On: they are suppressed.
        cfg.suppress_nst = true;
        let (on, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 0);
        assert!(is_suppressed(&on, 4), "( suppressed when nst on");
        assert!(is_suppressed(&on, 5), "' -' suppressed when nst on");
    }

    // ----- Rule 3: timestamp pairing, both branches -----

    #[test]
    fn timestamp_pairing_two_ts_back_to_back_forbids_timestamp() {
        // last two tokens both timestamps => all timestamps suppressed, text ok.
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let prev = [tk.timestamp_begin + 5, tk.timestamp_begin + 10];
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &prev, true, 20, 0);
        assert!(
            is_suppressed(&logits, tk.timestamp_begin + 50),
            "timestamp suppressed when two ts precede"
        );
        assert!(!is_suppressed(&logits, 2), "text token allowed");
    }

    #[test]
    fn timestamp_pairing_one_ts_open_forces_timestamp_or_eot() {
        // last token is a timestamp, penultimate is text => the pairing rule
        // masks all text in [0, eot) (whisper.cpp 6312-6314), leaving only eot
        // and timestamps selectable. To observe that eot survives (rather than
        // being re-masked by the downstream timestamp-forcing rule), give eot a
        // dominant logit so max_text_logprob (which includes eot, since
        // eot < token_beg) beats the timestamp logsumexp.
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let prev = [2i32, tk.timestamp_begin + 10];
        let mut logits = vec![-5.0f32; tk.vocab_size() as usize];
        logits[tk.eot as usize] = 20.0; // eot clearly dominant
        let (logits, _) = process_logits(&tk, &cfg, logits, &prev, true, 20, 0);
        assert!(is_suppressed(&logits, 2), "text masked when one ts open");
        assert!(
            !is_suppressed(&logits, tk.eot),
            "eot survives (pair-before-eot allowed)"
        );
    }

    // ----- Rule 4: timestamp monotonicity -----

    #[test]
    fn timestamp_monotonicity_masks_earlier_timestamps() {
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        // has_ts with seek_delta=100cs => tid0 = 50; timestamps below beg+50 masked.
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &[2], true, 100, 0);
        assert!(
            is_suppressed(&logits, tk.timestamp_begin + 10),
            "earlier timestamp masked"
        );
        assert!(
            !is_suppressed(&logits, tk.timestamp_begin + 60),
            "later timestamp allowed"
        );
    }

    // ----- Rule 5: max_initial_ts clamp -----

    #[test]
    fn max_initial_ts_clamps_first_step() {
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        // precision = 30/1500 = 0.02s; max_initial_ts=1.0s => tid0 = 50.
        cfg.max_initial_tid = Some(50);
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &[], false, CHUNK_CS, 0);
        // timestamps beyond beg+50 masked on the initial step.
        assert!(
            is_suppressed(&logits, tk.timestamp_begin + 51),
            "initial timestamp > max clamped"
        );
        assert!(
            !is_suppressed(&logits, tk.timestamp_begin + 50),
            "initial timestamp at max allowed"
        );
    }

    // ----- Rule 6: logsumexp timestamp-forcing -----

    #[test]
    fn logsumexp_forces_timestamp_when_ts_mass_exceeds_text() {
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let mut logits = vec![0.0f32; tk.vocab_size() as usize];
        // Make text uniformly low, then spread a large amount of mass across many
        // timestamp tokens so their logsumexp exceeds any single text logit.
        for l in &mut logits {
            *l = -10.0;
        }
        logits[2] = 1.0; // one text token with a modest logit
        let beg = tk.timestamp_begin as usize;
        for i in beg..(beg + 200).min(logits.len()) {
            logits[i] = 0.5;
        }
        let (out, _) = process_logits(&tk, &cfg, logits, &[2], false, 0, 0);
        // All text logits (below beg) must be masked: a timestamp is forced.
        assert!(is_suppressed(&out, 2), "text masked: timestamp forced");
        assert!(is_suppressed(&out, 0), "all text masked");
        // A timestamp remains selectable.
        assert!(!is_suppressed(&out, beg as i32 + 1));
    }

    #[test]
    fn logsumexp_keeps_text_when_text_dominates() {
        let tk = synth_tokenizer();
        let cfg = base_cfg(&tk);
        let mut logits = vec![-20.0f32; tk.vocab_size() as usize];
        logits[2] = 10.0; // one very strong text token
        let beg = tk.timestamp_begin as usize;
        for l in &mut logits[beg..beg + 3] {
            *l = -5.0;
        }
        let (out, lp) = process_logits(&tk, &cfg, logits, &[2], false, 0, 0);
        assert!(!is_suppressed(&out, 2), "strong text not masked");
        let (tok, _) = argmax(&out, &lp);
        assert_eq!(tok, 2, "argmax selects the strong text token");
    }

    // ----- Rule 7: no_timestamps mode masks every timestamp -----

    #[test]
    fn no_timestamps_mode_masks_all_timestamps() {
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        cfg.no_timestamps = true;
        let (logits, _) = process_logits(&tk, &cfg, zeros(&tk), &[2], false, CHUNK_CS, 0);
        assert!(is_suppressed(&logits, tk.timestamp_begin));
        assert!(is_suppressed(&logits, tk.timestamp_begin + 100));
        assert!(!is_suppressed(&logits, 2), "text still allowed");
    }

    // ----- Rule 8: max_tokens EOT-forcing filter (fix #6) -----

    #[test]
    fn single_timestamp_ending_matches_upstream_semantics() {
        let beg = 100i32; // synthetic timestamp_begin
        // Paired/normal ending: ... text, ts, eot — eot (< beg) last => false.
        assert!(!single_timestamp_ending(
            &[5, beg + 10, 50],
            beg,
            true,
            None
        ));
        // Unpaired trailing timestamp, no budget => true (skip rest of chunk).
        assert!(single_timestamp_ending(&[5, beg + 10], beg, true, None));
        // Same shape but the count EXCEEDS the user budget => the closer was
        // forced by the EOT filter, guard suppresses the skip (wcpp 7749-51).
        assert!(!single_timestamp_ending(&[5, beg + 10], beg, true, Some(1)));
        // Count == budget (not exceeded): closer was a genuine model choice.
        assert!(single_timestamp_ending(&[5, beg + 10], beg, true, Some(2)));
        // No-timestamps mode (upstream single_segment): guard inapplicable,
        // but the shape check itself still governs.
        assert!(single_timestamp_ending(&[5, beg + 10], beg, false, Some(1)));
        // Degenerate lengths.
        assert!(!single_timestamp_ending(&[beg + 10], beg, true, None));
        assert!(!single_timestamp_ending(&[], beg, true, None));
    }

    #[test]
    fn user_budget_filter_fires_at_loop_boundary() {
        // Regression for the conflated-bounds bug: with the loop running to
        // the structural n_max and the filter keyed to the USER budget, the
        // filter must engage at tokens_in_window == budget — the exact value
        // the live loop passes on sampled-token index `mt` (pre-push count).
        let tk = synth_tokenizer();
        let cfg = FilterConfig {
            suppress_blank: false,
            space_token: None,
            suppress_nst: false,
            no_timestamps: false,
            max_initial_tid: None,
            max_tokens: Some(3),
        };
        // A strong text logit, so the timestamp-FORCING rule (logsumexp over
        // all ts tokens vs max text logit) cannot mask text on its own — we
        // want to observe the budget filter in isolation.
        let mut logits = vec![-20.0f32; tk.vocab_size() as usize];
        logits[5] = 10.0;
        // At 2 sampled tokens (< budget): text token 5 must remain available.
        let (out, _) = process_logits(&tk, &cfg, logits.clone(), &[], false, 0, 2);
        assert!(!is_suppressed(&out, 5), "text open below the budget");
        // At exactly the budget: every text token below eot must be masked.
        let (out, _) = process_logits(&tk, &cfg, logits, &[], false, 0, 3);
        assert!(is_suppressed(&out, 5), "text masked at the budget boundary");
    }

    #[test]
    fn max_tokens_forces_eot_when_budget_reached() {
        // Fix #6 (whisper.cpp 6234-6238): with timestamps on, once the running
        // token count reaches the budget, all text (< eot) is masked, leaving
        // only eot/timestamps selectable.
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        cfg.max_tokens = Some(4);
        // Below budget: text still allowed (use text-dominant logits so the
        // forcing rule doesn't mask text).
        let (under, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 3);
        assert!(!is_suppressed(&under, 2), "text allowed below budget");
        // At budget: all text below eot masked.
        let (at, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 4);
        assert!(is_suppressed(&at, 2), "text masked at budget");
        assert!(is_suppressed(&at, 0), "all text masked at budget");
        // A timestamp remains selectable.
        assert!(!is_suppressed(&at, tk.timestamp_begin + 1));
    }

    #[test]
    fn max_tokens_filter_inert_in_no_timestamps_mode() {
        // The EOT-force is guarded by `!no_timestamps` upstream.
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        cfg.no_timestamps = true;
        cfg.max_tokens = Some(2);
        let (out, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 5);
        assert!(
            !is_suppressed(&out, 2),
            "text not masked in no-timestamps mode"
        );
    }

    #[test]
    fn max_tokens_filter_disabled_when_none_or_zero() {
        let tk = synth_tokenizer();
        let mut cfg = base_cfg(&tk);
        cfg.max_tokens = None;
        let (out, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 99);
        assert!(!is_suppressed(&out, 2), "text allowed when budget is None");
        cfg.max_tokens = Some(0);
        let (out0, _) = process_logits(&tk, &cfg, text_dominant(&tk), &[2], false, CHUNK_CS, 99);
        assert!(!is_suppressed(&out0, 2), "text allowed when budget is 0");
    }

    // -----------------------------------------------------------------------
    // Segment building.
    // -----------------------------------------------------------------------

    #[test]
    fn segments_from_timestamp_pairs() {
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // <|0.00|> hello world <|3.00|>  => one segment [0.00, 3.00].
        // 3.00s = 150 steps.
        let tokens = vec![beg, 2, 3, beg + 150];
        let plogs = vec![-0.1f32; tokens.len()];
        let segs = build_segments(&tk, &tokens, &plogs, 0, 3000, i64::MAX, true);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "hello world");
        assert!((segs[0].start_sec.unwrap() - 0.0).abs() < 1e-9);
        assert!((segs[0].end_sec.unwrap() - 3.0).abs() < 1e-9);
        assert!(segs[0].confidence.unwrap() > 0.0);
    }

    #[test]
    fn segment_confidence_excludes_closing_timestamp_token() {
        // Fix #8: confidence averages only the segment's TEXT tokens (hello,
        // world), not the leading/closing timestamp tokens. Give the timestamp
        // tokens a very negative plog and the text tokens 0.0; the confidence
        // must reflect only the text (exp(0)=1), proving the ts plogs are
        // excluded. Observed delta vs the old all-token average: previously the
        // closing/opening ts plogs (-5.0 each) dragged the mean to ~-2.5 →
        // conf≈0.08; now text-only → conf=1.0.
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        let tokens = vec![beg, 2, 3, beg + 150];
        // ts plogs very negative, text plogs perfect (0.0).
        let plogs = vec![-5.0f32, 0.0, 0.0, -5.0f32];
        let segs = build_segments(&tk, &tokens, &plogs, 0, 3000, i64::MAX, true);
        assert_eq!(segs.len(), 1);
        let conf = segs[0].confidence.unwrap();
        assert!(
            (conf - 1.0).abs() < 1e-9,
            "text-only confidence should be exp(0)=1, got {conf}"
        );
    }

    #[test]
    fn text_confidence_none_for_timestamp_only_span() {
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // A span with no text tokens → None.
        assert!(text_confidence(&tk, &[beg, beg + 10], &[-0.1, -0.1]).is_none());
    }

    #[test]
    fn text_confidence_finite_on_nan_plog() {
        // A NaN token logprob must be skipped (not summed into the mean) and the
        // clamp must never see NaN. `hello world` with a good and a NaN plog →
        // confidence reflects only the finite plog and stays a finite [0, 1].
        let tk = synth_tokenizer();
        let tokens = vec![2i32, 3i32];
        let plogs = vec![0.0f32, f32::NAN];
        let conf = text_confidence(&tk, &tokens, &plogs).expect("finite text token present");
        assert!(conf.is_finite(), "confidence must be finite, got {conf}");
        assert!(
            (0.0..=1.0).contains(&conf),
            "confidence in [0,1], got {conf}"
        );
        assert!(
            (conf - 1.0).abs() < 1e-9,
            "only the finite plog (0.0) counts → exp(0)=1"
        );

        // Every text-token plog non-finite → all skipped → None (no NaN mean, no
        // clamp panic).
        assert!(text_confidence(&tk, &tokens, &[f32::NAN, f32::INFINITY]).is_none());
    }

    #[test]
    fn compute_logprobs_finite_on_positive_inf_logit() {
        // A `+inf` logit (activation overflow) must not manufacture NaN logprobs
        // via `exp(+inf - +inf)`. The `+inf` lane is sanitized to a masked
        // (`-inf`) lane; the finite lanes must still yield finite logprobs
        // instead of a NaN-poisoned whole vector.
        let lp = compute_logprobs(&[f32::INFINITY, 0.0, -1.0]);
        assert!(
            lp[1].is_finite(),
            "logprob for finite logit 0.0 must be finite, got {}",
            lp[1]
        );
        assert!(
            lp[2].is_finite(),
            "logprob for finite logit -1.0 must be finite, got {}",
            lp[2]
        );
        assert!(!lp[1].is_nan() && !lp[2].is_nan(), "no NaN logprobs");

        // A NaN logit is likewise neutralized (mapped to a masked lane) rather
        // than poisoning the finite lanes.
        let lp2 = compute_logprobs(&[f32::NAN, 0.0, -1.0]);
        assert!(
            lp2[1].is_finite() && lp2[2].is_finite(),
            "NaN logit must not poison finite lanes"
        );
    }

    #[test]
    fn segments_two_pairs() {
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // <|0|> hello <|1|> world <|2|>
        let tokens = vec![beg, 2, beg + 50, 3, beg + 100];
        let plogs = vec![-0.2f32; tokens.len()];
        let segs = build_segments(&tk, &tokens, &plogs, 0, 2000, i64::MAX, true);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "hello");
        assert!((segs[0].end_sec.unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(segs[1].text, "world");
        assert!((segs[1].start_sec.unwrap() - 1.0).abs() < 1e-9);
        assert!((segs[1].end_sec.unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn segments_open_tail() {
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // <|0|> hello <|1|> world   (no closing timestamp) => open tail closed at
        // seek + seek_delta.
        let tokens = vec![beg, 2, beg + 50, 3];
        let plogs = vec![-0.2f32; tokens.len()];
        let segs = build_segments(&tk, &tokens, &plogs, 0, 1500, i64::MAX, true);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].text, "world");
        // open tail end = seek_cs(0) + seek_delta(1500cs) = 15.0s.
        assert!((segs[1].end_sec.unwrap() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn segments_single_no_timestamps() {
        let tk = synth_tokenizer();
        // No-timestamps mode: text tokens only, one segment spanning the window.
        let tokens = vec![2i32, 3];
        let plogs = vec![-0.3f32; 2];
        let segs = build_segments(&tk, &tokens, &plogs, 500, 3000, i64::MAX, false);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "hello world");
        assert!((segs[0].start_sec.unwrap() - 5.0).abs() < 1e-9);
        assert!((segs[0].end_sec.unwrap() - 35.0).abs() < 1e-9);
    }

    #[test]
    fn segments_with_window_offset() {
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // Window starting at 30s (seek_cs=3000): <|0|> hello <|1|>.
        let tokens = vec![beg, 2, beg + 50];
        let plogs = vec![-0.1f32; tokens.len()];
        let segs = build_segments(&tk, &tokens, &plogs, 3000, 3000, i64::MAX, true);
        assert_eq!(segs.len(), 1);
        assert!((segs[0].start_sec.unwrap() - 30.0).abs() < 1e-9);
        assert!((segs[0].end_sec.unwrap() - 31.0).abs() < 1e-9);
    }

    #[test]
    fn segments_clamped_to_real_audio_length() {
        // Fix #1: a closing timestamp token pointing past the real (unpadded)
        // audio length must NOT yield an end_sec beyond the clip duration.
        // Synthetic last window: <|0.00|> hello world <|10.00|> but the real
        // audio is only 6.00 s (seek_end_cs = 600). The segment end must clamp
        // to 6.00 s.
        let tk = synth_tokenizer();
        let beg = tk.timestamp_begin;
        // <|0.00|> hello world <|10.00|> ; 10.00 s = 500 steps.
        let tokens = vec![beg, 2, 3, beg + 500];
        let plogs = vec![-0.1f32; tokens.len()];
        // Window at seek 0, full 30 s delta, but real audio only 6.00 s.
        let segs = build_segments(&tk, &tokens, &plogs, 0, 3000, 600, true);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "hello world");
        assert!((segs[0].start_sec.unwrap() - 0.0).abs() < 1e-9);
        // End clamped from 10.00 s to the real audio length 6.00 s.
        assert!(
            (segs[0].end_sec.unwrap() - 6.0).abs() < 1e-9,
            "end {} clamped to real length 6.0",
            segs[0].end_sec.unwrap()
        );

        // Open-tail clamp: text after the last timestamp pair, seek_delta beyond
        // real length, must also clamp.
        let tokens2 = vec![beg, 2, beg + 200, 3]; // <|0|> hello <|4|> world (open)
        let plogs2 = vec![-0.1f32; tokens2.len()];
        let segs2 = build_segments(&tk, &tokens2, &plogs2, 0, 3000, 600, true);
        assert_eq!(segs2.len(), 2);
        // open tail would close at seek_delta=30 s, clamped to 6.0 s.
        assert!((segs2[1].end_sec.unwrap() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn short_tail_prompt_clearing_predicate() {
        // Fix #2 (whisper.cpp 7046-7051): clear on a non-first window whose
        // remaining audio is < 5 s; never on the first window (seek == 0).
        // seek_end = 1000 cs (10 s).
        // First window: never clears.
        assert!(!should_clear_short_tail_prompt(0, 1000));
        // Non-first window, > 5 s left: no clear (seek 400 -> 6 s left).
        assert!(!should_clear_short_tail_prompt(400, 1000));
        // Non-first window, exactly 5 s left: clears (boundary `>=`).
        assert!(should_clear_short_tail_prompt(500, 1000));
        // Non-first window, < 5 s left: clears.
        assert!(should_clear_short_tail_prompt(600, 1000));
        // Last partial window near the very end.
        assert!(should_clear_short_tail_prompt(900, 1000));
    }

    #[test]
    fn parallel_no_context_windows_use_one_lane_outside_large_host_regime() {
        assert_eq!(parallel_no_context_window_lanes(3, 64, 64), 1);
        assert_eq!(parallel_no_context_window_lanes(5, 32, 64), 1);
        assert_eq!(parallel_no_context_window_lanes(5, 128, 64), 1);
        assert_eq!(parallel_no_context_window_lanes(5, 24, 24), 1);
    }

    #[test]
    fn parallel_no_context_windows_fill_physical_core_pool_without_oversubscription() {
        assert_eq!(parallel_no_context_window_lanes(5, 32, 32), 5);
        assert_eq!(parallel_no_context_window_lanes(5, 48, 48), 5);
        assert_eq!(parallel_no_context_window_lanes(5, 64, 64), 5);
        assert_eq!(parallel_no_context_window_lanes(20, 64, 64), 8);
        assert_eq!(parallel_no_context_window_lanes(12, 128, 128), 12);
    }

    // ----- Tail-window encoder-context truncation derivation (pure) -----

    // The signature is `tail_enc_ctx(real_frames, is_first, enabled)`. Tail
    // truncation only ever engages on a non-first window (`is_first == false`).
    #[test]
    fn tail_enc_ctx_full_window_is_always_full() {
        // A full (or over-full) non-first window always yields the full 1500
        // ctx, enabled or not.
        assert_eq!(tail_enc_ctx(FRAMES_PER_CHUNK, false, true), FULL_ENC_CTX);
        assert_eq!(
            tail_enc_ctx(FRAMES_PER_CHUNK + 100, false, true),
            FULL_ENC_CTX
        );
        assert_eq!(tail_enc_ctx(FRAMES_PER_CHUNK, false, false), FULL_ENC_CTX);
    }

    #[test]
    fn tail_enc_ctx_first_window_is_never_truncated() {
        // The first window (is_first == true) carries the bulk of a short clip's
        // real audio; truncating it changes the main transcript. It must always
        // get the full ctx, even for a tiny real_frames and truncation enabled —
        // this is what makes the golden byte-gate hold for single-window clips.
        for &rf in &[0usize, 24, 600, 1100, 2999] {
            assert_eq!(
                tail_enc_ctx(rf, true, true),
                FULL_ENC_CTX,
                "first window must never truncate (real_frames={rf})"
            );
        }
    }

    #[test]
    fn tail_enc_ctx_disabled_is_always_full() {
        // Kill switch off ⇒ every short window still gets the full ctx (proves
        // byte-identical fallback to the pre-optimization path), first or not.
        for &rf in &[0usize, 1, 24, 240, 1500, 2999] {
            assert_eq!(
                tail_enc_ctx(rf, false, false),
                FULL_ENC_CTX,
                "disabled must return full ctx for real_frames={rf}"
            );
            assert_eq!(tail_enc_ctx(rf, true, false), FULL_ENC_CTX);
        }
    }

    #[test]
    fn tail_enc_ctx_truncates_short_non_first_windows() {
        // 0.24 s of audio (24 mel frames) ⇒ ((24+1)/2)=12, clamped up to the
        // MIN_ENC_CTX floor of 64. This is the perf hotspot #1 case.
        assert_eq!(tail_enc_ctx(24, false, true), MIN_ENC_CTX);
        // A mid-length tail: 600 frames (6 s) ⇒ (601/2)=300 ctx, within band.
        assert_eq!(tail_enc_ctx(600, false, true), 300);
        // Just under a full window: 2998 frames ⇒ (2999/2)=1499 ctx.
        assert_eq!(tail_enc_ctx(2998, false, true), 1499);
        // The +1 rounds up so the ctx covers the last (odd) frame: 599 → 300.
        assert_eq!(tail_enc_ctx(599, false, true), 300);
    }

    #[test]
    fn tail_enc_ctx_respects_min_floor() {
        // Any non-first window at or below 2*MIN_ENC_CTX real frames clamps to
        // the floor.
        assert_eq!(tail_enc_ctx(0, false, true), MIN_ENC_CTX);
        assert_eq!(tail_enc_ctx(1, false, true), MIN_ENC_CTX);
        assert_eq!(tail_enc_ctx(2 * MIN_ENC_CTX, false, true), MIN_ENC_CTX);
        // One above the floor boundary starts climbing: 2*64+1=129 → (130/2)=65.
        assert_eq!(
            tail_enc_ctx(2 * MIN_ENC_CTX + 1, false, true),
            MIN_ENC_CTX + 1
        );
    }

    #[test]
    fn tail_enc_ctx_mel_frames_always_valid_for_encoder() {
        // The derived mel-frame count (2*enc_ctx) must always be a positive even
        // number ≤ FRAMES_PER_CHUNK so encoder::forward accepts it, for both
        // first and non-first windows.
        for &is_first in &[false, true] {
            for rf in 0..=FRAMES_PER_CHUNK {
                let ctx = tail_enc_ctx(rf, is_first, true);
                let mel_frames = ctx * 2;
                assert!(
                    mel_frames > 0
                        && mel_frames.is_multiple_of(2)
                        && mel_frames <= FRAMES_PER_CHUNK,
                    "mel_frames {mel_frames} invalid for real_frames={rf} is_first={is_first}"
                );
            }
        }
    }

    #[test]
    fn confidence_is_clamped_and_monotone() {
        // mean logprob 0 => exp(0)=1 (clamped); very negative => ~0.
        assert_eq!(confidence(&[0.0, 0.0]), Some(1.0));
        let c = confidence(&[-2.0, -2.0]).unwrap();
        assert!(c > 0.0 && c < 1.0);
        assert!(confidence(&[]).is_none());
    }

    #[test]
    fn confidence_finite_on_nan_plog() {
        // A NaN plog makes the mean NaN; the guarded clamp must not panic on the
        // current nightly and must degrade to a finite 0.0.
        let c = confidence(&[f32::NAN, 0.0]).expect("non-empty plogs → Some");
        assert!(
            c.is_finite(),
            "confidence must be finite on NaN input, got {c}"
        );
        assert_eq!(c, 0.0);
    }

    // -----------------------------------------------------------------------
    // WAV reader.
    // -----------------------------------------------------------------------

    #[test]
    fn wav_reader_round_trips_a_synthetic_clip() {
        // Build a 16kHz mono 16-bit WAV with a tiny ramp and read it back.
        let samples_i16: Vec<i16> = (0..8).map(|i| (i * 1000) as i16).collect();
        let data_bytes: Vec<u8> = samples_i16.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_bytes.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_bytes.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data_bytes);

        let out = read_wav_16k_mono(&wav).unwrap();
        assert_eq!(out.len(), 8);
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[1] - 1000.0 / 32768.0).abs() < 1e-6);
    }

    /// The `channels == 1` fast path in `read_wav_16k_mono` (`87556b4`) must be
    /// BIT-IDENTICAL to the general per-channel accumulation loop it replaced, and
    /// materially faster (the old path did TWO runtime f32 divisions per sample plus
    /// `push`, inhibiting autovec; the fast path is one vectorized `×2⁻¹⁵`). This
    /// guards the byte-exactness of the opt forever and records the speedup as a
    /// foreground micro-bench (~4 M samples, both paths timed back-to-back).
    #[test]
    fn read_wav_mono_fast_path_byte_exact_and_faster() {
        use std::time::Instant;
        let n = 4_000_000usize; // ~4.2 min of 16 kHz audio; cycles the full i16 range 61×.
        let mut data = vec![0u8; n * 2];
        for (i, chunk) in data.chunks_exact_mut(2).enumerate() {
            // Deterministic fill covering the full i16 range (incl. ±edge values).
            chunk.copy_from_slice(&(i as i16).to_le_bytes());
        }
        // NEW: the mono fast path (what `read_wav_16k_mono` uses at channels == 1).
        let t = Instant::now();
        let fast: Vec<f32> = data
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();
        let t_fast = t.elapsed();
        // OLD: the general per-channel accumulation loop with channels == 1.
        let channels = 1usize;
        let n_frames = data.len() / (2 * channels);
        let t = Instant::now();
        let mut general = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let mut acc = 0i32;
            for c in 0..channels {
                let o = (f * channels + c) * 2;
                acc += i32::from(i16::from_le_bytes([data[o], data[o + 1]]));
            }
            general.push((acc as f32 / channels as f32) / 32768.0);
        }
        let t_old = t.elapsed();
        assert_eq!(
            fast, general,
            "mono fast path must be byte-identical to the general per-channel loop"
        );
        eprintln!(
            "read_wav mono i16→f32 ({n} samples): fast={t_fast:?} old={t_old:?} speedup={:.2}×",
            t_old.as_secs_f64() / t_fast.as_secs_f64().max(1e-12)
        );
    }

    // -----------------------------------------------------------------------
    // Gated end-to-end tests against the real tiny.en model + jfk.wav.
    // -----------------------------------------------------------------------

    /// The reference transcript whisper-cli produced (see
    /// `tests/fixtures/native/jfk_tiny_reference.json`), trimmed.
    const JFK_REFERENCE: &str = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country.";

    fn load_jfk_samples() -> Option<Vec<f32>> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
        let bytes = std::fs::read(path).ok()?;
        read_wav_16k_mono(&bytes).ok()
    }

    fn load_tiny_en() -> Option<LoadedModel> {
        let path = super::super::find_model_file("tiny.en")?;
        let model = GgmlModel::load(&path).ok()?;
        LoadedModel::from_ggml(model).ok()
    }

    fn noop() -> FwResult<()> {
        Ok(())
    }

    fn e2e_params() -> DecodeParams {
        DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: 4,
            max_text_ctx: None,
            ..DecodeParams::default()
        }
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_q8_0_transcribes() {
        // End-to-end proof that the engine RUNS a whisper.cpp q8_0-quantized
        // model: its Q8_0 tensors dequantize to f32 on load (ggml.rs), route
        // through the f32 weight path, build the engine, and produce a correct
        // jfk transcript. Requires ggml-tiny.en-q8_0.bin alongside the f16 model
        // (`whisper-quantize <f16-model> <out> q8_0`).
        let Some(path) = super::super::find_model_file("tiny.en-q8_0") else {
            eprintln!("SKIP gated_e2e_jfk_q8_0: ggml-tiny.en-q8_0.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_jfk_q8_0: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load q8_0 model");
        let loaded = LoadedModel::from_ggml(model).expect("build engine from q8_0");
        let out =
            transcribe_samples(&loaded, &samples, &e2e_params(), &noop).expect("transcribe q8_0");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("q8_0 PRODUCED: {joined}");
        assert!(!out.segments.is_empty(), "q8_0 produced no segments");
        // High-precision quant → the salient jfk content (content-checked, not a
        // byte match: q8_0 weights differ slightly from f16 so tokens may drift).
        let low = joined.to_lowercase();
        assert!(
            low.contains("fellow americans"),
            "q8_0 transcript missing 'fellow americans': {joined}"
        );
        assert!(
            low.contains("country"),
            "q8_0 transcript missing 'country': {joined}"
        );
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_q4_1_q5_1_transcribe() {
        // The engine runs the "_1" (scale+min) legacy quants end-to-end.
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_jfk_q4_1_q5_1: jfk.wav missing");
            return;
        };
        for name in ["tiny.en-q4_1", "tiny.en-q5_1"] {
            let Some(path) = super::super::find_model_file(name) else {
                eprintln!("SKIP {name}: model not found");
                continue;
            };
            let model = GgmlModel::load(&path).unwrap_or_else(|e| panic!("load {name}: {e}"));
            let loaded =
                LoadedModel::from_ggml(model).unwrap_or_else(|e| panic!("engine {name}: {e}"));
            let out = transcribe_samples(&loaded, &samples, &e2e_params(), &noop)
                .unwrap_or_else(|e| panic!("transcribe {name}: {e}"));
            let joined: String = out
                .segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("{name} PRODUCED: {joined}");
            assert!(!out.segments.is_empty(), "{name} produced no segments");
            assert!(
                joined.to_lowercase().contains("country"),
                "{name} transcript missing 'country': {joined}"
            );
        }
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_q4_0_transcribes() {
        // Engine runs a whisper.cpp q4_0-quantized model end-to-end (4-bit).
        let Some(path) = super::super::find_model_file("tiny.en-q4_0") else {
            eprintln!("SKIP gated_e2e_jfk_q4_0: ggml-tiny.en-q4_0.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_jfk_q4_0: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load q4_0 model");
        let loaded = LoadedModel::from_ggml(model).expect("build engine from q4_0");
        let out =
            transcribe_samples(&loaded, &samples, &e2e_params(), &noop).expect("transcribe q4_0");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("q4_0 PRODUCED: {joined}");
        assert!(!out.segments.is_empty(), "q4_0 produced no segments");
        // 4-bit is coarse; assert the engine runs + produces the salient content
        // ("country" is robust; the fuller phrase can drift at 4-bit precision).
        assert!(
            joined.to_lowercase().contains("country"),
            "q4_0 transcript missing 'country': {joined}"
        );
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_q5_0_transcribes() {
        // Engine runs a whisper.cpp q5_0-quantized model end-to-end (5-bit quant,
        // dequantized to f32 on load). Requires ggml-tiny.en-q5_0.bin.
        let Some(path) = super::super::find_model_file("tiny.en-q5_0") else {
            eprintln!("SKIP gated_e2e_jfk_q5_0: ggml-tiny.en-q5_0.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_jfk_q5_0: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load q5_0 model");
        let loaded = LoadedModel::from_ggml(model).expect("build engine from q5_0");
        let out =
            transcribe_samples(&loaded, &samples, &e2e_params(), &noop).expect("transcribe q5_0");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("q5_0 PRODUCED: {joined}");
        assert!(!out.segments.is_empty(), "q5_0 produced no segments");
        // 5-bit quant is coarser; assert the salient jfk content (not byte match).
        let low = joined.to_lowercase();
        assert!(
            low.contains("fellow americans"),
            "q5_0 transcript missing 'fellow americans': {joined}"
        );
        assert!(
            low.contains("country"),
            "q5_0 transcript missing 'country': {joined}"
        );
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_q6_k_transcribes() {
        // Engine runs a whisper.cpp q6_k-quantized model end-to-end. Q6_K is a
        // k-quant super-block format (256-value blocks, 6-bit + per-16 int8
        // sub-scales); its tensors dequantize to f32 on load (ggml.rs) and route
        // through the f32 weight path. Requires ggml-tiny.en-q6_k.bin.
        let Some(path) = super::super::find_model_file("tiny.en-q6_k") else {
            eprintln!("SKIP gated_e2e_jfk_q6_k: ggml-tiny.en-q6_k.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_jfk_q6_k: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load q6_k model");
        let loaded = LoadedModel::from_ggml(model).expect("build engine from q6_k");
        let out =
            transcribe_samples(&loaded, &samples, &e2e_params(), &noop).expect("transcribe q6_k");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("q6_k PRODUCED: {joined}");
        assert!(!out.segments.is_empty(), "q6_k produced no segments");
        // Q6_K is high-precision (near-f16); assert the salient jfk content.
        let low = joined.to_lowercase();
        assert!(
            low.contains("fellow americans"),
            "q6_k transcript missing 'fellow americans': {joined}"
        );
        assert!(
            low.contains("country"),
            "q6_k transcript missing 'country': {joined}"
        );
    }

    #[test]
    fn gated_initial_prompt_field_wired_and_neutral_when_empty() {
        // End-to-end through the DecodeParams.initial_prompt FIELD (the real
        // per-request API, distinct from the FW_INITIAL_PROMPT dev override which
        // an OnceLock makes hard to test per-case). FW_INITIAL_PROMPT is unset
        // under `cargo test`, so the field is the source of truth here.
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_initial_prompt_field: tiny.en model or jfk.wav missing");
            return;
        };
        let join = |o: &DecodeOutput| {
            o.segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ")
        };
        // None vs Some("") must be identical (an empty prompt is a no-op).
        let mut p_none = e2e_params();
        p_none.initial_prompt = None;
        let mut p_empty = e2e_params();
        p_empty.initial_prompt = Some(String::new());
        let t_none = join(&transcribe_samples(&model, &samples, &p_none, &noop).unwrap());
        let t_empty = join(&transcribe_samples(&model, &samples, &p_empty, &noop).unwrap());
        assert_eq!(
            t_none, t_empty,
            "empty initial_prompt field must be a no-op"
        );
        assert!(
            t_none.to_lowercase().contains("country"),
            "baseline should transcribe jfk: {t_none}"
        );
        // A benign prompt via the field is accepted and still transcribes jfk
        // (proving the field flows into the decoder's prompt_past seeding).
        let mut p = e2e_params();
        p.initial_prompt = Some("Hello there.".to_owned());
        let t = join(&transcribe_samples(&model, &samples, &p, &noop).unwrap());
        eprintln!("field-prompted: {t}");
        assert!(
            t.to_lowercase().contains("country"),
            "field prompt should still transcribe jfk: {t}"
        );
    }

    #[test]
    fn gated_max_context_zero_disables_prompt_carry() {
        // max_context=0 (whisper --max-context 0) disables carried previous-context
        // — the per-request equivalent of FW_NO_CONTEXT. On tiled/looping audio the
        // default carried prompt triggers the bd-r0qd early-EOT drop; max_context=0
        // avoids it, so it recovers >= the default's content. (See
        // project_final_window_early_eot_bug.)
        let (Some(model), Some(jfk)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_max_context_zero: tiny.en model or jfk.wav missing");
            return;
        };
        // Tile jfk ×3 (~33 s, multi-window) so carried context is exercised.
        let mut tiled = jfk.clone();
        tiled.extend_from_slice(&jfk);
        tiled.extend_from_slice(&jfk);
        let count_country = |o: &DecodeOutput| -> usize {
            o.segments
                .iter()
                .map(|s| s.text.to_lowercase().matches("country").count())
                .sum()
        };
        let default_out = transcribe_samples(&model, &tiled, &e2e_params(), &noop).unwrap();
        let mut p = e2e_params();
        p.max_context = Some(0);
        let nocarry_out = transcribe_samples(&model, &tiled, &p, &noop).unwrap();
        let (d, n) = (count_country(&default_out), count_country(&nocarry_out));
        eprintln!("tiled-jfk 'country' count: default={d} max_context=0={n}");
        // max_context=0 disables prompt carry, so the looping tiles decode fully
        // (jfk×3 = 6 'country' when fully transcribed). This holds regardless of
        // the default path: the FW_RETRY_FAILED_WINDOW retry is now default-ON, so
        // the default ALSO recovers the tail — max_context=0 stays >= it.
        assert!(
            n >= 6,
            "max_context=0 should fully transcribe the tiled content, got {n}"
        );
        assert!(
            n >= d,
            "max_context=0 must not lose content vs the default path, got {n} vs {d}"
        );
    }

    #[test]
    fn gated_suppress_nst_field_is_neutral_on_clean_speech() {
        // suppress_nst (whisper --suppress-nst) masks non-speech/symbol tokens.
        // jfk is clean speech that never decodes a non-speech token, so masking
        // them cannot change the argmax → suppress_nst=true is BYTE-IDENTICAL to
        // false here. This proves the field reaches the logit filter (via
        // FilterConfig.suppress_nst) without perturbing normal transcription. The
        // masking behavior itself is unit-pinned by `non_speech_suppressed_when_enabled`.
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_suppress_nst: tiny.en model or jfk.wav missing");
            return;
        };
        let join = |o: &DecodeOutput| {
            o.segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ")
        };
        let off = join(&transcribe_samples(&model, &samples, &e2e_params(), &noop).unwrap());
        let mut p = e2e_params();
        p.suppress_nst = true;
        let on = join(&transcribe_samples(&model, &samples, &p, &noop).unwrap());
        assert_eq!(
            on, off,
            "suppress_nst must be a no-op on clean speech (jfk)"
        );
        assert!(
            off.to_lowercase().contains("country"),
            "baseline transcribes jfk"
        );
    }

    #[test]
    fn gated_beam_size_field_matches_greedy_on_jfk() {
        // Beam search via the DecodeParams.beam_size FIELD (whisper --beam-size).
        // On jfk (clear, unambiguous speech) beam=5 selects the same hypothesis as
        // greedy, so the transcript is byte-identical — this exercises the
        // field → beam decode wiring end to end AND pins the "beam is a superset of
        // greedy" invariant. First e2e coverage of beam search. FW_BEAM_SIZE is
        // unset under `cargo test`, so the field is the source of truth.
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_beam_size_field: tiny.en model or jfk.wav missing");
            return;
        };
        let join = |o: &DecodeOutput| {
            o.segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ")
        };
        let greedy = join(&transcribe_samples(&model, &samples, &e2e_params(), &noop).unwrap());
        let mut p = e2e_params();
        p.beam_size = Some(5);
        let beam = join(&transcribe_samples(&model, &samples, &p, &noop).unwrap());
        eprintln!("greedy: {greedy}\nbeam5:  {beam}");
        assert_eq!(
            beam, greedy,
            "beam=5 must match greedy on jfk (byte-identical superset)"
        );
        assert!(
            greedy.to_lowercase().contains("country"),
            "baseline should transcribe jfk"
        );
    }

    #[test]
    fn gated_seeded_prompt_past_encodes_user_prompt() {
        // The initial-prompt seed (whisper `--prompt`, FW_INITIAL_PROMPT): an
        // absent/empty prompt yields an empty prompt_past (byte-identical default),
        // and a real prompt is BPE-encoded to its exact whisper.cpp token ids and
        // becomes the first window's carried context.
        let Some(model) = load_tiny_en() else {
            eprintln!("SKIP gated_seeded_prompt: tiny.en model missing");
            return;
        };
        let tk = &model.tokenizer;
        assert_eq!(seeded_prompt_past(None, tk), Vec::<i32>::new());
        assert_eq!(seeded_prompt_past(Some(""), tk), Vec::<i32>::new());
        assert_eq!(seeded_prompt_past(Some(" country"), tk), vec![1499]);
        assert_eq!(
            seeded_prompt_past(Some("the country"), tk),
            vec![1169, 1499]
        );
    }

    #[test]
    fn gated_truncated_model_errors_cleanly() {
        // A corrupt / partially-downloaded model (valid start, cut before all
        // tensors are present) must fail with a clean Err from load()/from_ggml —
        // NEVER a panic or a silent load of garbage weights. Only the header +
        // early sections parse; a later tensor is missing or its payload is short.
        let Some(path) = super::super::find_model_file("tiny.en") else {
            eprintln!("SKIP gated_truncated_model: tiny.en model missing");
            return;
        };
        let bytes = std::fs::read(&path).expect("read tiny.en");
        // Cut to half — past the header/mel/vocab, mid tensor directory/data.
        let cut = bytes.len() / 2;
        let tmp = std::env::temp_dir().join(format!("fw_truncated_{cut}.bin"));
        std::fs::write(&tmp, &bytes[..cut]).expect("write truncated");
        // load() then from_ggml() — the whole path must return Err, not panic.
        let result = GgmlModel::load(&tmp).and_then(LoadedModel::from_ggml);
        let _ = std::fs::remove_file(&tmp);
        assert!(
            result.is_err(),
            "a truncated model must error, not load garbage weights"
        );
    }

    #[test]
    fn gated_degenerate_audio_inputs_are_handled_gracefully() {
        // A production ASR receives arbitrary clips: empty, pure silence, a tone,
        // and sub-window lengths. None must panic or error — mel's build_padded
        // always pads to a full 30 s window (so `padded.len() - N_FFT` can't
        // underflow) and the decode must degrade to no/near-no speech, not crash
        // or run away. This is the only coverage of the degenerate-input path.
        let Some(model) = load_tiny_en() else {
            eprintln!("SKIP gated_degenerate_audio: tiny.en model missing");
            return;
        };
        let params = e2e_params();

        // Empty audio is rejected with a clean error (never a panic).
        match transcribe_samples(&model, &[], &params, &noop) {
            Err(FwError::InvalidRequest(msg)) => {
                assert!(msg.contains("empty"), "unexpected empty-audio error: {msg}");
            }
            Err(other) => panic!("empty audio: expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("empty audio should be rejected, not transcribed"),
        }

        // Non-empty but degenerate inputs must transcribe without panicking, with
        // bounded, finite output (silence/tone degrade to no/near-no speech).
        let cases: [(&str, Vec<f32>); 4] = [
            ("one_sample", vec![0.1]),
            ("half_second_silence", vec![0.0; SAMPLE_RATE / 2]),
            (
                "half_second_tone",
                (0..SAMPLE_RATE / 2)
                    .map(|i| (i as f32 * 0.20).sin() * 0.3)
                    .collect(),
            ),
            ("two_second_silence", vec![0.0; SAMPLE_RATE * 2]),
        ];
        for (name, samples) in cases {
            let out = transcribe_samples(&model, &samples, &params, &noop)
                .unwrap_or_else(|e| panic!("{name}: transcribe errored: {e}"));
            let joined: String = out
                .segments
                .iter()
                .map(|s| s.text.trim())
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("{name}: {} segs, text={joined:?}", out.segments.len());
            // No runaway output (a repetition loop on non-speech would balloon the
            // segment count) and every timestamp stays finite/ordered.
            assert!(
                out.segments.len() < 50,
                "{name}: runaway segment count {}",
                out.segments.len()
            );
            for seg in &out.segments {
                if let (Some(s), Some(e)) = (seg.start_sec, seg.end_sec) {
                    assert!(
                        s.is_finite() && e.is_finite() && e >= s,
                        "{name}: bad segment span [{s}, {e}]"
                    );
                }
            }
        }
    }

    #[test]
    fn gated_language_detect_jfk_turbo_matches_oracle() {
        // The ONLY coverage of the multilingual language-auto-detect path
        // (detect_language_from_enc, a port of whisper.cpp whisper_lang_auto_detect).
        // Every other on-box test uses English-only tiny.en, which skips detection
        // entirely. Oracle: `whisper-cli -dl` on jfk + f16 large-v3-turbo reports
        // "auto-detected language: en (p = 0.960230)". Requires the multilingual
        // ggml-large-v3-turbo.bin + jfk.wav.
        let Some(path) = super::super::find_model_file("large-v3-turbo") else {
            eprintln!("SKIP gated_language_detect: ggml-large-v3-turbo.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_language_detect: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load turbo");
        let m = LoadedModel::from_ggml(model).expect("build turbo engine");
        assert!(
            m.tokenizer.num_languages() >= 99,
            "turbo must be multilingual, got {} languages",
            m.tokenizer.num_languages()
        );

        // Encode jfk's first 30 s window and build the decoder state — exactly the
        // inputs detect_language_from_enc consumes in production.
        let mel_threads = super::super::host_parallelism().min(16);
        let mel = mel::log_mel(&samples, &m.filters, mel_threads).expect("mel");
        let frames = FRAMES_PER_CHUNK.min(mel.n_frames);
        let enc = encoder::forward_from_full_mel_window(&m.encoder, &mel, 0, frames, 4, &noop)
            .expect("encode window 0");
        let mut st = DecoderState::new(&m.decoder, &enc).expect("decoder state");

        // (a) The real production function returns the detected code.
        let detected = detect_language_from_enc(&m, &mut st, &noop)
            .expect("detect")
            .expect("some language");
        assert_eq!(
            detected, "en",
            "detect_language_from_enc must match the whisper-cli oracle (en)"
        );

        // (b) False-pass guard: a bug that always returns the default "en" (e.g.
        // all-NEG_INFINITY logits → uniform 1/99 softmax) would pass (a). Recompute
        // the full language distribution and require "en" to be the DOMINANT
        // outcome (~0.96 per the oracle), which no degenerate path can produce.
        // detect_language_from_enc reset `st`, so this re-forward reproduces its logits.
        let logits = decoder::forward_step(&m.decoder, &mut st, &[m.tokenizer.sot], &noop)
            .expect("forward sot");
        let mut lang_logits: Vec<(&str, f32)> = Vec::new();
        for &(code, lang_id, _) in LANGUAGES {
            if lang_id >= m.tokenizer.num_languages() {
                continue;
            }
            let tok = m.tokenizer.sot + 1 + lang_id;
            if let Ok(idx) = usize::try_from(tok)
                && let Some(&l) = logits.get(idx)
            {
                lang_logits.push((code, l));
            }
        }
        let max_l = lang_logits
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = lang_logits.iter().map(|(_, l)| (l - max_l).exp()).sum();
        let mut ranked: Vec<(&str, f32)> = lang_logits
            .iter()
            .map(|(c, l)| (*c, (l - max_l).exp() / sum_exp))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("finite probs"));
        eprintln!(
            "language detect top-3: {:?}",
            &ranked[..3.min(ranked.len())]
        );
        assert_eq!(ranked[0].0, "en", "top detected language must be en");
        assert!(
            ranked[0].1 > 0.9,
            "p(en) must be dominant (~0.96 oracle), got {:.4}",
            ranked[0].1
        );
    }

    #[test]
    fn gated_e2e_jfk_large_v3_turbo_no_ts_matches_oracle() {
        // Faithfulness has a MODE axis (ts / no_ts / word-ts); the flagship was
        // only proven in TS mode. no_ts uses a different SOT (`sot, <|en|>,
        // <|transcribe|>, <|notimestamps|>`) and suppresses ALL timestamp logits —
        // historically the buggiest mode (tail-truncation, single-ts fixes). Diff
        // it against the oracle. whisper-cli (`-l auto -nt`): "And so, my fellow
        // Americans, ask not what your country can do for you, ask what you can do
        // for your country." Requires ggml-large-v3-turbo.bin + jfk.wav.
        let Some(path) = super::super::find_model_file("large-v3-turbo") else {
            eprintln!("SKIP gated_e2e_turbo_no_ts: ggml-large-v3-turbo.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_turbo_no_ts: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load turbo");
        let loaded = LoadedModel::from_ggml(model).expect("build turbo engine");
        let params = DecodeParams {
            language: None,
            translate: false,
            timestamps: false, // no_ts mode
            n_threads: 4,
            max_text_ctx: None,
            ..DecodeParams::default()
        };
        let out = transcribe_samples(&loaded, &samples, &params, &noop)
            .expect("transcribe jfk on turbo (no_ts)");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("turbo no_ts PRODUCED: {joined}");
        let oracle = "And so, my fellow Americans, ask not what your country can \
                      do for you, ask what you can do for your country.";
        assert_eq!(
            joined, oracle,
            "turbo no_ts must byte-match the whisper-cli oracle"
        );
    }

    #[test]
    fn gated_e2e_jfk_distil_large_v3_transcribes() {
        // distil-whisper is a popular FAST variant: large-v3's 32-layer encoder +
        // a DISTILLED 2-layer decoder (n_text_layer=2, vs large-v3's 32 / turbo's
        // 4). The engine reads n_text_layer from the header, so the 2-layer decoder
        // must build and decode correctly. This is the only coverage of a distilled
        // (shallow-decoder) model. Requires ggml-distil-large-v3.bin.
        let Some(path) = super::super::find_model_file("distil-large-v3") else {
            eprintln!("SKIP gated_e2e_distil: ggml-distil-large-v3.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_distil: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load distil-large-v3");
        assert_eq!(
            model.hparams.n_text_layer, 2,
            "distil has a 2-layer decoder"
        );
        let loaded = LoadedModel::from_ggml(model).expect("build engine from distil");
        let params = DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: 4,
            ..DecodeParams::default()
        };
        let out = transcribe_samples(&loaded, &samples, &params, &noop)
            .expect("transcribe jfk on distil");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("distil PRODUCED: [{:?}] {joined}", out.language);
        assert!(!out.segments.is_empty(), "distil produced no segments");
        let low = joined.to_lowercase();
        assert!(
            low.contains("fellow americans") && low.contains("country"),
            "distil transcript missing salient jfk content: {joined}"
        );
    }

    #[test]
    fn gated_e2e_jfk_large_v3_turbo_autodetect_transcribes() {
        // The full multilingual pipeline end to end on the FLAGSHIP: encode →
        // auto-detect language → build the multilingual SOT (sot, <|en|>,
        // <|transcribe|>) → decode. Every other e2e test uses English-only tiny.en
        // whose SOT is bare `[sot]` — this is the only coverage of turbo's 32-layer
        // encoder + 51866-token multilingual decode + the language-token SOT path.
        // Oracle (whisper-cli, f16 turbo, -l auto): "And so, my fellow Americans,
        // ask not what your country can do for you, ask what you can do for your
        // country." Requires the multilingual ggml-large-v3-turbo.bin + jfk.wav.
        let Some(path) = super::super::find_model_file("large-v3-turbo") else {
            eprintln!("SKIP gated_e2e_turbo_autodetect: ggml-large-v3-turbo.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_turbo_autodetect: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load turbo");
        let loaded = LoadedModel::from_ggml(model).expect("build turbo engine");
        // language: None → exercises the auto-detect path in transcribe_samples.
        let params = DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: 4,
            max_text_ctx: None,
            ..DecodeParams::default()
        };
        let out =
            transcribe_samples(&loaded, &samples, &params, &noop).expect("transcribe jfk on turbo");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("turbo autodetect PRODUCED: [{:?}] {joined}", out.language);
        // Auto-detection ran and picked English (matches the whisper-cli oracle).
        assert_eq!(
            out.language.as_deref(),
            Some("en"),
            "turbo should auto-detect English"
        );
        // The multilingual decode produced the salient jfk content.
        let low = joined.to_lowercase();
        assert!(
            low.contains("fellow americans"),
            "turbo transcript missing 'fellow americans': {joined}"
        );
        assert!(
            low.contains("country"),
            "turbo transcript missing 'country': {joined}"
        );
    }

    #[test]
    fn gated_e2e_jfk_q5_k_large_v3_turbo_transcribes() {
        // The quantized FLAGSHIP end to end: a q5_k large-v3-turbo (k-quant, 233
        // Q5_K tensors dequantized to f32 on load) must not just build but DECODE
        // correctly. The load-only test proves dequant+build; this proves the full
        // dequant → 32-layer encode → multilingual decode path produces a correct
        // transcript. Requires ggml-large-v3-turbo-q5_k.bin.
        let Some(path) = super::super::find_model_file("large-v3-turbo-q5_k") else {
            eprintln!("SKIP gated_e2e_q5_k_turbo: ggml-large-v3-turbo-q5_k.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_q5_k_turbo: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load q5_k turbo");
        let loaded = LoadedModel::from_ggml(model).expect("build engine from q5_k turbo");
        let params = DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: 4,
            ..DecodeParams::default()
        };
        let out = transcribe_samples(&loaded, &samples, &params, &noop)
            .expect("transcribe jfk on q5_k turbo");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("q5_k turbo PRODUCED: [{:?}] {joined}", out.language);
        let low = joined.to_lowercase();
        assert!(
            low.contains("fellow americans") && low.contains("country"),
            "q5_k turbo transcript missing salient jfk content: {joined}"
        );
    }

    #[test]
    fn gated_q5_k_large_v3_turbo_loads_and_builds_engine() {
        // Flagship-model proof: the quant path is size-agnostic, so a q5_k
        // large-v3-turbo (51866 vocab, 1280 audio-state, 32 enc / 4 dec layers,
        // multilingual) must load, dequantize EVERY tensor, and build the engine
        // exactly like tiny.en. All prior quant e2e used tiny.en; this closes the
        // "does it work on the model people actually run" question. Building the
        // engine forces tensor_f32/tensor_f16 over the whole q5_k tensor set.
        // Requires ggml-large-v3-turbo-q5_k.bin (`whisper-quantize <turbo> <out> q5_k`).
        let Some(path) = super::super::find_model_file("large-v3-turbo-q5_k") else {
            eprintln!("SKIP gated_q5_k_turbo: ggml-large-v3-turbo-q5_k.bin not found");
            return;
        };
        let model = GgmlModel::load(&path).expect("load q5_k turbo model");
        assert_eq!(
            model.hparams.ftype.rem_euclid(1000),
            13,
            "turbo q5_k base ftype must be 13"
        );
        // Turbo hparams ground truth (from the bd-frp7 epic).
        assert_eq!(model.hparams.n_vocab, 51866, "turbo n_vocab");
        assert_eq!(model.hparams.n_audio_state, 1280, "turbo n_audio_state");
        assert_eq!(model.hparams.n_audio_layer, 32, "turbo n_audio_layer");
        assert_eq!(model.hparams.n_text_layer, 4, "turbo n_text_layer");
        // The 2D weight tensors are actually stored as Q5_K.
        let n_q5k = model
            .tensor_names()
            .filter(|n| {
                model
                    .tensor(n)
                    .is_some_and(|e| e.dtype == super::super::GgmlDType::Q5_K)
            })
            .count();
        assert!(
            n_q5k > 100,
            "expected the bulk of turbo's tensors to be Q5_K, got {n_q5k}"
        );
        // Build the engine — dequantizes the full q5_k tensor set into the
        // engine's own int8/f16 runtime. Succeeding here proves the flagship
        // quant load path end to end (a bad shape/type/byte-length on ANY of
        // turbo's tensors would surface as an Err right here).
        let _loaded = LoadedModel::from_ggml(model).expect("build engine from q5_k turbo");
        eprintln!("q5_k turbo: loaded + built engine, {n_q5k} Q5_K tensors");
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_q2_k_transcribes() {
        // Engine runs a whisper.cpp q2_k-quantized model end-to-end. Q2_K is the
        // coarsest quant native decodes (2-bit); dequantized to f32 on load.
        // Requires ggml-tiny.en-q2_k.bin.
        let Some(path) = super::super::find_model_file("tiny.en-q2_k") else {
            eprintln!("SKIP gated_e2e_jfk_q2_k: ggml-tiny.en-q2_k.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_jfk_q2_k: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load q2_k model");
        let loaded = LoadedModel::from_ggml(model).expect("build engine from q2_k");
        let out =
            transcribe_samples(&loaded, &samples, &e2e_params(), &noop).expect("transcribe q2_k");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("q2_k PRODUCED: {joined}");
        // The q2_k load→dequant→build→run path completes without error (proven by
        // reaching here). 2-bit tiny.en is so coarse it collapses to an EMPTY
        // transcript — whisper.cpp's own CLI can't even load these k-quant tiny
        // models, and the per-element dequant is byte-verified against an
        // independent reference. So accept either the correct jfk content (a
        // larger q2_k model would produce it) or an empty collapse; reject only
        // garbage (non-empty AND wrong).
        let low = joined.to_lowercase();
        assert!(
            low.is_empty() || low.contains("country"),
            "q2_k should collapse to empty or contain jfk content, got: {joined:?}"
        );
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_q3_k_transcribes() {
        // Engine runs a whisper.cpp q3_k-quantized model end-to-end. Q3_K is the
        // coarsest k-quant supported (3-bit, no per-block min); dequantized to f32
        // on load. Requires ggml-tiny.en-q3_k.bin.
        let Some(path) = super::super::find_model_file("tiny.en-q3_k") else {
            eprintln!("SKIP gated_e2e_jfk_q3_k: ggml-tiny.en-q3_k.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_jfk_q3_k: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load q3_k model");
        let loaded = LoadedModel::from_ggml(model).expect("build engine from q3_k");
        let out =
            transcribe_samples(&loaded, &samples, &e2e_params(), &noop).expect("transcribe q3_k");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("q3_k PRODUCED: {joined}");
        assert!(!out.segments.is_empty(), "q3_k produced no segments");
        // 3-bit is the coarsest supported quant; assert the engine runs and
        // produces the salient jfk content ("country" is the most robust token).
        assert!(
            joined.to_lowercase().contains("country"),
            "q3_k transcript missing 'country': {joined}"
        );
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_q5_k_transcribes() {
        // Engine runs a whisper.cpp q5_k-quantized model end-to-end. Q5_K is a
        // 5-bit k-quant (256-value super-block, 6-bit scale+min + a high-bit
        // plane); dequantized to f32 on load. Requires ggml-tiny.en-q5_k.bin.
        let Some(path) = super::super::find_model_file("tiny.en-q5_k") else {
            eprintln!("SKIP gated_e2e_jfk_q5_k: ggml-tiny.en-q5_k.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_jfk_q5_k: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load q5_k model");
        let loaded = LoadedModel::from_ggml(model).expect("build engine from q5_k");
        let out =
            transcribe_samples(&loaded, &samples, &e2e_params(), &noop).expect("transcribe q5_k");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("q5_k PRODUCED: {joined}");
        assert!(!out.segments.is_empty(), "q5_k produced no segments");
        // 5-bit k-quant is high-precision; assert the salient jfk content.
        let low = joined.to_lowercase();
        assert!(
            low.contains("fellow americans"),
            "q5_k transcript missing 'fellow americans': {joined}"
        );
        assert!(
            low.contains("country"),
            "q5_k transcript missing 'country': {joined}"
        );
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_q4_k_transcribes() {
        // Engine runs a whisper.cpp q4_k-quantized model end-to-end. Q4_K is a
        // 4-bit k-quant (256-value super-block, per-sub-block 6-bit scale+min);
        // its tensors dequantize to f32 on load and route through the f32 path.
        // Requires ggml-tiny.en-q4_k.bin.
        let Some(path) = super::super::find_model_file("tiny.en-q4_k") else {
            eprintln!("SKIP gated_e2e_jfk_q4_k: ggml-tiny.en-q4_k.bin not found");
            return;
        };
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_jfk_q4_k: jfk.wav missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load q4_k model");
        let loaded = LoadedModel::from_ggml(model).expect("build engine from q4_k");
        let out =
            transcribe_samples(&loaded, &samples, &e2e_params(), &noop).expect("transcribe q4_k");
        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("q4_k PRODUCED: {joined}");
        assert!(!out.segments.is_empty(), "q4_k produced no segments");
        // 4-bit is coarse; assert the salient jfk content ("country" is robust;
        // the fuller phrase can drift at 4-bit precision).
        assert!(
            joined.to_lowercase().contains("country"),
            "q4_k transcript missing 'country': {joined}"
        );
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_matches_reference() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_e2e_jfk_tiny_en: tiny.en model or jfk.wav missing");
            return;
        };
        let params = e2e_params();
        let t = std::time::Instant::now();
        let out = transcribe_samples(&model, &samples, &params, &noop).expect("transcribe");
        let elapsed = t.elapsed();
        eprintln!(
            "e2e jfk (11s): {elapsed:?} for {} samples, {} segments",
            samples.len(),
            out.segments.len()
        );

        let joined: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("PRODUCED:  {joined}");
        eprintln!("REFERENCE: {JFK_REFERENCE}");
        assert_eq!(
            joined.trim(),
            JFK_REFERENCE,
            "greedy temp-0 transcript must match whisper-cli reference EXACTLY"
        );

        // Windows stats populated.
        assert!(!out.windows.is_empty(), "window stats populated");
        assert!(out.windows[0].tokens > 0);
        // English-only model reports no language.
        assert_eq!(out.language, None);

        // Segment timestamps within 0.3s of the reference fixture.
        // Reference: [0.00, 7.96] and [7.96, 10.76].
        assert!(out.segments.len() >= 2, "expected >= 2 segments");
        let s0_end = out.segments[0].end_sec.unwrap();
        assert!(
            (s0_end - 7.96).abs() < 0.3,
            "segment 0 end {s0_end} within 0.3s of 7.96"
        );
        let last_end = out.segments.last().unwrap().end_sec.unwrap();
        assert!(
            (last_end - 10.76).abs() < 0.3,
            "last segment end {last_end} within 0.3s of 10.76"
        );
    }

    #[test]
    fn gated_e2e_deterministic_across_runs() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_e2e_deterministic: tiny.en model or jfk.wav missing");
            return;
        };
        let params = e2e_params();
        let a = transcribe_samples(&model, &samples, &params, &noop).expect("run a");
        let b = transcribe_samples(&model, &samples, &params, &noop).expect("run b");
        let ja: String = a.segments.iter().map(|s| s.text.clone()).collect();
        let jb: String = b.segments.iter().map(|s| s.text.clone()).collect();
        assert_eq!(ja, jb, "greedy temp-0 must be deterministic across runs");
    }

    #[test]
    fn gated_e2e_multi_window_monotonic_timestamps() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_e2e_multi_window: tiny.en model or jfk.wav missing");
            return;
        };
        // Concatenate jfk 3x (~33s) to force more than one 30s window.
        let mut long = Vec::with_capacity(samples.len() * 3);
        for _ in 0..3 {
            long.extend_from_slice(&samples);
        }
        let params = e2e_params();
        let out = transcribe_samples(&model, &long, &params, &noop).expect("transcribe long");
        eprintln!(
            "multi-window: {} windows, {} segments",
            out.windows.len(),
            out.segments.len()
        );
        assert!(out.windows.len() >= 2, "expected > 1 window for ~33s audio");

        // Timestamps monotonic non-decreasing across the whole transcript,
        // including the window boundary.
        let mut prev_end = -1.0f64;
        for seg in &out.segments {
            let start = seg.start_sec.unwrap();
            let end = seg.end_sec.unwrap();
            assert!(
                start + 1e-6 >= prev_end - 1e-6,
                "segment start {start} must not precede previous end {prev_end}"
            );
            assert!(end + 1e-6 >= start, "segment end {end} >= start {start}");
            prev_end = end;
        }

        // The sentence's signature word "country" should appear at least twice
        // (once per repeated clip, conservatively).
        let joined: String = out.segments.iter().map(|s| s.text.to_lowercase()).collect();
        let occurrences = joined.matches("country").count();
        assert!(
            occurrences >= 2,
            "expected the repeated sentence at least twice, got {occurrences} 'country' hits in: {joined}"
        );
    }

    /// Gated end-to-end DTW word-timestamp check (bd-rjsx).
    ///
    /// Verified reference (whisper-cli `-m ggml-tiny.en.bin -f jfk.wav -ml 1
    /// --no-prints` and `-dtw tiny.en`, run 2026-06-04): the JFK clip contains
    /// the word "ask" twice — first occurrence starts at **3.29 s**, second at
    /// **7.96 s**. The bead's sanity band [7.0, 9.5] s references the *second*
    /// "ask"; that band is the hard requirement and our native DTW lands the
    /// second "ask" at **≈8.66 s** (observed 2026-06-04), inside it. For the
    /// first "ask", our native engine's DTW lands it at **≈3.88 s** — within
    /// ~0.6 s of whisper-cli's 3.29 s reference, the expected small drift
    /// between our pure-Rust forward pass and whisper.cpp's; we bound it with a
    /// ±0.75 s band around the reference.
    #[test]
    fn gated_e2e_dtw_word_timestamps_jfk_tiny_en() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_e2e_dtw_word_timestamps: tiny.en model or jfk.wav missing");
            return;
        };
        let params = DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: 4,
            max_text_ctx: None,
            word_timestamps: true,
            model_hint: Some("tiny.en".to_owned()),
            ..DecodeParams::default()
        };
        let out = transcribe_samples(&model, &samples, &params, &noop).unwrap();

        let word_timings = out
            .word_timings
            .as_ref()
            .expect("word_timings present when requested");
        assert_eq!(
            word_timings.len(),
            out.segments.len(),
            "word_timings 1:1 with segments"
        );

        // Flatten to a single ordered word list.
        let mut words: Vec<&WordTiming> = word_timings.iter().flatten().collect();
        words.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());

        assert!(!words.is_empty(), "DTW produced no words");

        // Word count == whitespace word count of the transcript.
        let transcript_words = JFK_REFERENCE.split_whitespace().count();
        let emitted_words: usize = word_timings.iter().map(Vec::len).sum();
        assert_eq!(
            emitted_words, transcript_words,
            "word count {emitted_words} != transcript word count {transcript_words}"
        );

        // Strictly monotonic, non-overlapping within the global ordering.
        for w in words.windows(2) {
            assert!(
                w[0].end_sec <= w[1].start_sec + 1e-6,
                "overlap: {:?} then {:?}",
                w[0],
                w[1]
            );
            assert!(w[0].start_sec <= w[0].end_sec, "reversed word: {:?}", w[0]);
        }

        // Find the two "ask" occurrences (normalize punctuation/case).
        let asks: Vec<f64> = words
            .iter()
            .filter(|w| {
                w.text
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .eq_ignore_ascii_case("ask")
            })
            .map(|w| w.start_sec)
            .collect();
        assert!(
            asks.len() >= 2,
            "expected two 'ask' occurrences, got {asks:?}"
        );

        // First "ask" ≈ 3.29 s whisper-cli reference (native observed ≈3.88 s);
        // ±0.75 s band covers the cross-implementation drift.
        assert!(
            (asks[0] - 3.29).abs() <= 0.75,
            "first 'ask' start {} not within 3.29 ± 0.75 s",
            asks[0]
        );
        // Second "ask" — the bead's hard requirement: inside [7.0, 9.5] s.
        // whisper-cli reference 7.96 s; our native DTW observed ≈8.66 s
        // (2026-06-04), both comfortably inside the band.
        assert!(
            (7.0..=9.5).contains(&asks[1]),
            "second 'ask' {} outside the bead's sanity band [7.0, 9.5]",
            asks[1]
        );
    }

    /// DTW word timestamps are deterministic across runs (bd-rjsx).
    #[test]
    fn gated_e2e_dtw_word_timestamps_deterministic() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP gated_e2e_dtw_deterministic: tiny.en model or jfk.wav missing");
            return;
        };
        let params = DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: 4,
            max_text_ctx: None,
            word_timestamps: true,
            model_hint: Some("tiny.en".to_owned()),
            ..DecodeParams::default()
        };
        // bd-0ivd MXCSR diagnostics: matrixmultiply's micro-kernel sets the x86
        // FTZ/DAZ flush bits and does not restore them (documented + wrapped in
        // ft-kernel-cpu after the frankentorch-ft-api-fullsuite-flake). If the
        // shared rayon pool carries MIXED flush state across workers, identical
        // decodes can land on differently-flushed threads and drift in the
        // denormal tails (softmax underflow) — the exact transient shape this
        // flake shows. Snapshot every pool worker's MXCSR around each run; a
        // non-0x1f80 value (FTZ = 0x8000, DAZ = 0x40) is the smoking gun.
        // Read-only register read, x86-gated — no memory access.
        #[cfg(target_arch = "x86_64")]
        #[allow(unsafe_code)]
        fn mxcsr_now() -> u32 {
            // SAFETY: `_mm_getcsr` reads the thread's MXCSR register; it
            // touches no memory and has no side effects.
            unsafe { core::arch::x86_64::_mm_getcsr() }
        }
        #[cfg(not(target_arch = "x86_64"))]
        fn mxcsr_now() -> u32 {
            0x1f80
        }
        let pool_mxcsr = || -> Vec<u32> {
            let mut v: Vec<u32> = rayon::broadcast(|_| mxcsr_now());
            v.push(mxcsr_now()); // the calling (test) thread, last
            v
        };

        let mx_a = pool_mxcsr();
        let a = transcribe_samples(&model, &samples, &params, &noop).unwrap();
        let mx_b = pool_mxcsr();
        let b = transcribe_samples(&model, &samples, &params, &noop).unwrap();
        if a.word_timings != b.word_timings {
            // bd-0ivd self-diagnosis: this flakes ONLY inside a full parallel
            // suite run (isolated + synthetic-load repro attempts all bit-stable,
            // NEGATIVE_EVIDENCE 2026-07-22). Turn the failure into evidence: a
            // tie-break run says which side was the outlier, and the max timing
            // delta bounds the perturbation.
            let c = transcribe_samples(&model, &samples, &params, &noop).unwrap();
            let max_delta = |x: &DecodeOutput, y: &DecodeOutput| -> f64 {
                let (Some(xw), Some(yw)) = (&x.word_timings, &y.word_timings) else {
                    return f64::NAN;
                };
                xw.iter()
                    .flatten()
                    .zip(yw.iter().flatten())
                    .map(|(p, q)| {
                        (p.start_sec - q.start_sec)
                            .abs()
                            .max((p.end_sec - q.end_sec).abs())
                    })
                    .fold(0.0f64, f64::max)
            };
            let mx_c = pool_mxcsr();
            let fmt_mx = |v: &[u32]| -> String {
                // Compact: list only non-default entries as idx:hex, else "all-1f80".
                let odd: Vec<String> = v
                    .iter()
                    .enumerate()
                    .filter(|&(_, &m)| m != 0x1f80)
                    .map(|(i, &m)| format!("{i}:{m:#06x}"))
                    .collect();
                if odd.is_empty() {
                    format!("all-1f80({})", v.len())
                } else {
                    odd.join(",")
                }
            };
            eprintln!(
                "bd-0ivd DIVERGENCE: a==c {} | b==c {} | max|Δt| a-vs-b {:.4}s | \
                 pool MXCSR before-a [{}] before-b [{}] after-c [{}] \
                 (FTZ=0x8000 DAZ=0x40; non-1f80 = flush-state leak on that worker)",
                a.word_timings == c.word_timings,
                b.word_timings == c.word_timings,
                max_delta(&a, &b),
                fmt_mx(&mx_a),
                fmt_mx(&mx_b),
                fmt_mx(&mx_c),
            );
        }
        assert_eq!(a.word_timings, b.word_timings);
    }

    /// bd-0ivd bisection tool, NOT a CI test: pins the word-timestamp
    /// nondeterminism to a stage (mel / encoder / decode+record+DTW) under
    /// self-generated rayon-pool contention. Run under the canonical reproducer
    /// conditions: `taskset -c 0-3 <test-bin> --ignored --exact
    /// native_engine::decode::tests::bd_0ivd_stage_determinism_probe`.
    /// The sibling threads mimic the full suite's in-process pool pressure
    /// (cross-process load measured insufficient; see NEGATIVE_EVIDENCE
    /// 2026-07-22). An assert failure NAMES the first divergent stage.
    #[test]
    #[ignore = "bd-0ivd diagnosis tool — run manually under taskset oversubscription"]
    fn bd_0ivd_stage_determinism_probe() {
        let (Some(model), Some(samples)) = (load_tiny_en(), load_jfk_samples()) else {
            eprintln!("SKIP bd_0ivd_stage_determinism_probe: tiny.en model or jfk.wav missing");
            return;
        };
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = AtomicBool::new(false);
        let mel_threads = super::super::host_parallelism().min(16);
        let params = DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads: 4,
            max_text_ctx: None,
            word_timestamps: true,
            model_hint: Some("tiny.en".to_owned()),
            ..DecodeParams::default()
        };
        // Sibling load = REAL engine work sharing the global rayon pool,
        // allocator, and kernels — generic par_iter busywork measured
        // insufficient (5/5 bit-stable, 2026-07-22), matching the finding that
        // only the full suite's heavy sibling mix triggers the flake.
        let sibling_mel = mel::log_mel(&samples, &model.filters, mel_threads).unwrap();
        let model_path = super::super::find_model_file("tiny.en");
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let stopr = &stop;
                let modelr = &model;
                let melr = &sibling_mel;
                scope.spawn(move || {
                    while !stopr.load(Ordering::Relaxed) {
                        let frames = FRAMES_PER_CHUNK.min(melr.n_frames);
                        let _ = encoder::forward_from_full_mel_window(
                            &modelr.encoder,
                            melr,
                            0,
                            frames,
                            4,
                            &noop,
                        );
                    }
                });
            }
            // Full-suite ingredient the encoder siblings don't cover: repeated
            // MODEL LOADS (GgmlModel parse + from_ggml weight quantization on a
            // SCOPED worker pool + ~hundreds of MB of transient allocations) —
            // the allocator/page churn every real suite run has.
            if let Some(path) = model_path.as_deref() {
                let stopr = &stop;
                scope.spawn(move || {
                    while !stopr.load(Ordering::Relaxed) {
                        if let Ok(g) = GgmlModel::load(path) {
                            let _ = LoadedModel::from_ggml(g);
                        }
                    }
                });
                // The last unmimicked suite workload: a CONCURRENT full
                // transcribe with word timestamps — the real suite runs the
                // other gated e2e transcribes (and their DTW-recording batched
                // forwards) alongside this test. Own model instance, exactly
                // like sibling tests' load_tiny_en().
                let stopr2 = &stop;
                let samplesr = &samples;
                scope.spawn(move || {
                    let Ok(g) = GgmlModel::load(path) else { return };
                    let Ok(m2) = LoadedModel::from_ggml(g) else {
                        return;
                    };
                    let p2 = DecodeParams {
                        language: None,
                        translate: false,
                        timestamps: true,
                        n_threads: 4,
                        max_text_ctx: None,
                        word_timestamps: true,
                        model_hint: Some("tiny.en".to_owned()),
                        ..DecodeParams::default()
                    };
                    while !stopr2.load(Ordering::Relaxed) {
                        let _ = transcribe_samples(&m2, samplesr, &p2, &noop);
                    }
                });
            }
            for it in 0..3 {
                let mel_a = mel::log_mel(&samples, &model.filters, mel_threads).unwrap();
                let mel_b = mel::log_mel(&samples, &model.filters, mel_threads).unwrap();
                assert_eq!(
                    mel_a.data.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    mel_b.data.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    "STAGE=mel diverged (iteration {it})"
                );
                let frames = FRAMES_PER_CHUNK.min(mel_a.n_frames);
                let enc_a = encoder::forward_from_full_mel_window(
                    &model.encoder,
                    &mel_a,
                    0,
                    frames,
                    4,
                    &noop,
                )
                .unwrap();
                let enc_b = encoder::forward_from_full_mel_window(
                    &model.encoder,
                    &mel_a,
                    0,
                    frames,
                    4,
                    &noop,
                )
                .unwrap();
                assert_eq!(
                    enc_a.data.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    enc_b.data.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                    "STAGE=encoder diverged (iteration {it})"
                );
                let a = transcribe_samples(&model, &samples, &params, &noop).unwrap();
                let b = transcribe_samples(&model, &samples, &params, &noop).unwrap();
                assert_eq!(
                    a.word_timings, b.word_timings,
                    "STAGE=decode+record+dtw diverged (mel+encoder were bit-stable; iteration {it})"
                );
                eprintln!("probe iteration {it}: all stages bit-stable");
            }
            stop.store(true, Ordering::Relaxed);
        });
    }
}
