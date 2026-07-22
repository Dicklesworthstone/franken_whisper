//! Word-level timestamps via cross-attention DTW (bd-rjsx).
//!
//! This module turns the decoder's recorded cross-attention weights into real
//! per-token start times by aligning the (negated, normalized, median-filtered)
//! attention matrix against the audio-frame axis with classic monotonic dynamic
//! time warping. It is a faithful port of whisper.cpp's
//! `whisper_exp_compute_token_level_timestamps_dtw` (whisper.cpp 8837-8990) and
//! `dtw_and_backtrace` (whisper.cpp 8712-8796), which in turn follow OpenAI
//! whisper's `timing.py` (`find_alignment` / `dtw`).
//!
//! Pipeline (per 30 s window):
//! 1. Select the model's **alignment heads** (whisper.cpp ships per-model head
//!    presets — [`alignment_heads`]; the openai "top half of layers, all heads"
//!    fallback is used for unknown models).
//! 2. Stack the selected heads' `[tokens, frames]` weight matrices, restrict the
//!    frame axis to the window's *actual* (unpadded) audio length, and discard
//!    the padded tail (whisper.cpp `n_audio_tokens = n_frames/2`, 8898).
//! 3. **Normalize** each head over the token axis (subtract mean / divide std —
//!    OpenAI `dim=-2`; whisper.cpp `ggml_norm`, 8929).
//! 4. **Median-filter** each token row over the frame axis (width 7, reflect
//!    padding; whisper.cpp `median_filter`, 8802-8835).
//! 5. **Average** the heads, **negate** to a cost matrix, run **DTW**
//!    ([`dtw_path`]), and read each token's END boundary off the frame at which
//!    the path leaves that token's row, scaled by 0.02 s/frame (encoder frames
//!    are 20 ms; whisper.cpp `time_index * 2` centiseconds, 8975). A token's
//!    *start* is the previous token's end (the first token starts at the window
//!    start), reconciled by the decode caller — see fix #4 in [`token_timestamps`].
//!
//! Token start times are then aggregated into **word** times by the
//! space-prefix convention (a word begins at the first token whose decoded bytes
//! start with a space; punctuation glues to the previous word) — see
//! [`group_tokens_into_words`].
//!
//! ## Memory
//!
//! Recording attention for one 30 s window costs `n_align_heads * n_tokens *
//! 1500` f32 (the encoder produces 1500 frames). For tiny.en that is
//! `8 * ~64 * 1500 * 4 B ≈ 3 MB` — bounded and transient (dropped at window
//! end). Large models with more heads/tokens stay in the low tens of MB.

use crate::native_engine::{Mat, WhisperHParams};

/// Seconds of audio per encoder frame. The encoder downsamples the 10 ms mel
/// frames by 2 (conv stride 2), so 1500 encoder frames span 30 s ⇒ 0.02 s each
/// (whisper.cpp `time_index * 2` centiseconds, 8975).
pub const FRAME_SEC: f32 = 0.02;

/// Default median-filter width (whisper.cpp passes `7`, 7739). Odd, as the
/// algorithm requires.
pub const DEFAULT_MEDFILT_WIDTH: usize = 7;

// ───────────────────────────────────────────────────────────────────────────
// Alignment-head presets (port of whisper.cpp g_aheads_*, 384-409)
// ───────────────────────────────────────────────────────────────────────────

// Each entry is `(text_layer, head)`, copied verbatim from whisper.cpp
// (src/whisper.cpp 384-395). These are the published per-model alignment-head
// lists; the ggml file does not contain them, so they are embedded here.

/// `g_aheads_tiny_en` (whisper.cpp 384).
const AHEADS_TINY_EN: &[(usize, usize)] = &[
    (1, 0),
    (2, 0),
    (2, 5),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
];
/// `g_aheads_tiny` (whisper.cpp 385).
const AHEADS_TINY: &[(usize, usize)] = &[(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)];
/// `g_aheads_base_en` (whisper.cpp 386).
const AHEADS_BASE_EN: &[(usize, usize)] = &[(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)];
/// `g_aheads_base` (whisper.cpp 387).
const AHEADS_BASE: &[(usize, usize)] = &[
    (3, 1),
    (4, 2),
    (4, 3),
    (4, 7),
    (5, 1),
    (5, 2),
    (5, 4),
    (5, 6),
];
/// `g_aheads_small_en` (whisper.cpp 388).
const AHEADS_SMALL_EN: &[(usize, usize)] = &[
    (6, 6),
    (7, 0),
    (7, 3),
    (7, 8),
    (8, 2),
    (8, 5),
    (8, 7),
    (9, 0),
    (9, 4),
    (9, 8),
    (9, 10),
    (10, 0),
    (10, 1),
    (10, 2),
    (10, 3),
    (10, 6),
    (10, 11),
    (11, 2),
    (11, 4),
];
/// `g_aheads_small` (whisper.cpp 389).
const AHEADS_SMALL: &[(usize, usize)] = &[
    (5, 3),
    (5, 9),
    (8, 0),
    (8, 4),
    (8, 7),
    (8, 8),
    (9, 0),
    (9, 7),
    (9, 9),
    (10, 5),
];
/// `g_aheads_medium_en` (whisper.cpp 390).
const AHEADS_MEDIUM_EN: &[(usize, usize)] = &[
    (11, 4),
    (14, 1),
    (14, 12),
    (14, 14),
    (15, 4),
    (16, 0),
    (16, 4),
    (16, 9),
    (17, 12),
    (17, 14),
    (18, 7),
    (18, 10),
    (18, 15),
    (20, 0),
    (20, 3),
    (20, 9),
    (20, 14),
    (21, 12),
];
/// `g_aheads_medium` (whisper.cpp 391).
const AHEADS_MEDIUM: &[(usize, usize)] = &[(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)];
/// `g_aheads_large_v1` (whisper.cpp 392).
const AHEADS_LARGE_V1: &[(usize, usize)] = &[
    (9, 19),
    (11, 2),
    (11, 4),
    (11, 17),
    (22, 7),
    (22, 11),
    (22, 17),
    (23, 2),
    (23, 15),
];
/// `g_aheads_large_v2` (whisper.cpp 393).
const AHEADS_LARGE_V2: &[(usize, usize)] = &[
    (10, 12),
    (13, 17),
    (16, 11),
    (16, 12),
    (16, 13),
    (17, 15),
    (17, 16),
    (18, 4),
    (18, 11),
    (18, 19),
    (19, 11),
    (21, 2),
    (21, 3),
    (22, 3),
    (22, 9),
    (22, 12),
    (23, 5),
    (23, 7),
    (23, 13),
    (25, 5),
    (26, 1),
    (26, 12),
    (27, 15),
];
/// `g_aheads_large_v3` (whisper.cpp 394).
const AHEADS_LARGE_V3: &[(usize, usize)] = &[
    (7, 0),
    (10, 17),
    (12, 18),
    (13, 12),
    (16, 1),
    (17, 14),
    (19, 11),
    (21, 4),
    (24, 1),
    (25, 6),
];
/// `g_aheads_large_v3_turbo` (whisper.cpp 395). v3-turbo has only 4 text layers.
const AHEADS_LARGE_V3_TURBO: &[(usize, usize)] =
    &[(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)];

/// Resolve the alignment-head list `(layer, head)` for a model.
///
/// Selection mirrors whisper.cpp: a model-family preset when one is known,
/// otherwise the OpenAI fallback of **all heads of the top half of decoder
/// layers** (whisper.cpp `WHISPER_AHEADS_N_TOP_MOST` with `dtw_n_top =
/// n_text_layer/2`, 8693-8696).
///
/// The family is inferred from `(n_text_layer, n_text_state)` plus the
/// English-vs-multilingual distinction (`hparams.is_multilingual()`), with an
/// optional `model_hint` (e.g. the short model name `"tiny.en"`) to disambiguate
/// large-v1/v2/v3/v3-turbo, which share `(n_text_layer, n_text_state)` =
/// `(32, 1280)` for v1/v2/v3. v3-turbo is uniquely identified by its 4 text
/// layers, so it needs no hint.
///
/// Heads that fall outside this model's `(n_text_layer, n_text_head)` grid are
/// dropped (defensive — a mismatched preset can never index a head that does not
/// exist).
#[must_use]
pub fn alignment_heads(hparams: &WhisperHParams, model_hint: Option<&str>) -> Vec<(usize, usize)> {
    let n_layer = hparams.n_text_layer.max(0) as usize;
    let n_head = hparams.n_text_head.max(0) as usize;
    let n_state = hparams.n_text_state;
    let multilingual = hparams.is_multilingual();

    let preset = preset_for(n_layer, n_state, multilingual, model_hint);
    let heads: Vec<(usize, usize)> = match preset {
        Some(p) => p.to_vec(),
        None => top_half_fallback(n_layer, n_head),
    };

    // Defensive clamp to the actual grid.
    heads
        .into_iter()
        .filter(|&(l, h)| l < n_layer && h < n_head)
        .collect()
}

/// The OpenAI fallback: all heads of the top half of decoder layers
/// (`il >= n_text_layer - n_text_layer/2`; whisper.cpp 8693-8696).
fn top_half_fallback(n_layer: usize, n_head: usize) -> Vec<(usize, usize)> {
    let n_top = n_layer / 2;
    let start = n_layer.saturating_sub(n_top);
    let mut out = Vec::with_capacity(n_top * n_head);
    for l in start..n_layer {
        for h in 0..n_head {
            out.push((l, h));
        }
    }
    out
}

/// Map `(n_text_layer, n_text_state, multilingual, hint)` to a preset table, or
/// `None` to request the fallback.
fn preset_for(
    n_layer: usize,
    n_state: i32,
    multilingual: bool,
    hint: Option<&str>,
) -> Option<&'static [(usize, usize)]> {
    let hint = hint.map(normalize_hint);
    // A hint that explicitly names a family wins (covers large-v* disambiguation
    // and is robust to unusual hparams).
    if let Some(h) = hint.as_deref()
        && let Some(p) = preset_by_name(h)
    {
        return Some(p);
    }
    match (n_layer, n_state) {
        (4, 384) => Some(if multilingual {
            AHEADS_TINY
        } else {
            AHEADS_TINY_EN
        }),
        (6, 512) => Some(if multilingual {
            AHEADS_BASE
        } else {
            AHEADS_BASE_EN
        }),
        (12, 768) => Some(if multilingual {
            AHEADS_SMALL
        } else {
            AHEADS_SMALL_EN
        }),
        (24, 1024) => Some(if multilingual {
            AHEADS_MEDIUM
        } else {
            AHEADS_MEDIUM_EN
        }),
        // large-v3-turbo: uniquely 4 decoder layers at 1280 state.
        (4, 1280) => Some(AHEADS_LARGE_V3_TURBO),
        // large-v1/v2/v3 all share (32, 1280); without a hint we cannot tell
        // them apart, so default to v3 (the current production large) and let an
        // explicit hint override above.
        (32, 1280) => Some(AHEADS_LARGE_V3),
        _ => None,
    }
}

/// Normalize a model hint to a lowercase comparison key (strip a leading
/// `ggml-` and a trailing `.bin`, and any directory).
fn normalize_hint(hint: &str) -> String {
    let base = hint.rsplit(['/', '\\']).next().unwrap_or(hint);
    let base = base.strip_prefix("ggml-").unwrap_or(base);
    let base = base.strip_suffix(".bin").unwrap_or(base);
    base.to_ascii_lowercase()
}

/// Match a normalized hint to a preset family. Order matters: longer / more
/// specific names are checked first so `large-v3-turbo` does not match
/// `large-v3`.
fn preset_by_name(h: &str) -> Option<&'static [(usize, usize)]> {
    // large-v3-turbo first (superstring of large-v3).
    if h.contains("large-v3-turbo") || h.contains("large-v3.turbo") || h.contains("turbo") {
        return Some(AHEADS_LARGE_V3_TURBO);
    }
    if h.contains("large-v3") {
        return Some(AHEADS_LARGE_V3);
    }
    if h.contains("large-v2") {
        return Some(AHEADS_LARGE_V2);
    }
    if h.contains("large-v1") {
        return Some(AHEADS_LARGE_V1);
    }
    if h.contains("large") {
        // bare "large" historically == v1.
        return Some(AHEADS_LARGE_V1);
    }
    // *.en families before their multilingual counterparts (superstring).
    if h.contains("medium.en") {
        return Some(AHEADS_MEDIUM_EN);
    }
    if h.contains("medium") {
        return Some(AHEADS_MEDIUM);
    }
    if h.contains("small.en") {
        return Some(AHEADS_SMALL_EN);
    }
    if h.contains("small") {
        return Some(AHEADS_SMALL);
    }
    if h.contains("base.en") {
        return Some(AHEADS_BASE_EN);
    }
    if h.contains("base") {
        return Some(AHEADS_BASE);
    }
    if h.contains("tiny.en") {
        return Some(AHEADS_TINY_EN);
    }
    if h.contains("tiny") {
        return Some(AHEADS_TINY);
    }
    None
}

// ───────────────────────────────────────────────────────────────────────────
// Median filter (port of whisper.cpp median_filter, 8802-8835)
// ───────────────────────────────────────────────────────────────────────────

/// In-place 1-D median filter over `row` with odd `width`, using **reflect**
/// (edge-mirroring) padding — a faithful port of whisper.cpp's `median_filter`
/// (8802-8835): for an out-of-range index `idx`, `idx<0 ⇒ -idx` and `idx>=n ⇒
/// 2*(n-1)-idx`, then the window is sorted and the middle element chosen.
///
/// `width` must be odd and `< 2*row.len()` for the reflect to be well-defined; a
/// `width` of `1` (or any `width >= 2*len`) leaves `row` unchanged for the
/// degenerate single-sample case. Even `width` is rounded down to the nearest
/// odd value (whisper.cpp asserts oddness; we are lenient).
pub fn median_filter(row: &mut [f32], width: usize) {
    let n = row.len();
    if n == 0 || width <= 1 {
        return;
    }
    let w = if width.is_multiple_of(2) {
        width - 1
    } else {
        width
    };
    let half = (w / 2) as i64;
    let len = n as i64;

    let src: Vec<f32> = row.to_vec();
    let mut window: Vec<f32> = Vec::with_capacity(w);
    for (k, out) in row.iter_mut().enumerate() {
        window.clear();
        for off in -half..=half {
            let mut idx = k as i64 + off;
            if idx < 0 {
                idx = -idx;
            } else if idx >= len {
                idx = 2 * (len - 1) - idx;
            }
            // Reflect can still land out of range when width >= 2*len; clamp.
            let idx = idx.clamp(0, len - 1) as usize;
            window.push(src[idx]);
        }
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        *out = window[window.len() / 2];
    }
}

// ───────────────────────────────────────────────────────────────────────────
// DTW (port of whisper.cpp dtw_and_backtrace, 8712-8796)
// ───────────────────────────────────────────────────────────────────────────

/// Classic monotonic DTW + backtrace over a `[n_tokens, n_frames]` cost matrix
/// (`cost.data[i * n_frames + j]` = cost of aligning token `i` to frame `j`,
/// typically `-attention`).
///
/// Returns the optimal warping path as `(token, frame)` pairs, monotonically
/// non-decreasing in both axes, from `(0, 0)` to `(n_tokens-1, n_frames-1)`.
/// Allowed steps are diagonal `(i-1, j-1)`, "hold token" `(i-1, j)`, and "hold
/// frame" `(i, j-1)` — a faithful port of whisper.cpp's accumulation and
/// backtrace, including its tie-break (strict `<` comparisons that fall through
/// to the "hold frame" / `t = 2` branch on ties).
///
/// Degenerate shapes are handled: an empty matrix (`n_tokens == 0` or
/// `n_frames == 0`) yields an empty path; a `1×1` matrix yields `[(0, 0)]`.
#[must_use]
pub fn dtw_path(cost: &Mat) -> Vec<(usize, usize)> {
    let n = cost.rows; // tokens
    let m = cost.cols; // frames
    if n == 0 || m == 0 {
        return Vec::new();
    }

    // Accumulated cost, padded by one (row/col 0 are the "before start"
    // boundary). cost_acc[(i)*(m+1) + j], i in 0..=n, j in 0..=m.
    let inf = f32::INFINITY;
    let mut acc = vec![inf; (n + 1) * (m + 1)];
    // 0 = diagonal, 1 = up (hold token), 2 = left (hold frame).
    let mut trace = vec![-1i8; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;

    acc[at(0, 0)] = 0.0;

    for j in 1..=m {
        for i in 1..=n {
            let c0 = acc[at(i - 1, j - 1)]; // diagonal
            let c1 = acc[at(i - 1, j)]; // hold token
            let c2 = acc[at(i, j - 1)]; // hold frame

            // whisper.cpp tie-break: strict comparisons, default to c2/t=2.
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0i8)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1i8)
            } else {
                (c2, 2i8)
            };
            // Sanitize a NaN cost (e.g. from a poisoned upstream softmax) to
            // +inf so a NaN frame is never selected as a path minimum — a NaN
            // makes every `<` comparison false and would silently bias the path
            // toward the hold-frame branch and corrupt token boundaries.
            let cell = cost.data[(i - 1) * m + (j - 1)];
            let cell = if cell.is_nan() { inf } else { cell };
            acc[at(i, j)] = cell + c;
            trace[at(i, j)] = t;
        }
    }

    // Boundary trace seeding (whisper.cpp 8756-8759): trace[0, :] = 2 (left),
    // trace[:, 0] = 1 (up). This guarantees the backtrace terminates at (0,0).
    for j in 0..=m {
        trace[at(0, j)] = 2;
    }
    for i in 0..=n {
        trace[at(i, 0)] = 1;
    }

    // Backtrace from (n, m) to (0, 0), emitting (i-1, j-1) pairs.
    let mut path: Vec<(usize, usize)> = Vec::with_capacity(n + m);
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        path.push((i - 1, j - 1));
        match trace[at(i, j)] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1, // 2 (and the seeded boundaries)
        }
    }
    path.reverse();
    path
}

// ───────────────────────────────────────────────────────────────────────────
// Full per-window token-timestamp pipeline
// ───────────────────────────────────────────────────────────────────────────

/// Compute per-**text-token** END times (seconds) for one window from the
/// recorded cross-attention weights.
///
/// - `attn`: the decoder's recorded weights, one `[tokens, enc_frames]` matrix
///   per `(layer, head)` in `layer * n_head + head` order (the exact shape
///   produced by [`crate::native_engine::decoder::DecoderState::cross_attn_weights`]).
///   The recorded prompt is `sot_sequence (sot[,lang],not) + <text tokens> +
///   eot`, so the rows are `[<sot-seq rows>, <text rows>, <eot row>]`.
/// - `n_head`: the model's `n_text_head` (used to map `(layer, head)` → index).
/// - `heads`: the selected alignment heads from [`alignment_heads`].
/// - `first_text_row`: index of the first **text** token row in `attn`
///   (= `sot_sequence_length`). Upstream removes the `sot_sequence_length + 1`
///   leading rows (sot seq + ... actually sot seq, with the trailing eot row
///   dropped separately) **before** normalization (whisper.cpp 8946-8947), so
///   the z-norm / median-filter / DTW statistics see text rows only. We slice
///   the attention matrices to `[first_text_row, first_text_row + n_text_rows)`
///   up front to match those upstream stats exactly.
/// - `n_text_rows`: number of text token rows (excludes the trailing eot row,
///   whisper.cpp's `- sot_sequence_length - 1` view, 8947).
/// - `n_audio_frames`: the window's *actual* audio length in encoder frames
///   (`audio_len_sec / 0.02`), so the padded tail is excluded
///   (whisper.cpp `n_audio_tokens = n_frames/2`, 8898).
/// - `medfilt_width`: median-filter width (use [`DEFAULT_MEDFILT_WIDTH`]).
///
/// The returned vector has one **end** time per text token (length
/// `n_text_rows`), in order — the frame at which the DTW path leaves that
/// token's row (whisper.cpp 8958-8985 END-boundary convention, see below). An
/// empty result is returned when no usable heads/frames/text-tokens are present.
///
/// # Boundary convention (fix #4)
///
/// Upstream (whisper.cpp 8958-8985) walks the DTW path with `last_v = 0` and,
/// each time the path's token index `v` changes, assigns that step's frame to
/// the token it is *leaving* (the previous `tok_i`), then advances `tok_i`.
/// Hence the recorded time is the frame where the path ENTERS THE NEXT ROW,
/// which is the **END boundary** of the current token (not the first frame it
/// enters). We reproduce that here: `ends[r]` = `FRAME_SEC * frame` at the step
/// where the path transitions from row `r` to row `r+1`. The final text token's
/// exit is never observed (the path ends inside its row), so it is left to the
/// caller to close at the window/segment end — exactly as upstream leaves the
/// last token's `t_dtw` unset and derives it from the segment bound.
///
/// Port of whisper.cpp `whisper_exp_compute_token_level_timestamps_dtw`
/// (8837-8990): select heads → restrict frames → **slice to text rows** →
/// z-normalize per head over the token axis → median-filter each token row over
/// frames → average heads → negate → DTW → token END-boundary times.
#[must_use]
pub fn token_timestamps(
    attn: &[Mat],
    n_head: usize,
    heads: &[(usize, usize)],
    first_text_row: usize,
    n_text_rows: usize,
    n_audio_frames: usize,
    medfilt_width: usize,
) -> Vec<f32> {
    if attn.is_empty() || n_head == 0 || n_text_rows == 0 {
        return Vec::new();
    }
    let all_rows = attn[0].rows;
    let enc_frames = attn[0].cols;
    if all_rows == 0 || enc_frames == 0 {
        return Vec::new();
    }
    // Text rows must fit inside the recorded matrix.
    if first_text_row + n_text_rows > all_rows {
        return Vec::new();
    }
    let n_tokens = n_text_rows;
    // Restrict to the window's real audio length (exclude padded tail).
    let n_frames = n_audio_frames.min(enc_frames).max(1);

    // Gather the selected head matrices that actually exist in `attn`.
    let selected: Vec<&Mat> = heads
        .iter()
        .filter_map(|&(l, h)| {
            let idx = l * n_head + h;
            attn.get(idx)
        })
        .filter(|m| m.rows == all_rows && m.cols == enc_frames)
        .collect();
    if selected.is_empty() {
        return Vec::new();
    }

    // Accumulator for the head-averaged, normalized+filtered matrix, laid out
    // [n_tokens (text rows), n_frames].
    let mut avg = vec![0.0f32; n_tokens * n_frames];

    for m in &selected {
        // Copy this head, sliced to the TEXT rows only and restricted to
        // [n_tokens, n_frames]. Upstream removes the sot-sequence + eot rows
        // BEFORE ggml_norm (whisper.cpp 8946-8947), so the per-frame mean/std
        // below see text rows only (fix #3).
        let mut head = vec![0.0f32; n_tokens * n_frames];
        for t in 0..n_tokens {
            let src_row = first_text_row + t;
            let src = &m.data[src_row * enc_frames..src_row * enc_frames + n_frames];
            head[t * n_frames..(t + 1) * n_frames].copy_from_slice(src);
        }

        // z-normalize over the TOKEN axis, per frame (OpenAI dim=-2;
        // whisper.cpp ggml_norm after permute, 8929). i.e. for each frame
        // column, subtract the column mean and divide by the column std.
        normalize_over_tokens(&mut head, n_tokens, n_frames);

        // median-filter each token row over the frame axis (whisper.cpp 8936).
        for t in 0..n_tokens {
            let row = &mut head[t * n_frames..(t + 1) * n_frames];
            median_filter(row, medfilt_width);
        }

        for (a, &h) in avg.iter_mut().zip(head.iter()) {
            *a += h;
        }
    }

    let inv = 1.0 / selected.len() as f32;
    // Average over heads, then negate for the cost matrix (whisper.cpp
    // ggml_mean + ggml_scale(-1), 8946-8947).
    let mut cost_data = avg;
    for v in &mut cost_data {
        *v = -(*v * inv);
    }
    let cost = Mat::from_vec(n_tokens, n_frames, cost_data);

    let path = dtw_path(&cost);

    // Token END-boundary extraction (fix #4 — whisper.cpp 8958-8985): walk the
    // path; each time the row index changes, the frame at that transition is the
    // END boundary of the row we are LEAVING. `time = frame * 0.02`.
    let mut ends = vec![f32::NAN; n_tokens];
    let mut last_tok: i64 = 0;
    for &(tok, frame) in &path {
        let v = tok as i64;
        if v != last_tok {
            // Path just entered row `v` at `frame`; that frame closes every row
            // in (last_tok, v) (a multi-row jump closes each on the same frame —
            // matches upstream advancing `tok_i` one at a time on the same
            // timestamp). Assign the END to the row being left, i.e. v-1.
            let end_t = frame as f32 * FRAME_SEC;
            let lo = last_tok.max(0) as usize;
            let hi = (v as usize).min(n_tokens);
            for slot in ends.iter_mut().take(hi).skip(lo) {
                if slot.is_nan() {
                    *slot = end_t;
                }
            }
            last_tok = v;
        }
    }
    // The final text token's exit is never observed (the path terminates inside
    // its row); upstream leaves it for the segment bound. Default any unobserved
    // (NaN) row to the last real frame's time, never going backwards — callers
    // typically override the last token's end with the segment/window end.
    let last_frame_t = (n_frames.saturating_sub(1)) as f32 * FRAME_SEC;
    let mut prev = 0.0f32;
    for t in &mut ends {
        if t.is_nan() {
            *t = prev.max(last_frame_t);
        } else {
            prev = *t;
        }
    }
    ends
}

/// z-normalize a `[n_tokens, n_frames]` matrix over the **token** axis: for each
/// frame column, subtract the column mean and divide by the column standard
/// deviation (population std; whisper.cpp `ggml_norm` with eps `1e-9`, 8929).
///
/// OpenAI's `find_alignment` normalizes `weights` over `dim=-2` (the token
/// axis); whisper.cpp permutes the token axis into `ggml_norm`'s leading axis to
/// achieve the same. We operate directly column-wise.
fn normalize_over_tokens(data: &mut [f32], n_tokens: usize, n_frames: usize) {
    if n_tokens == 0 {
        return;
    }
    let n = n_tokens as f32;
    for f in 0..n_frames {
        let mut mean = 0.0f32;
        for t in 0..n_tokens {
            mean += data[t * n_frames + f];
        }
        mean /= n;
        let mut var = 0.0f32;
        for t in 0..n_tokens {
            let d = data[t * n_frames + f] - mean;
            var += d * d;
        }
        var /= n;
        let inv_std = 1.0 / (var + 1e-9).sqrt();
        // Defensive: if this column carried NaN attention weights (a poisoned
        // upstream softmax makes mean/var/inv_std NaN), degrade inv_std to a
        // finite 0.0 rather than scaling by NaN.
        let inv_std = if inv_std.is_finite() { inv_std } else { 0.0 };
        for t in 0..n_tokens {
            let v = &mut data[t * n_frames + f];
            *v = (*v - mean) * inv_std;
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Token → word grouping (OpenAI/whisper space-prefix convention)
// ───────────────────────────────────────────────────────────────────────────

/// A timed word: its text and `[start, end]` in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    pub text: String,
    pub start_sec: f64,
    pub end_sec: f64,
}

/// Group decoded text tokens into timed words.
///
/// `tokens` are the window's decoded **text** token byte slices (special tokens
/// already removed by the caller), in order; `starts` are their per-token start
/// times in seconds (from [`token_timestamps`], same length as `tokens`).
/// `window_end_sec` closes the final word.
///
/// Word boundaries follow the whisper/OpenAI convention: a new word begins at
/// the first token, and at any later token whose decoded bytes **begin with a
/// space** (`0x20`). Tokens that do not start with a space (continuation pieces,
/// and standalone punctuation) glue onto the current word. A word's start is its
/// first token's start; its end is the next word's start (the last word ends at
/// `window_end_sec`). Leading whitespace is trimmed from each emitted word;
/// empty/whitespace-only words are dropped.
#[must_use]
pub fn group_tokens_into_words(
    tokens: &[&[u8]],
    starts: &[f32],
    window_end_sec: f64,
) -> Vec<WordTiming> {
    if tokens.is_empty() {
        return Vec::new();
    }

    // First pass: build (text_bytes, start_sec) per word.
    let mut words: Vec<(Vec<u8>, f64)> = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        let start = f64::from(starts.get(i).copied().unwrap_or(0.0));
        let begins_word = i == 0 || tok.first() == Some(&b' ');
        if begins_word || words.is_empty() {
            words.push((tok.to_vec(), start));
        } else {
            words.last_mut().unwrap().0.extend_from_slice(tok);
        }
    }

    // Second pass: close each word's end at the next word's start.
    let mut out = Vec::with_capacity(words.len());
    for k in 0..words.len() {
        let (bytes, start) = &words[k];
        let text = String::from_utf8_lossy(bytes).trim().to_string();
        if text.is_empty() {
            continue;
        }
        let end = if k + 1 < words.len() {
            words[k + 1].1
        } else {
            window_end_sec
        };
        // Guard against any tiny non-monotonicity from the boundary heuristic.
        let end = end.max(*start);
        out.push(WordTiming {
            text,
            start_sec: *start,
            end_sec: end,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hp(n_text_layer: i32, n_text_state: i32, n_text_head: i32, n_vocab: i32) -> WhisperHParams {
        WhisperHParams {
            n_vocab,
            n_audio_ctx: 1500,
            n_audio_state: n_text_state,
            n_audio_head: n_text_head,
            n_audio_layer: n_text_layer,
            n_text_ctx: 448,
            n_text_state,
            n_text_head,
            n_text_layer,
            n_mels: 80,
            ftype: 1,
        }
    }

    // ── alignment_heads ──────────────────────────────────────────────────

    #[test]
    fn tiny_en_returns_its_preset() {
        // tiny.en: 4 layers, 384 state, 6 heads, vocab 51864 (English-only).
        let heads = alignment_heads(&hp(4, 384, 6, 51864), Some("tiny.en"));
        assert_eq!(heads, AHEADS_TINY_EN.to_vec());
    }

    #[test]
    fn tiny_multilingual_returns_tiny_preset() {
        // tiny (multilingual): vocab 51865.
        let heads = alignment_heads(&hp(4, 384, 6, 51865), None);
        assert_eq!(heads, AHEADS_TINY.to_vec());
    }

    #[test]
    fn unknown_model_falls_back_to_top_half() {
        // 8 layers, no preset, 4 heads → top half = layers 4..8, all heads.
        let heads = alignment_heads(&hp(8, 999, 4, 51864), None);
        let mut expected = Vec::new();
        for l in 4..8 {
            for h in 0..4 {
                expected.push((l, h));
            }
        }
        assert_eq!(heads, expected);
    }

    #[test]
    fn v3_turbo_detected_by_four_layers() {
        // large-v3-turbo: 4 text layers, 1280 state — distinct from tiny's 384.
        let heads = alignment_heads(&hp(4, 1280, 20, 51866), None);
        assert_eq!(heads, AHEADS_LARGE_V3_TURBO.to_vec());
    }

    #[test]
    fn large_hint_disambiguates_v1_v2() {
        // (32, 1280) is shared by v1/v2/v3; hint picks the family.
        let h = hp(32, 1280, 20, 51865);
        assert_eq!(
            alignment_heads(&h, Some("large-v1")),
            AHEADS_LARGE_V1.to_vec()
        );
        assert_eq!(
            alignment_heads(&h, Some("large-v2")),
            AHEADS_LARGE_V2.to_vec()
        );
        assert_eq!(
            alignment_heads(&h, Some("large-v3")),
            AHEADS_LARGE_V3.to_vec()
        );
        // bare ggml-large-v3-turbo.bin hint still resolves turbo even at (32,1280).
        assert_eq!(
            alignment_heads(&hp(4, 1280, 20, 51866), Some("ggml-large-v3-turbo.bin")),
            AHEADS_LARGE_V3_TURBO.to_vec()
        );
    }

    #[test]
    fn out_of_grid_heads_are_dropped() {
        // A model with fewer heads than the preset references must not yield
        // head indices that don't exist.
        let heads = alignment_heads(&hp(4, 384, 3, 51864), Some("tiny.en"));
        assert!(heads.iter().all(|&(l, h)| l < 4 && h < 3));
        // tiny_en references head 5 at layer 2 and head 4 at layer 3 — dropped.
        assert!(!heads.contains(&(2, 5)));
    }

    // ── median_filter ────────────────────────────────────────────────────

    #[test]
    fn median_filter_known_vector_width3() {
        // width 3, reflect padding. [1, 5, 2, 8, 3]
        // k=0: window {refl(-1)=row[1]=5, 1, 5} -> sorted {1,5,5} -> 5
        // k=1: {1,5,2} -> {1,2,5} -> 2
        // k=2: {5,2,8} -> {2,5,8} -> 5
        // k=3: {2,8,3} -> {2,3,8} -> 3
        // k=4: {8,3, refl(5)=row[3]=8} -> {3,8,8} -> 8
        let mut row = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        median_filter(&mut row, 3);
        assert_eq!(row, vec![5.0, 2.0, 5.0, 3.0, 8.0]);
    }

    #[test]
    fn median_filter_width1_is_identity() {
        let mut row = vec![3.0, 1.0, 2.0];
        median_filter(&mut row, 1);
        assert_eq!(row, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn median_filter_single_element() {
        let mut row = vec![42.0];
        median_filter(&mut row, 7);
        assert_eq!(row, vec![42.0]);
    }

    #[test]
    fn median_filter_constant_row_unchanged() {
        let mut row = vec![2.0; 9];
        median_filter(&mut row, 7);
        assert_eq!(row, vec![2.0; 9]);
    }

    #[test]
    fn median_filter_even_width_rounds_down() {
        // width 4 behaves like width 3.
        let mut a = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let mut b = a.clone();
        median_filter(&mut a, 4);
        median_filter(&mut b, 3);
        assert_eq!(a, b);
    }

    // ── dtw_path ─────────────────────────────────────────────────────────

    #[test]
    fn dtw_single_cell() {
        let cost = Mat::from_vec(1, 1, vec![0.5]);
        assert_eq!(dtw_path(&cost), vec![(0, 0)]);
    }

    #[test]
    fn dtw_empty_matrix() {
        assert!(dtw_path(&Mat::from_vec(0, 0, vec![])).is_empty());
        assert!(dtw_path(&Mat::from_vec(0, 3, vec![])).is_empty());
        assert!(dtw_path(&Mat::from_vec(3, 0, vec![])).is_empty());
    }

    #[test]
    fn dtw_single_token_many_frames() {
        // One token, 4 frames: path holds the token, walking every frame.
        let cost = Mat::from_vec(1, 4, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(dtw_path(&cost), vec![(0, 0), (0, 1), (0, 2), (0, 3)]);
    }

    #[test]
    fn dtw_single_frame_many_tokens() {
        // One frame, 3 tokens: path holds the frame, walking every token.
        let cost = Mat::from_vec(3, 1, vec![0.0, 0.0, 0.0]);
        assert_eq!(dtw_path(&cost), vec![(0, 0), (1, 0), (2, 0)]);
    }

    #[test]
    fn dtw_diagonal_passes_through_low_cost_cells() {
        // Strong negative cost on the diagonal. With whisper.cpp's tie-break
        // (ties default to the "hold frame" step), the minimum-cost path may
        // expand around the diagonal (e.g. (0,0),(1,0),(1,1),(2,1),(2,2) — same
        // total cost -30 as the pure diagonal). Assert it visits every diagonal
        // low-cost cell, hits the endpoints, and stays monotonic.
        let mut data = vec![0.0f32; 9];
        for i in 0..3 {
            data[i * 3 + i] = -10.0;
        }
        let cost = Mat::from_vec(3, 3, data);
        let path = dtw_path(&cost);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(2, 2)));
        for d in 0..3 {
            assert!(
                path.contains(&(d, d)),
                "path misses diagonal cell ({d},{d})"
            );
        }
        for w in path.windows(2) {
            assert!(w[1].0 >= w[0].0 && w[1].1 >= w[0].1);
        }
    }

    #[test]
    fn dtw_strict_diagonal_when_offdiag_positive() {
        // When off-diagonal cells are strictly *more* costly, the unique optimum
        // is the pure diagonal.
        let mut data = vec![10.0f32; 9];
        for i in 0..3 {
            data[i * 3 + i] = -10.0;
        }
        let cost = Mat::from_vec(3, 3, data);
        assert_eq!(dtw_path(&cost), vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn dtw_endpoints_and_monotonicity() {
        // Arbitrary cost: verify endpoints + monotonic non-decreasing path.
        let data: Vec<f32> = (0..(4 * 6)).map(|x| (x % 5) as f32 - 2.0).collect();
        let cost = Mat::from_vec(4, 6, data);
        let path = dtw_path(&cost);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(3, 5)));
        for win in path.windows(2) {
            assert!(win[1].0 >= win[0].0 && win[1].1 >= win[0].1);
            // each step advances at least one axis.
            assert!(win[1] != win[0]);
        }
    }

    #[test]
    fn dtw_nan_costs_produce_monotonic_finite_path() {
        // A poisoned upstream softmax can leave NaN in the cost matrix. NaN
        // costs are sanitized to +inf so a NaN frame is never chosen as a path
        // minimum; the backtrace must still yield a full, monotonic path (and
        // must not panic). 4 tokens x 6 frames with a NaN spike in the interior.
        let mut data: Vec<f32> = (0..(4 * 6)).map(|x| (x % 5) as f32 - 2.0).collect();
        data[2 * 6 + 3] = f32::NAN;
        let cost = Mat::from_vec(4, 6, data);
        let path = dtw_path(&cost);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(3, 5)));
        for win in path.windows(2) {
            assert!(
                win[1].0 >= win[0].0 && win[1].1 >= win[0].1,
                "not monotonic: {path:?}"
            );
            assert!(win[1] != win[0]);
        }
    }

    // ── token_timestamps ─────────────────────────────────────────────────

    #[test]
    fn token_timestamps_diagonal_attention() {
        // Synthetic: 3 text tokens, 6 frames, attention peaks on a diagonal-ish
        // mapping token t -> frame 2t. One head, head index 0. All rows are
        // text rows (first_text_row=0, n_text_rows=3). Returns END boundaries.
        let n_tokens = 3;
        let enc_frames = 6;
        let mut data = vec![0.01f32; n_tokens * enc_frames];
        for t in 0..n_tokens {
            data[t * enc_frames + (2 * t)] = 1.0;
        }
        let attn = vec![Mat::from_vec(n_tokens, enc_frames, data)];
        // n_head=1, head (0,0); first_text_row=0, n_text_rows=3.
        let ends = token_timestamps(&attn, 1, &[(0, 0)], 0, n_tokens, enc_frames, 7);
        assert_eq!(ends.len(), n_tokens);
        // END boundaries monotonic non-decreasing.
        for w in ends.windows(2) {
            assert!(w[1] >= w[0], "ends not monotonic: {ends:?}");
        }
        // Token 0's END boundary is at/after frame 0 (it must be > 0 — the path
        // leaves row 0 at some real frame).
        assert!(ends[0] >= 0.0);
    }

    #[test]
    fn token_timestamps_end_boundary_convention() {
        // The END-boundary convention (fix #4) records, for each row, the frame
        // at which the DTW path LEAVES that row (= enters the next). Use a clear
        // block-diagonal: 3 tokens over 9 frames, each token owning a 3-frame
        // block. The path tracks the blocks, so token 0's END boundary lands
        // near the block-0/block-1 seam (~frame 3), well above token 0's own
        // first frame (0) — i.e. the time is the END, not the start. The last
        // token's exit is unobserved and defaults to the last real frame.
        let n_tokens = 3;
        let enc_frames = 9;
        let mut data = vec![0.0f32; n_tokens * enc_frames];
        for t in 0..n_tokens {
            for f in (t * 3)..(t * 3 + 3) {
                data[t * enc_frames + f] = 1.0;
            }
        }
        let attn = vec![Mat::from_vec(n_tokens, enc_frames, data)];
        let ends = token_timestamps(&attn, 1, &[(0, 0)], 0, n_tokens, enc_frames, 7);
        assert_eq!(ends.len(), 3);
        // Monotonic non-decreasing END boundaries.
        assert!(ends[0] <= ends[1] && ends[1] <= ends[2], "{ends:?}");
        // Token 0's END boundary is strictly past its first frame (0): it is the
        // frame the path EXITS row 0, an END boundary.
        assert!(
            ends[0] >= 2.0 * FRAME_SEC - 1e-6,
            "token 0 END should be near the block seam, got {ends:?}"
        );
        // The last token defaults to the last real frame's time.
        assert!(
            (ends[2] - (enc_frames - 1) as f32 * FRAME_SEC).abs() < 1e-6,
            "last token end defaults to last frame, got {ends:?}"
        );
    }

    #[test]
    fn token_timestamps_empty_inputs() {
        assert!(token_timestamps(&[], 1, &[(0, 0)], 0, 1, 10, 7).is_empty());
        let attn = vec![Mat::from_vec(0, 6, vec![])];
        assert!(token_timestamps(&attn, 1, &[(0, 0)], 0, 1, 6, 7).is_empty());
        // Zero text rows requested → empty.
        let attn = vec![Mat::from_vec(3, 6, vec![0.0; 18])];
        assert!(token_timestamps(&attn, 1, &[(0, 0)], 0, 0, 6, 7).is_empty());
        // Text rows out of range → empty.
        assert!(token_timestamps(&attn, 1, &[(0, 0)], 2, 5, 6, 7).is_empty());
    }

    #[test]
    fn token_timestamps_slices_text_rows() {
        // 4 rows: row 0 = sot, rows 1-2 = text, row 3 = eot. first_text_row=1,
        // n_text_rows=2. Only the two text rows should be timed.
        let all_rows = 4;
        let enc_frames = 6;
        let mut data = vec![0.01f32; all_rows * enc_frames];
        // text row 1 (token 0) peaks at frame 1; text row 2 (token 1) at frame 4.
        data[enc_frames + 1] = 1.0;
        data[2 * enc_frames + 4] = 1.0;
        let attn = vec![Mat::from_vec(all_rows, enc_frames, data)];
        let ends = token_timestamps(&attn, 1, &[(0, 0)], 1, 2, enc_frames, 7);
        assert_eq!(ends.len(), 2, "one end per text token");
        assert!(ends[0] <= ends[1], "ends monotonic: {ends:?}");
    }

    #[test]
    fn token_timestamps_no_selected_heads() {
        // Head (1,0) requested but only one head matrix present (index 0).
        let attn = vec![Mat::from_vec(2, 4, vec![0.0; 8])];
        assert!(token_timestamps(&attn, 1, &[(1, 0)], 0, 2, 4, 7).is_empty());
    }

    #[test]
    fn token_timestamps_restricts_padded_frames() {
        // 2 tokens, 6 enc frames but only 3 real audio frames. A strong
        // attention spike in the padded region (frame 5) must be ignored.
        let n_tokens = 2;
        let enc_frames = 6;
        let mut data = vec![0.01f32; n_tokens * enc_frames];
        data[0] = 1.0; // token 0 -> frame 0 (real)
        data[enc_frames + 5] = 5.0; // token 1 -> frame 5 (PADDED, ignored)
        data[enc_frames + 2] = 1.0; // token 1 -> frame 2 (real)
        let attn = vec![Mat::from_vec(n_tokens, enc_frames, data)];
        let times = token_timestamps(&attn, 1, &[(0, 0)], 0, n_tokens, 3, 7);
        // Max possible time is bounded by 3 frames * 0.02 = 0.06s.
        assert!(times.iter().all(|&t| t <= 0.06 + 1e-6), "{times:?}");
    }

    #[test]
    fn token_timestamps_deterministic() {
        let n_tokens = 4;
        let enc_frames = 8;
        let mut data = vec![0.0f32; n_tokens * enc_frames];
        for t in 0..n_tokens {
            data[t * enc_frames + (2 * t)] = 1.0;
        }
        let attn = vec![Mat::from_vec(n_tokens, enc_frames, data)];
        let a = token_timestamps(&attn, 1, &[(0, 0)], 0, n_tokens, enc_frames, 7);
        let b = token_timestamps(&attn, 1, &[(0, 0)], 0, n_tokens, enc_frames, 7);
        assert_eq!(a, b);
    }

    // ── group_tokens_into_words ──────────────────────────────────────────

    #[test]
    fn group_words_space_prefix_convention() {
        // " And" " so" " my" — three words, each token starts a word.
        let tokens: Vec<&[u8]> = vec![b" And", b" so", b" my"];
        let starts = vec![0.32, 0.40, 0.69];
        let words = group_tokens_into_words(&tokens, &starts, 1.0);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "And");
        assert!((words[0].start_sec - 0.32).abs() < 1e-6);
        assert!((words[0].end_sec - 0.40).abs() < 1e-6);
        assert_eq!(words[2].text, "my");
        assert!((words[2].end_sec - 1.0).abs() < 1e-6); // closed at window end.
    }

    #[test]
    fn group_words_continuation_pieces_glue() {
        // "Amer" + "icans" (no leading space) => one word "Americans".
        let tokens: Vec<&[u8]> = vec![b" Amer", b"icans"];
        let starts = vec![1.75, 2.5];
        let words = group_tokens_into_words(&tokens, &starts, 3.29);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "Americans");
        assert!((words[0].start_sec - 1.75).abs() < 1e-6);
        assert!((words[0].end_sec - 3.29).abs() < 1e-6);
    }

    #[test]
    fn group_words_punctuation_glues_to_previous() {
        // "country" + "." => "country." one word (no leading space on ".").
        let tokens: Vec<&[u8]> = vec![b" country", b"."];
        let starts = vec![10.0, 10.7];
        let words = group_tokens_into_words(&tokens, &starts, 10.76);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "country.");
    }

    #[test]
    fn group_words_first_token_starts_word_without_space() {
        // First token has no leading space but still begins a word.
        let tokens: Vec<&[u8]> = vec![b"Hello", b" world"];
        let starts = vec![0.0, 0.5];
        let words = group_tokens_into_words(&tokens, &starts, 1.0);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn group_words_empty_tokens() {
        assert!(group_tokens_into_words(&[], &[], 1.0).is_empty());
    }
}
