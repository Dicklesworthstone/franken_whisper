# Known Conformance Divergences

> This document tracks intentional or investigating deviations from the 
> `docs/engine_compatibility_spec.md`.

## DISC-001: Floating-point precision in SRT timestamps
- **Reference:** whisper.cpp SRT output rounds to milliseconds.
- **Our impl:** `src/export.rs` uses `.round()` then formats.
- **Impact:** Negligible drift (< 1ms).
- **Resolution:** ACCEPTED.
- **Tests affected:** All fixtures using `diarization_srt` format.
- **Review date:** 2026-04-12

## DISC-002: Speaker label remapping
- **Reference:** Bridge adapters use engine-specific prefixes (e.g., `SPEAKER_00`).
- **Our impl:** Native pilots may use different internal IDs during rollout.
- **Impact:** Cross-engine comparison requires `require_speaker_exact = false`.
- **Resolution:** ACCEPTED per spec §3.3.
- **Tests affected:** `corpus/*_cross_engine.json`.
- **Review date:** 2026-04-12

## DISC-003: Greedy vs beam-search divergence between native engine and whisper-cli defaults
- **Reference:** `whisper-cli` defaults to **beam search** (`-bs 5`).
- **Our impl:** The native in-process engine (`src/native_engine/decode.rs`) decodes **greedily** (temperature 0, no beam) at rollout stages below `primary`.
- **Impact:** Occasional word-choice differences and timestamp drift between the two engines on the same audio. Measured on `jfk.wav` + `tiny.en`: text WER 0.0 but final-segment **end-timestamp drift ~240 ms** (native 11.00s vs bridge 10.76s). The transcript text matches; only the tail segment boundary shifts.
- **Resolution:** **ACCEPTED for rollout stages below `primary`.** The bridge-vs-native conformance gate uses a dedicated **native-rollout tolerance profile** — WER ≤ 0.10 and per-segment timestamps within **0.3 s** — deliberately looser than the canonical 50 ms (`CANONICAL_TIMESTAMP_TOLERANCE_SEC`). **Revisit (tighten back toward canonical) when native beam search lands** and the engine is promoted to `primary`.
- **Tests affected:** `tests/conformance_comparator_tests.rs::gated_bridge_vs_native_conformance_jfk_tiny_en` (the new bridge-vs-native real-engine comparison).
- **Review date:** 2026-06-04

## DISC-004: Tail-window encoder-context truncation (audio_ctx)
- **Reference:** whisper.cpp's **default** behavior pads every 30 s window to the full `n_audio_ctx = 1500` encoder context (3000 mel frames), even a near-empty final window. whisper.cpp *also* ships an **opt-in** `audio_ctx` / `-ac` knob (`whisper_full_params.audio_ctx`, whisper.cpp 6967-6972; conv input is `2*n_ctx` wide with `n_ctx = exp_n_audio_ctx`, 1982/1995) that runs the encoder with a *reduced* context for shorter audio — explicitly trading a small accuracy hit for a large speedup.
- **Our impl:** `src/native_engine/decode.rs` enables a **scoped, automatic** form of `audio_ctx` **by default**: for any **non-first** window whose remaining real (unpadded) audio is under a full 30 s, the encoder runs with `enc_ctx = ceil(real_frames/2).clamp(64, 1500)` and is fed a truncated `2*enc_ctx`-frame mel chunk (`tail_enc_ctx` + the relaxed even-frame check in `encoder::forward`). The **first window is never truncated** (it carries the bulk of a short clip's real audio; truncating it would change the *main* transcript), so single-window clips and every window's *speech* content are byte-identical to the full-pad path.
- **Floor:** `MIN_ENC_CTX = 64` encoder frames (≈ 1.28 s; conv sees 128 mel frames) — a conservative practical floor (whisper.cpp's `-ac` has none) that keeps the embedding well-conditioned while still saving the bulk of a tail encode.
- **Precision invariance:** the `max_initial_ts` clamp's `tid0` stays derived from the **full model** `n_audio_ctx` (1500), never the truncated window ctx — matching whisper.cpp 6322 (`precision = WHISPER_CHUNK_SIZE / hparams.n_audio_ctx`, which uses `hparams.n_audio_ctx`, not `exp_n_audio_ctx`). Timestamp tokens are window-relative 0.02 s steps and are unaffected. The decoder cross-attention / cross-K-V and DTW frame count are already `enc_frames`-driven (`DecoderState::new`, `dtw::token_timestamps` clamps `n_audio_frames.min(enc_frames)`), so they adapt with no plumbing change.
- **Impact (measured, jfk.wav, large-v3-turbo, 8 threads, release-perf):** the tail window (#2, 0.6 s of real audio after the speech ends) encoder pass drops **4210 ms → 236 ms (~94 %, ~3.97 s saved)**; cross-K-V 55.7 ms → 3.8 ms; end-to-end wall **11.0 s → 7.0 s (~4.0 s saved)** — this is profiling hotspot #1. The **main transcript is byte-identical** to the full-pad golden (`...your country.`). The **tail segment text changes**: the full-pad golden emits the hallucination `"Thank you."` on that 0.6 s of trailing silence; truncated emits a *different* hallucination `"a."`. whisper-cli (large-v3-turbo, beam search) emits **no** trailing segment at all on the same clip, so neither the golden nor the truncated tail matches ground truth — the lever does not regress any real speech, it only perturbs an already-spurious silence hallucination. tiny.en/jfk is a single window → truncation never engages → byte-identical to golden.
- **Kill switch:** `FRANKEN_WHISPER_NATIVE_TAIL_TRUNCATE=0` (or `false`) disables the lever entirely (read once via `OnceLock`), restoring exact full-pad behavior — verified byte-identical to the golden for **both** tiny.en and large-v3-turbo.
- **Resolution:** **ACCEPTED (default ON).** Mirrors upstream's own sanctioned `audio_ctx` optimization, scoped to never touch the content-bearing first window; output divergence is confined to spurious trailing-silence hallucinations on tail windows, which are not ground-truth content. Revisit the `MIN_ENC_CTX` floor / first-window-exemption if a future corpus shows a tail window carrying real speech that the truncation degrades.
- **Tests affected:** `src/native_engine/decode.rs` hermetic `tail_enc_ctx_*` unit tests; `src/native_engine/encoder.rs` `truncated_even_frame_window_is_accepted` / `odd_or_oversized_frame_count_is_rejected`; gated e2e (`gated_e2e_jfk_tiny_en_matches_reference`) stays byte-exact (single window, no truncation).
- **Review date:** 2026-06-05

## DISC-005: Temperature-fallback ladder (`FW_TEMP_FALLBACK`) is sampling-based and not byte-reproducible against pure-greedy
- **Reference:** whisper.cpp recovers failed windows via temperature fallback — retries at `t = 0.2…1.0` with multinomial sampling, `greedy.best_of = 5` candidates per temperature, prompt conditioning only below `t = 0.5`, and per-sequence scoring (`whisper_sequence_score`).
- **Our impl:** `src/native_engine/decode.rs` ships the same ladder **gated, default-OFF** (`FW_TEMP_FALLBACK=1`; `FW_TEMP_BEST_OF` overrides the candidate count). Triggers: window closes no timestamp, avg logprob < −1.0, or a low-entropy repetitive tail (whisper.cpp `entropy_thold` 2.4). Sampling is deterministic per (window, rung, candidate) via a seeded SplitMix64 stream, so gate-ON output is **replayable run-to-run** — but by construction it diverges from the pure-greedy transcript whenever a window fails the quality gate.
- **Impact (measured, tiny.en, `example_audio_track_01` 124.5 s, timestamps mode):** default greedy drops two full 30 s windows (643 chars); gate-ON recovers both (1273 chars with best_of = 5), md5-identical across runs. Divergence is confined to quality-failed windows; clean clips (`jfk.wav`) are byte-identical with the gate on or off.
- **Kill switch:** default. The gate is opt-in; unset ⇒ the ladder never fires and decode is byte-identical to the pre-ladder engine. `FW_TEMP_BEST_OF=1` additionally reproduces the single-candidate ladder byte-for-byte for A/B archaeology.
- **Resolution:** **ACCEPTED (default OFF; enabling is a quality-over-reproducibility trade the operator makes explicitly).** The default-ON decision is deliberately reserved (see NEGATIVE_EVIDENCE 2026-07-14: "owner faithfulness call") because it changes golden transcripts on repetitive/tiled audio — toward whisper.cpp, per the measured tiled-jfk comparison. Revisit alongside native beam search (DISC-003's revisit point).
- **Tests affected:** `src/native_engine/decode.rs` sampler/entropy/score unit tests (`sample_token_*`, `token_tail_entropy_matches_whisper_cpp_reference`, `sequence_score_matches_whisper_cpp_defaults`); gated e2e goldens unaffected (gate off in CI).
- **Review date:** 2026-07-22
