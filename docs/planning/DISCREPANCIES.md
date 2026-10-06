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
- **Our impl:** The native in-process engine (`src/native_engine/decode.rs`) decodes **greedily** (temperature 0, no beam) by default. Beam search is implemented behind `FW_BEAM_SIZE` (default 1 = greedy) but stays opt-in. **Measured to reduce WER on hard audio** (Steve Jobs iPhone keynote, tiny.en, drop removed via `FW_RETRY_FAILED_WINDOW`): `FW_BEAM_SIZE=5` cuts WER **0.1044 → 0.0984**, crossing below this profile's 0.10 gate that greedy fails (NEGATIVE_EVIDENCE 2026-07-23). The full long-form quality stack is `FW_RETRY_FAILED_WINDOW=1 FW_BEAM_SIZE=5` (the retry flag has been **default-on** since 2026-07-24; residual drops are surfaced structurally via `raw_output.dropped_windows` / `RunReport.warnings` per bd-nqzf).
- **Impact:** Occasional word-choice, punctuation, and timestamp differences between engines or compute backends on the same audio. On `jfk.wav` + `tiny.en`, native beam-5 on aarch64 inserts one comma after `so`; native greedy and the current Metal `whisper-cli -bs 5 -bo 1 -nf` omit it, while current CPU-only whisper.cpp also changes capitalization and segment closure. All observed variants have WER 0.0. The final-segment **end-timestamp drift is ~240 ms** (native 11.00s vs bridge 10.76s).
- **Resolution:** **ACCEPTED for rollout stages below `primary`.** The bridge-vs-native conformance gate uses a dedicated **native-rollout tolerance profile** — WER ≤ 0.10 and per-segment timestamps within **0.3 s** — deliberately looser than the canonical 50 ms (`CANONICAL_TIMESTAMP_TOLERANCE_SEC`). **Revisit (tighten back toward canonical) when native beam search lands** and the engine is promoted to `primary`.
- **WER metric:** the gate uses real edit-distance WER (`conformance::word_error_rate`, edits ÷ reference words, ASR-normalized). Measured long-form quality (2026-07-23, `tiny.en`, `example_audio_track_01`, vs `whisper-cli` beam=5): greedy default WER 0.528, `FW_RETRY_FAILED_WINDOW` 0.164, `FW_TEMP_FALLBACK` 0.192 — the residual above 0.10 is the greedy-vs-beam gap this discrepancy is about.
- **Tests affected:** `tests/conformance_comparator_tests.rs::gated_bridge_vs_native_conformance_jfk_tiny_en` and `src/native_engine/decode.rs::gated_beam_size_field_preserves_jfk_words`. The latter retains the byte-exact greedy oracle, requires beam WER 0.0, and admits only the two reviewed native punctuation strings.
- **Review date:** 2026-08-08 (corrected the invalid beam-equals-greedy premise)

## DISC-004: Tail-window encoder-context truncation (audio_ctx)
- **Reference:** whisper.cpp's **default** behavior pads every 30 s window to the full `n_audio_ctx = 1500` encoder context (3000 mel frames), even a near-empty final window. whisper.cpp *also* ships an **opt-in** `audio_ctx` / `-ac` knob that runs the encoder with a *reduced* context for shorter audio, explicitly trading accuracy for speed.
- **Our impl:** `src/native_engine/decode.rs` offers a scoped, automatic form of `audio_ctx` behind `FRANKEN_WHISPER_NATIVE_TAIL_TRUNCATE`: for any **non-first** window whose remaining real audio is under 30 s, the encoder runs with `enc_ctx = ceil(real_frames/2).clamp(64, 1500)` on a truncated `2*enc_ctx`-frame mel chunk (`tail_enc_ctx`). The first window is never truncated. Per-request `AudioCtxPolicy::{Auto, Fixed}` (listen) is separate and unaffected.
- **Precision invariance:** `max_initial_ts` stays derived from the full model `n_audio_ctx`; timestamp tokens keep absolute 20 ms meaning; cross-K/V and DTW adapt to `enc_frames`.
- **History:** landed default-ON 2026-06-05 as a speed lever (jfk.wav turbo tail encode 4210 ms → 236 ms, e2e 11.0 s → 7.0 s on 8 threads) on the evidence that it only perturbed spurious trailing-silence hallucinations.
- **Correctness finding (2026-10-06, Linux x86_64, large-v3-turbo f16 + tiny.en pinned packages, matched greedy timestamp-mode decoding, whisper.cpp v1.7.5 and v1.8.2 built from source as oracles):** a truncated tail context makes the decoder hallucinate prior content that is not in the audio.
  - turbo, JFK ×4 with 1.5 s digital-silence gaps (48.5 s): truncation emits a **fifth verbatim copy** of the sentence as a 47.88–47.90 s segment inside the 0.62 s final window. Truncation floors of 64, 250, and 500 encoder frames all hallucinate it; only the full 1500 context does not. whisper.cpp v1.8.2 (with and without temperature fallback) and v1.7.5 emit exactly four segments ending at 47.88 s.
  - tiny.en, gapless JFK ×3: truncation re-renders the final tile ("country" ×8 instead of ×6) — the long-red `gated_max_context_zero_disables_prompt_carry` pin (bd-4ep1), previously attributed to irreducible FFT epsilon. Full context: exactly ×6.
  - turbo, JFK ×3 / ×5 with gaps: content correct either way, but truncation stretches the final segment end to the file end (35.99 s / 60.99 s) where full context and whisper.cpp end at 35.40 / 60.40 s (wc 35.38 / 60.38 s).
  - turbo, JFK and JFK + 4 s silence (single speech window): identical either way.
  - Full context matched whisper.cpp v1.8.2 in segment count, text, and end times (±20 ms) on every fixture above.
- **Resolution:** **REVISED — default OFF (opt-in).** The lever is not a transcript-equivalent optimization, so it cannot back speed claims. Unset/`0`/`false` ⇒ full-context tails exactly like whisper.cpp; any other value of `FRANKEN_WHISPER_NATIVE_TAIL_TRUNCATE` re-enables truncation for operators who accept the risk. Cost: one full encoder pass for each file's final partial window (measured here at opt-level 2, 4 threads, including model load: JFK ×3 53 → 58 s, JFK ×5 62 → 68 s; short single-window clips pay one extra full encode for their tail window, ~4 s on the 8-thread reference box).
- **Tests affected:** `src/native_engine/decode.rs::gated_max_context_zero_disables_prompt_carry` (green by default again), new `gated_turbo_tiled_jfk_final_window_emits_no_repeat`; `tests/conformance_comparator_tests.rs::gated_audio_ctx_policy_mechanism_ab_jfk_tiny_en` (Full now expects full context at every seek); hermetic `tail_enc_ctx_*` unit tests unchanged (they pass `enabled` explicitly).
- **Review date:** 2026-10-06

## DISC-005: Temperature-fallback ladder (`FW_TEMP_FALLBACK`) is sampling-based and not byte-reproducible against pure-greedy
- **Reference:** whisper.cpp recovers failed windows via temperature fallback — retries at `t = 0.2…1.0` with multinomial sampling, `greedy.best_of = 5` candidates per temperature, prompt conditioning only below `t = 0.5`, and per-sequence scoring (`whisper_sequence_score`).
- **Our impl:** `src/native_engine/decode.rs` ships the same ladder **gated, default-OFF** (`FW_TEMP_FALLBACK=1`; `FW_TEMP_BEST_OF` overrides the candidate count). Triggers: window closes no timestamp, avg logprob < −1.0, or a low-entropy repetitive tail (whisper.cpp `entropy_thold` 2.4). Sampling is deterministic per (window, rung, candidate) via a seeded SplitMix64 stream, so gate-ON output is **replayable run-to-run** — but by construction it diverges from the pure-greedy transcript whenever a window fails the quality gate.
- **Impact (measured, tiny.en, `example_audio_track_01` 124.5 s, timestamps mode):** default greedy drops two full 30 s windows (643 chars); gate-ON recovers both (1273 chars with best_of = 5), md5-identical across runs. Divergence is confined to quality-failed windows; clean clips (`jfk.wav`) are byte-identical with the gate on or off.
- **Kill switch:** default. The gate is opt-in; unset ⇒ the ladder never fires and decode is byte-identical to the pre-ladder engine. `FW_TEMP_BEST_OF=1` additionally reproduces the single-candidate ladder byte-for-byte for A/B archaeology.
- **Resolution:** **ACCEPTED (default OFF; enabling is a quality-over-reproducibility trade the operator makes explicitly).** The default-ON decision is deliberately reserved (see NEGATIVE_EVIDENCE 2026-07-14: "owner faithfulness call") because it changes golden transcripts on repetitive/tiled audio — toward whisper.cpp, per the measured tiled-jfk comparison. Revisit alongside native beam search (DISC-003's revisit point).
- **Tests affected:** `src/native_engine/decode.rs` sampler/entropy/score unit tests (`sample_token_*`, `token_tail_entropy_matches_whisper_cpp_reference`, `sequence_score_matches_whisper_cpp_defaults`); gated e2e goldens unaffected (gate off in CI).
- **Review date:** 2026-07-22

## DISC-006: Acoustic speaker labels are permutation-stable, not identity-stable

- **Reference:** External diarizers expose backend-specific labels and confidence
  semantics; whisper.cpp byte-exactness does not define a waveform speaker
  profile or independent turn timeline.
- **Our impl:** `src/diarization.rs` emits opaque within-run references after
  deterministic constrained clustering. Anchored caller references sort first;
  unanchored labels use earliest reliable occurrence plus a total-order compact
  feature-vector tie-break. ASR text/confidence remain authoritative and
  unchanged by projection, but cluster numbers need not match an external
  backend.
- **Impact:** Cross-engine scoring must use maximum-overlap permutation before
  DER/JER. Labels cannot be interpreted as name, gender, or legal identity.
  The historical text/temporal heuristic is rejected by both acoustic and
  verified-external provenance gates.
- **Rollout:** `auto` defaults to `shadow`; explicit
  `--diarization-engine acoustic` is available. Promotion requires retained
  public-corpus accuracy/calibration and same-host performance evidence. Those
  certification states are currently `NO-DATA`.
- **Resolution:** **ACCEPTED as a new typed output contract, not a byte-exact
  whisper.cpp claim.**
- **Tests affected:** acoustic contract/scoring, deterministic replay,
  hard/soft hint, clustering, projection, persistence, and rollout resolver
  tests.
- **Review date:** 2026-07-28

## DISC-007: Native Sortformer archive-profile comparison used mismatched streaming geometry

- **Reference:** The pinned NeMo Streaming Sortformer v2.1 adapter emits the
  authenticated anonymous speaker turns for the same normalized public WAV.
- **Our impl:** The original safe Rust invocation used the archive-default
  `188/1/1/0/188/188` chunk/context/FIFO/update/cache geometry while the NeMo
  comparison lane used NVIDIA's recommended `340/1/40/40/300/188` profile.
  That mismatched comparison differed at four 80 ms boundaries among 16 turns:
  native/reference `13840/13760` ms (start), `64800/64880` ms (end),
  `74480/74400` ms (end), and `101840/101760` ms (end).
- **Historical impact:** On this one public development row, archive-profile
  native DER/JER was
  `0.021214713430` / `0.029991623791` versus NeMo
  `0.019846022241` / `0.029477961362`. Speaker-lane identities and the other
  turn boundaries matched. It was a valid configuration-comparison loss, not a
  valid same-profile implementation-parity row.
- **Resolution:** **RESOLVED for the accepted recommended profile.** Native now
  uses the published recommended geometry. A regenerated identity-bound public
  pack covers four complete/declared fixtures and 4,540 L1-L8 tensors. On the
  full 102-second row, native L5 drift is inside the unchanged frozen envelope
  and native L7 activity plus all 16 L8 turns are byte-exact against source.
  Sortformer remains evaluation-only because corpus accuracy, fixed-four-lane
  capacity, resource tiers, and product routing are separate gates.
- **Tests affected:** Complete recommended-profile L1-L8 public parity,
  short-final-chunk FIFO regression, pinned libc++ top-k tie regression, and
  full-recording session parity.
- **Review date:** 2026-08-08

## DISC-008: Unclosed timestamp rescue is first-window only

- **Reference:** whisper.cpp's end-of-window rescue retains a timestamp-mode
  sequence with no closed timestamp when its current seek delta covers the
  remaining audio.
- **Our impl:** The native greedy and beam paths apply that has-timestamp-free
  rescue only at `seek_cs == 0`. Later windows must close a timestamp before
  their text can be retained; otherwise they follow the existing failed-window
  path.
- **Reason:** A short first-window streaming decode can legitimately emit
  `<|0.00|>` plus text and EOT, so removing the rescue entirely loses real
  speech. On a later short tail, however, the default `CHUNK_CS` seek delta
  makes coverage true before the model closes a timestamp. If the preceding
  window already emitted the overlapping speech, accepting that tail emits it
  a second time. The first-window boundary preserves the streaming case without
  adding transcript-deduplication heuristics.
- **Observed impact:** On canonical tiny.en plus tiled `jfk.wav` (three tiles),
  the unbounded coverage rescue produced eight occurrences of `country` in
  both the default and `max_context=0` cells; the exact oracle is six. This is
  a decoding-boundary defect, not a prompt-carry difference.
- **Resolution:** **ACCEPTED as a narrow native quality divergence.** The policy
  is shared by greedy and beam decoding and is pinned by opposite first-window
  and later-tail unit cells plus the model-level exact tiled-JFK gate.
- **Tests affected:**
  `src/native_engine/decode.rs::unclosed_window_rescue_is_first_window_only`
  and `gated_max_context_zero_disables_prompt_carry`.
- **Review date:** 2026-08-24
