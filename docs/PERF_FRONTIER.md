# Perf Frontier — actionable handoff for the next (owner-gated) optimization session

> Forward-looking playbook, not a log. The historical record is `PERF_LEDGER.md`
> (measured wins) and `NEGATIVE_EVIDENCE.md` (rejections + blockers). This file is
> the short answer to "what's left and exactly how to do it." Owned by swarm agent
> **BlackThrush**. Last updated 2026-07-12.

## ⚠ TOP PRIORITY IS NOW A CORRECTNESS BUG, NOT A PERF LEVER (2026-07-12)

The byte-exact **perf** envelope is exhausted (below), but this session surfaced a bigger issue and
worked it end-to-end. **This outranks every perf lever below.** Full write-up: `NEGATIVE_EVIDENCE`
2026-07-12 / `project_final_window_early_eot_bug`.

- **Bug:** shipping default **drops ~48% of long-form content on tiny.en** (bd-r0qd), reproduced on the
  in-repo `example_audio_track_01.mp3` — **two 30 s windows dropped** in TS mode
  (`fw transcribe --json`: gaps `[24.88-54.88]`, `[80.08-110.08]`). A "faster" long-form tiny.en is
  partly from decoding LESS (non-comparable).
- **Root cause:** greedy decode with **no temperature fallback** — a window that closes no timestamp
  (`result_len==0`) while carrying the prior-window prompt (prompt × int8 numerics → early `eot`) is
  dropped (empty, full `CHUNK_CS` advance).
- **Severity bound:** **tiny.en ONLY** — `large-v3-turbo` covers the full clip (no drops); its stronger
  decoder doesn't early-EOT. Quality-seeking (turbo) users unaffected.
- **Landed this session (all byte-exact / default-OFF):** `FW_RETRY_FAILED_WINDOW=1` (`1caba18`) retries
  a failed window once with the prompt cleared → **recovers real audio EXACTLY = whisper.cpp** (track01
  643→1301 chars); a `tracing::warn!` (`1d777f0`) surfaces the otherwise-silent drop (track01 warns 2×
  at seek 24.88/80.08); a `decode_to_wav` example (`e221630`) makes mp3s whisper-cli-readable.
- **CAVEAT keeping the retry OFF:** it drops the prompt entirely for the failed window, so on
  repetitive/tiled audio it re-transcribes covered content (jfk×3 239→379ch). Real speech is fine.
- **Owner action (pick one):** (a) implement the proper whisper.cpp **temperature fallback** (tracks
  `prompt_reset_since`, avoids even the minor tiled +1, covers non-prompt failure modes) in
  `transcribe_samples` — the correct superset; or (b) **flip `FW_RETRY_FAILED_WINDOW` default-on** —
  the case is now fully evidenced (measured, not asserted): fixes the ~48% drop; **more faithful to
  whisper.cpp** on both real audio (recovers exactly) and tiled audio (jfk×3 "country": wc 7, default 4
  DROPS, retry 8 — retry 3× closer to wc); **test-safe** (`FW_RETRY_FAILED_WINDOW=1 cargo test --lib
  native_engine` = 238/0); **cheap** (encode-reused, `f3d8550`); **safety-audited** (retry TS-only,
  pipeline no_ts-only ⇒ no desync). It is left OFF only because it reverses the deliberate greedy/temp-0
  design (an owner call), not for any measured risk. **Test-safety is now COMPLETE** (238/0 lib + the
  integration/conformance suites confirmed flag-agnostic by construction: `native_engine_e2e.rs`
  transcribes only `jfk.wav` = single-window, so the retry — which needs a carried prompt / multi-window
  — never fires; `conformance_harness.rs` validates replay/backend metadata, not live native decode).
  No test uses a multi-window native clip, so nothing exercises the retry outside the 238/0 lib suite.
- **RE-CONFIRMED current-code 2026-07-12** (fresh `fw`, tiny.en, `track01.wav`): default TS mode transcribes
  only **126 words**, `FW_RETRY_FAILED_WINDOW=1` restores **254 words** (== the no_ts full coverage measured
  the same day) — the retry recovers the dropped ~50% **exactly** to full coverage. The bug + fix both
  reproduce on current main. **PERF-MEASUREMENT-INTEGRITY consequence:** any tiny.en **TS** speed number is
  **non-comparable by default** (it decodes ~50% less), so a valid tiny.en-TS head-to-head MUST set
  `FW_RETRY_FAILED_WINDOW=1` (which restores the full decode ⇒ slower but correct). The headline's tiny.en
  figure (~1.93×) is safe because it was measured in **no_ts** mode (full 254-word coverage, verified).

## ⚡ Current-code headline vs whisper.cpp is ~1.4–2.3× across all model×modes (a WIN in every one) — "~1.2×" is STALE

**Quick reference (all 2026-07-12, matched threads — wc's best: turbo `-t 32`, tiny.en `-t 16`; total wall unless noted).** franken wins every cell; evidence + caveats in the bullets below. **⚠ These speed cells are fw-greedy vs whisper.cpp's DEFAULT (beam/best-of-5) — NOT matched-greedy. Matched-greedy (fair) is lower: turbo ~2.07× (small correction, encoder-dominated), tiny.en ~1.10× (the "1.93×" was mostly wc's beam-5 decode tax). Only the isolated encoder 2.29× is framing-independent. See the "CORRECTION 2026-07-13" in the Recommendation.**

| clip | model | mode | fw | whisper.cpp | ratio | coverage |
|---|---|---|---|---|---|---|
| jfk 11 s (encoder only) | turbo | — | ~1.40 s | ~3.21 s | **2.29×** | — |
| track01 124.5 s | turbo | no_ts | ~9.2 s | ~21.5 s | **~2.3×** | full (259 w) |
| track01 | turbo | seg-TS | ~12.2 s | ~21.6 s | **~1.77×** | full |
| track01 | turbo | word-TS (DTW) | ~12.2 s | ~20.9 s | **~1.71×** | full |
| track01 | tiny.en | no_ts | ~1.26 s | ~2.43 s | **~1.93×** | full (254 w) |
| track01 | tiny.en | seg-TS (retry¹) | ~1.95 s | ~2.72 s | **~1.39×** | full |
| **sjobs 840 s / 28 win** | turbo | no_ts | ~52 s | ~134 s | **~2.58×²** | fw clean / **wc loops** |
| sjobs 840 s | tiny.en | no_ts | ~8.4 s | ~17.3 s | **~2.06×** | both clean (valid) |

¹ tiny.en TS default drops ~50% (content-drop bug) → non-comparable; `FW_RETRY_FAILED_WINDOW=1` restores full
coverage (slower but correct). ² sjobs turbo ratio is muddied *in franken's favour* — wc degrades into greedy
repetition loops + drops ~40%; fw stays clean. Whole-pipeline rows are load-sensitive on this shared box
(the ratio can compress under contention); the isolated-encoder 2.29× and the fw-vs-fw self-improvement are
the confound-free anchors. **Net: franken is ~1.4–2.3× across every model×mode/clip, and at least as
faithful — cleaner than wc on the one clip where either engine degraded.**

**Realistic multi-window no_ts (the target workload), measured 2026-07-12:** turbo, `track01.wav`
(**124.5 s / 5 windows** — the *same clip* [[project_realistic_workload_dominated]] benched pre-int8),
`--no-timestamps`, matched `-t 32`, total wall:
- **fw ~9.2 s** (load ~4, output verified correct) vs **whisper.cpp ~21.5 s** (6 clean reps 21.1–22.8 s)
  = **fw ~2.3× FASTER**.
- **Rock-solid anchor (fw-vs-fw, no wc-build confound):** the memory documents fw's *pre-int8* time on
  this EXACT clip/threads/box (52fb1cb, 2026-07-04) as **20.34 s** ⇒ fw is now **2.22× faster than its
  own pre-int8 self**. wc is ~unchanged (24.78 s→21.5 s), so the realistic ratio jumped **1.22×
  (pre-int8) → ~2.3× (now)**. The gain compounds the full-int8 encoder (`a997f37`) + int8 decode
  (`FW_I8_BATCH_4COL`) + cross-window pipelining, all landed after 2026-07-04.
- **Caveat:** only 1 clean fw sample — the shared box turned hostile mid-run (the `fw` binary was
  evicted 3× by disk-pressure cleanup at 86% full, load spiked 4→34). wc reps were clean/consistent,
  and the fw-vs-fw self-improvement is confound-free, so the ~2.3× is directionally solid. This
  supersedes the "~1.68–1.8× no_ts" boilerplate (pre-int8).
- **DEFAULT TS mode (segment timestamps) — same clip, measured 2026-07-12 (6 interleaved reps, both engines
  stable even as load spiked 8→35):** **fw ~12.2 s vs whisper.cpp `-t 32` ~21.6 s = ~1.77× FASTER**
  (coverage-verified: fw 259 words, turbo doesn't drop). Lower than the ~2.3× no_ts because TS mode has
  **no cross-window pipelining** ([[project_window_pipelining_lever]]) — encode/decode serialize — so this
  is fw's honest *worst-case* headline. Still supersedes the stale "~1.2× ts" boilerplate.
- **tiny.en TS, done CORRECTLY (`FW_RETRY_FAILED_WINDOW=1`, so both engines cover the full clip):** fw
  ~1.95 s (254 words) vs whisper.cpp `-t 16` ~2.72 s = **~1.39× FASTER** (6 interleaved reps, load ~6–10).
  This is the LOWEST cell — TS has no pipelining AND the retry re-decodes the 2 recovered windows — but
  it's the only *valid* tiny.en-TS number (the default drops ~50%, non-comparable — see the content-drop
  entry above).
- **WORD timestamps (DTW), turbo track01:** fw `--timestamp-level word` ~12.2 s (259 words) vs
  whisper.cpp `-dtw large.v3.turbo -t 32` ~20.9 s = **~1.71× FASTER** (2 interleaved reps, both stable even
  at load 35–39). Notably fw word-ts (~12.2 s) ≈ fw segment-TS (~12.2 s) ⇒ **the DTW alignment overhead is
  NEGLIGIBLE** — confirms with measurement that `dtw.rs` (1185 lines) is sub-floor, no DTW speed lever.
  **Full current-code headline (track01, matched threads): turbo encoder 2.29× · turbo no_ts ~2.3× · turbo
  segTS ~1.77× · turbo wordTS ~1.71× · tiny.en no_ts ~1.93× · tiny.en TS (retry) ~1.39× — every model×mode
  ~1.4–2.3×, roughly double the pre-int8 "~1.2×" boilerplate, and franken WINS every one measured.**
- **GENERALIZES to a 2nd, much longer, different-content clip + a FAITHFULNESS win (2026-07-12):** the
  Steve Jobs iPhone keynote (`sjobs.wav`, **840.5 s / 28 windows**, turbo no_ts, `-t 32`, 2 reps, load
  40–50 both stable) — **fw ~52 s vs whisper.cpp ~134 s = ~2.58× FASTER** (even higher than track01, as the
  28-window encode weights the 2.29× encoder more). **BUT it's also a QUALITY win that muddies the raw
  ratio in franken's favour:** fw is **clean** (1855 words, max repeated 5-gram = 3 — a natural phrase),
  while **whisper.cpp DEGRADES into severe greedy repetition loops** ("you know, you know" ×19, "use a
  stylus" ×11, tail "We're going to ship it." ×4) and only 1090 words (~1.3 w/s, too low for a keynote ⇒
  ~40 % dropped). Both start byte-identical. So here whisper.cpp — *with* temp-fallback — loops/drops while
  franken's greedy int8 decoder stays faithful; **franken is faster AND cleaner** on this clip. (Note: this
  is the reverse of the tiny.en content-drop above — the greedy repetition/drop failure mode is
  clip-and-decoder-specific and afflicts BOTH engines on different audio.)
  - **MATCHED-GREEDY (2026-07-13): the speed is ~2.29× (not 2.58×), and the faithfulness win HOLDS — even
    STRENGTHENS — under wc-greedy.** The 2.58× above was fw vs wc-DEFAULT (beam-5); forcing wc greedy
    (`-bs 1 -bo 1`) on sjobs turbo: fw 48.7 s vs wc-greedy 111.4 s = **2.29×** (vs wc-default 119 s = 2.44×)
    — small correction, this clip is encoder-heavy (28 windows). And wc-greedy ALSO degrades: **1478 words,
    max-5gram-rep = 8, truncated "…state." tail** — worse than fw (1855 words, 5gram-rep = 3, clean) though
    not as bad as wc-beam-5 (1090 / 19). So franken is cleaner than BOTH whisper.cpp decode modes here, and
    ~2.29× faster matched-greedy.
  - **REFINED (2026-07-12): the wc sjobs loop is TURBO-specific, not general.** Same clip on **tiny.en
    no_ts** (`-t 16`, 2 reps): **both engines clean and full-coverage** — fw 1849 words vs wc 1852, both
    max-5gram-rep = 3, fw ends "…portrait to landscape." So this is a *valid same-content* comparison:
    **fw ~8.4 s vs wc ~17.3 s = ~2.06× FASTER** (consistent with track01's tiny.en 1.93×). So (a) the ~2×
    headline generalizes to a 2nd, much longer, different clip on BOTH models; (b) tiny.en no_ts is
    **full-coverage on 28 windows** (the content-drop bug stays TS-mode-only); (c) the turbo wc repetition
    loop above is a **large-model greedy failure mode on this clip**, NOT "wc always loops" — don't
    over-generalize it. franken is clean on this clip at both model sizes.
  - **FAITHFULNESS quantified (2026-07-12) — the "faithful" half of the claim:** on that clean tiny.en
    sjobs case (both full-coverage), fw vs whisper.cpp share a **1921-word longest-common-subsequence =
    98.0 %** of the transcript (normalized, `difflib`), fw 1959 / wc 1961 normalized words. So franken is
    **~2× faster AND ~98 % word-faithful to the reference** on a clean 840 s clip; the 2 % differences are
    minor (int8-vs-f16 numerics + segmentation). Caveat: this is fw↔wc *agreement* (no ground-truth WER on
    box), and on HARD cases franken is *more* faithful than wc (the sjobs-turbo loop), so 98 % is a floor,
    not a divergence. Net: the project's "faster **faithful** whisper" claim is now quantitative on both axes.
  - **The 2 % gap is BENIGN — no systematic franken bug (characterized 2026-07-12, `difflib` opcodes, 54
    hunks):** the diffs are (1) **punctuation attachment** ("while"↔"while,") — a *normalization artifact*,
    not a real word difference ⇒ true content agreement is even HIGHER than 98 %; (2) **segmentation/spacing**
    ("smartphones"↔"smart phones"); (3) **filler** ("so"↔"so,", "actually"); (4) a few genuine **tiny.en
    mishearings** ("pom"↔"palm" for *Palm*) — expected of the 39 M model, not a port defect. Both engines
    capture the same content incl. the iconic "an iPod, a phone, an internet communicator" line, and wc's
    `[applause]` markers. So franken's faithfulness is clean — the ~2 % is model-grade + formatting noise,
    NOT a divergence to fix.
  - **HONEST refinement — faithfulness is CLIP-DEPENDENT (88–98 %), gap dominated by FILLER not errors
    (2026-07-12):** the flagship **turbo on track01** (casual, disfluency-heavy tech-demo) agrees only
    **88.5 %** (LCS 239/270) — lower than the sjobs keynote's 98 %. But the diffs are **wc transcribing
    disfluencies fw suppresses** ("um, and, um,", "showcase, uh,", "it's, you know,") + a proper-noun
    ("cast"↔"cas" for *CAS*) + segmentation — i.e. the known **DISC-003 greedy-vs-beam filler** behavior
    (fw is *cleaner*, drops um/uh/you-know), NOT transcription errors. So don't quote a single faithfulness
    number: it's **88–98 % depending on how disfluent the audio is**, and the divergence is largely
    stylistic (verbatim-wc vs clean-fw), with core content captured by both.
    - **Reproducible baseline (`scripts/whisper_cpp_faithfulness.sh`, 2026-07-13):** the committed harness
      (uniform punctuation normalization) gives track01 **turbo 91.9 %** / **tiny.en 92.5 %** agreement,
      **both clean** (most-repeated 5-gram = 2, no loops) — slightly above the ad-hoc 88.5 % precisely
      because it strips the punctuation-attachment artifacts, confirming "true content agreement is higher
      than the raw number." Re-run `whisper_cpp_faithfulness.sh` for these; on jfk it reproduces the 0.0 %
      WER / 100 % anchor. These are the canonical, re-runnable faithfulness numbers.
  - **PROPER NOUNS (the faithfulness-CRITICAL axis) match (2026-07-12):** on track01 turbo, fw and wc agree
    exactly on **FrankenSearch, XF, Twitter, Franken, Franco, Daniel**; the only proper-noun difference is
    **CAS** (a 3-letter acronym → fw "Coding"/"cast"), a model-grade acronym miss, NOT int8 mangling. This
    matters because **FrankenSearch is the exact word [[project_turbo_encoder_dominates]] feared the shipped
    int8 encoder would mangle ("Frank at")** — it's correct. So the 88.5 % gap is filler + one acronym, NOT
    proper-noun errors: the content-critical tokens are faithful, confirming the shipped int8 encoder is
    proper-noun-safe on this clip vs the reference (not just vs a golden).
  - **DEFINITIVE: int8 adds ZERO proper-noun divergence vs the reference on EITHER model (2026-07-12).**
    On tiny.en track01, fw and wc render **every** proper noun identically — including the same mistake:
    both say **"Frankenstein"** for FrankenSearch (the 39 M model's limit, NOT franken-int8 — whisper.cpp's
    tiny.en makes the exact same error), and both agree on XF / Twitter / CAS / Daniel / Search. Turbo gets
    FrankenSearch *right* (both). So proper-noun accuracy is **purely model-size-dependent and matched
    exactly by the reference** at both sizes ⇒ the shipped int8 encoder costs nothing on the
    faithfulness-critical axis; the "Frank at" fear ([[project_turbo_encoder_dominates]]) is fully closed.
  - **GROUND-TRUTH ANCHOR — 0.0 % WER on jfk, BOTH models (2026-07-12).** Every other faithfulness number
    above is fw↔wc *agreement* (and wc itself degrades on hard clips), so the rigorous anchor is the one
    clip whose correct transcript is *known*: jfk.wav. fw vs the canonical words = **0.0 % WER (S=D=I=0,
    N=22)** on **large-v3-turbo AND tiny.en** — perfect, word-for-word, even on the 39 M model. So on the
    clip with ground truth, franken is exactly right; the 88–98 % fw↔wc gaps elsewhere are **wc's
    degradations or benign filler, not franken errors**. This is the strongest form of the "faithful"
    claim: measured against truth, franken is 0 % WER.

**Isolated encoder** (2026-07-12, idle box load ~7, `jfk.wav`, matched 32 threads —
whisper.cpp's *best*: `-t 16`=4382 ms, **`-t 32`=~3210 ms**, `-t 48`=3212, `-t 64`=5448 ms, the
all-core freq-throttle wall [[project_encoder_wall_is_clock_throttle]]):

| stage | fw (ms) | whisper.cpp `-t 32` (ms) | fw speedup |
|---|---|---|---|
| **encoder** (82% of compute, ts-independent) | **~1404** (`encoder_window` span, 3 reps 1420/1379/1414) | **~3210** (`encode`, 3 reps 3227/3231/3184) | **2.29×** |
| decode | ~223 | ~285 `batchd` + 36 `sample` | ~1.4× |
| compute (enc+dec+xkv+mel) | ~1666 | ~3538 | **2.12×** |
| total wall incl. load (single-shot) | ~2593 (compute + 927 load) | ~4398 (compute + 860 load) | **1.70×** |

**The encoder 2.29× is the solid number** (isolated per-engine spans, matched threads, ts-independent, the
dominant stage; fw ships full-int8 encoder, wc runs f16; both reps low-variance). **Why it's a real
regime change and not noise:** [[project_realistic_workload_dominated]] measured the *pre-int8* encoder as
**~tied** (franken 3.53 s/win vs wc 3.19 s/win); the `a997f37` full-int8 ship (2026-07-09) is what moved
franken's encoder ~3.5 s → ~1.4 s. So the ubiquitous `NEGATIVE_EVIDENCE` boilerplate "~1.2× ts / ~1.68–1.8×
no_ts vs whisper.cpp" is **STALE (pre-int8)** — treat those trailing ratios in older entries as such.

**CAVEAT (do not over-quote the total/compute rows):** [[project_realistic_workload_dominated]] proved the
*whole-pipeline* ratio on THIS shared box **swings 0.90–1.22× with ambient load 8→100** (franken's 32-thread
rayon oversubscribes harder than wc's OpenMP under contention) — a true whole-pipeline number needs a
*dedicated quiet* box. This run was load ~7 with stable reps, so the **isolated encoder span** is trustworthy,
but the compute (~2.1×) / total-wall (~1.7×) rows are load-~7 point estimates, not guaranteed floors. Also the
decode row is fw-TS vs wc-`-nt` (not matched). **Quote the encoder; treat the rest as directional.**

### bd-b4hp RESOLVED — franken's one documented LOSING case (tiny.en long-form) is now a ~1.9× WIN (2026-07-12)

[[project_realistic_workload_dominated]] documented **tiny.en long-form as ~1.73–1.84× SLOWER than
whisper.cpp** (bd-b4hp, 2026-06-29, **pre-int8**). Over two turns I (a) hypothesised 32-way encoder
over-threading and **MEASURED-REJECTED it** (no code change — the bench honours `RAYON_NUM_THREADS`:
`encoder_window_tiny` 8t=113.8 / 16t=78.7 / **32t=63.7 ms** — scales monotonically, 32t fastest, so fewer
threads would LOSE; this avoided a 4th entry in the threading 3-revert history
[[project_decode_overthreaded_rayon_lead]]); then (b) **re-measured the full transcribe current-code** now
that the box let `fw` survive:

```
tiny.en, track01.wav (124.5 s / 5 windows), --no-timestamps, total wall, 6 interleaved reps, load ~8:
  fw ~1.26 s   vs   whisper-cli -t 16 ~2.43 s   =   fw ~1.93× FASTER
```

**COVERAGE-VERIFIED (not the content-drop bug):** fw 254 words vs wc 250, both end "…that's it basically."
— fw transcribes the FULL clip, so the speed is real, not [[project_final_window_early_eot_bug]] decoding
less. So **the ONLY documented losing case is now a decisive WIN** — the pre-int8 1.84× *loss* → ~1.93×
*win* is a ~3.6× relative swing, from the cumulative int8 decode (`FW_I8_BATCH_4COL`) + int8 encoder
(`a997f37`) + pipelining wins landed since 2026-06-29. **franken now dominates BOTH turbo AND tiny.en
long-form.** bd-b4hp is closeable.

## Live full-pipeline span breakdown (measured 2026-07-12, real `fw transcribe`, not isolated benches)

`FRANKEN_WHISPER_PERF_SPANS=1 fw transcribe --input jfk.wav --no-persist` (single 11 s window):

| span | tiny.en (ms) | turbo (ms) | note |
|---|---|---|---|
| encoder_window | 80 | 1441 | per-window compute — dominates (ledger ceiling) |
| **model_weights** | **59** | **745** | **one-time load** (`from_ggml`: format-dequant→f32→i7/i8 requant→layout) |
| decode_loop | 48 | 231 | per-window token decode |
| model_parse | 14 | 182 | one-time ggml file parse |
| cross_kv | 9 | 36 | per-window cross-attn KV precompute |
| mel | 2.5 | 4 | per-window |
| backend_run (total) | 216 | 2666 | |

**This confirms the per-window compute ceiling** (encoder+decode dominate, both audited-at-ceiling)
**but reframes "load is sub-floor":** load (`model_parse`+`model_weights`) is **~35 % of single-shot
turbo wall time** (927 / 2666 ms) — sub-floor only for BATCH/long-file/server-resident workloads
(`load_resident` amortizes it to ~0), NOT for single-clip CLI / serverless / first-request latency.

- **CANDIDATE (byte-exact) — BUILT + MEASURED = WASH, reverted 2026-07-12.** Hypothesis: `quantize_mat_to_i7`
  (nn.rs:573) reads each output column **strided** (`w_t.data[i*out+o]`, stride `out`) — columns `o`/`o+1`
  share a cache line so the column loop re-reads it ~16× ⇒ memory-read-bound, so a **cache-blocked
  transpose-quant** (own a contiguous `BC=64`-column block, sweep rows, read each line ~once) would win.
  Implemented behind `FW_QUANT_BLOCKED`, **byte-identical** (unit-test-passed across partial blocks + live
  shapes; the tricky part was preserving the default **error-feedback** rounding — EF diffuses the residual
  along the contraction dim WITHIN each column, so a per-column `err[BC]` in the i-outer loop keeps it
  bit-exact). **But the `model_weights` span did NOT move: default ≈ blocked ≈ ~640 ms turbo (ABBA, both
  EF and non-EF).** So the 16× read amplification is REAL but NOT the bottleneck: the default path is EF,
  whose per-column serial `err`-chain is **latency-bound not bandwidth-bound**, and `model_weights` is
  anyway dominated by the ggml-Q-format **dequant + f32 convert + layout**, not the i7 re-quant. Reverted
  (byte-exact impl in git history; do not re-attempt cache-blocking this kernel — it's not memory-bound).
  **Lesson: "strided ⇒ memory-bound" is a HYPOTHESIS ([[project_nominal_vs_dram_bytes]]); a serial
  dependency (EF) or a dominating sibling stage can make the amplified reads free.** Load stays "at parity
  with wc"; cold-start latency is not an autonomously-movable lever here.
- **The OTHER load component — f16→f32 weight dequant — is BANDWIDTH-BOUND (probe, 2026-07-12).** The
  turbo weights are f16; `ggml::dequant_f16_parallel` (ggml.rs:623) converts them f16→f32 with a scalar
  per-element `half::f16::to_f32`. Hypothesis: SIMD `HalfFloatSliceExt::convert_to_f32_slice` (F16C
  `vcvtph2ps`, already used in the GEMV path) would beat it. `examples/f16_dequant_probe` (100M f16, 8
  workers, best-of-7): **SIMD 22.5 ms vs scalar 21.7 ms = 0.96× (WASH), 0/100M differing bits.** Both hit
  ~27 GB/s (read f16 + write 2× f32) ⇒ **bandwidth-bound, not compute-bound** — `half::to_f32` already
  autovectorises. So neither piece of `model_weights` is movable: the quant is EF-latency-bound, the
  dequant is bandwidth-bound. **The single-shot load path is fully characterized and at its floor.** (The
  memory's f16-conversion audit [[project_half_from_f32_software_no_site]] covered `from_f32` sites; this
  measures the previously-uncovered `to_f32` LOAD dequant. Don't re-dig either.)

## State: the byte-exact, autonomously-verifiable envelope is CLOSED

Everything that could be landed with a *quick, local, byte-exact* verify has been:

- **Peripheral IO/DB/export lane — fully optimized** (this session, 10 measured wins,
  all default-on, byte-exact): tty zlib `bufread`; `BufWriter` on every export writer
  (~40×) + sync incremental; streaming SHA-256 checksums (full + incremental export);
  64 KiB checksum read buffer (~1.16×); per-statement-savepoint skip on persist (1.48×)
  + sync import; DB-level N+1 → `IN (…)` on incremental export (1.32×); app-level N+1
  batch on routing history (**~14×**).
- **Transcription hot path — at its byte-exact ceiling**: encoder full int8 already
  default-ON for **both calibrated models — turbo AND tiny.en** (`calibrated_encoder_int8_model`
  = `tiny_en || is_large_v3_turbo`, shipped `a997f37`, ~1.47× encoder; `FW_ENC_ATTN_OUT_I8I32=0`
  kills); int8 logits head default-ON; `nn::quantize_act_i8_into` already AVX2-vectorized w/ correct
  round-half-away; SDPA poly-exp shipped for turbo; decode alloc-light rewrite landed. Measured/closed.
- **Flag audit (2026-07-12): every byte-exact `FW_*` win is already default-ON** — nothing dormant
  to flip. Verified default-ON: `FW_I8_BATCH_4COL`, `FW_I8_BATCH_2COL`, `FW_I7_M2N4`,
  `FW_I7_QKV_HEADMAJOR_ROWCO`, `FW_F16_BATCH_M2COL`, `FW_BATCH_GEMV_ROW_MORSEL`,
  `FW_SDPA_GATHER_CHUNKS` (=16), `FW_PERSIST_SKIP_STMT_SP`, `FW_STORAGE_BATCH_HISTORY`,
  `FW_SYNC_BATCH_{QUERY,IMPORT,SKIP_STMT_SP}`. The remaining **default-OFF** flags are all
  **lossy/quality** (NOT byte-exact → owner/WER-gated, cannot be autonomously flipped):
  `FW_CROSS_V_BLOCK`, `FW_DEC_EF`, `FW_ENC_INT8_ATTN_IN`, `FW_SIMD_EXP`, `FT_SDPA_POLY_EXP` (tiny.en).
- **Two rejections** kept the discipline honest: `load_run_details` scan (sub-floor),
  persist multi-row INSERT (regression). Unifying rule: **batching helps only when it
  cuts execution COUNT, not per-row work.**
- **Sweeps that found nothing** (so nobody re-runs them): the youtube batch pipeline
  already shares one engine across all videos (`transcribe_and_render(&engine, …)` — no
  per-video model-reload N+1); `Regex::new` in hot loops (none); raw-`File` write/read
  loops (all buffered or one-time SHA); `Vec::contains`/O(n²) in hot paths (none); stdout
  per-item emit (streaming-unsafe to buffer, sub-floor for batch dumps).
- **LLVM-leaves-perf antipattern sweep — CLOSED** (re-verified against current code
  2026-07-12, post the recent default-on flips): the four exploited scalar-hot-loop classes
  are all covered. **argmax** (index-tracking reduction, loop-carried `best_i` ⇒ won't
  autovec) is the ONLY one that needed hand-AVX2 — landed `argmax_idx` 5.10× byte-exact
  (`decode.rs:614`, `[[project_argmax_avx2_landed]]`); its *siblings* are NOT levers:
  **max/min folds** (`decode.rs:387` timestamp-rule `max_text_logprob`, `decode.rs:1941`
  lang-detect, softmax) already lower to `llvm.vector.reduce.fmax` (byte-identical, ~1.2–1.36×,
  sub-noise — ledger `7469`/`7478`/`7779`); **`.round()`** quant maps are AVX2'd
  (`encoder.rs:1228`, `nn.rs:2332`); **gather** (gelu) exhausted. No uncovered index-tracking
  hot serial loop exists (grep). Don't re-grep this vein.
- **Cross-attention K/V — verified fully optimized** (2026-07-12, the last per-window area not yet
  re-checked this session): the per-window K/V PROJECTION (`encoder_out @ Wk/Wv`, tq=1500) runs the
  dequant-once f32 sgemm (`cross_proj_f32_enabled` **DEFAULT-ON**, mod.rs:724; **2.25×** on turbo,
  golden-checked, `examples/cross_f16path_probe`); the per-TOKEN K/V read is **f16 by default**
  (byte-identical) with int8/block-wise variants gated for quality (`FW_CROSS_V_BLOCK`,
  [[project_cross_v_block_win]]). `cross_attn` is only ~4.4% of decode. No byte-exact cross lever.
  **With this, every per-window area (encoder int8/FLOP/SDPA/conv/LN, decode mlp/logits/qkv/self-attn/
  cross, mel) is personally re-verified closed this session, and the load path is floored (above) —
  the autonomous byte-exact frontier is empirically exhausted; remaining levers are owner/infra only.**

## ⚠ CORRECTION 2026-07-12: tiny.en FULL int8 is ALREADY SHIPPED (rows 1 & 2 below were STALE)

Verified against current code (`a997f37 "perf(native): default quality-safe encoder int8"`):
`calibrated_encoder_int8_model()` returns `tiny_en || is_large_v3_turbo`, so **tiny.en gets the
full quality-safe int8 encoder (q/k/v/fc1/fc2 i7 + attn.out i8, the ~1.47× lever) DEFAULT-ON** —
it was calibrated `2026-07-10`, not "uncalibrated/pending" as the old rows (and memory) claimed.
Empirically confirmed (prebuilt `fw`, tiny.en): unset **≠** `FW_ENC_ATTN_OUT_I8I32=0` (the f32
kill-switch) and unset **==** `=1` — i.e. the shipped default IS int8. `FW_ENC_INT8_FC1` is
therefore **inert in the default config** (branch precedence: the full-int8 branch runs first;
fc1-only is only reachable with `FW_ENC_ATTN_OUT_I8I32=0`). **This invalidates the a18fed2 "fc1-int8
WER-neutral proxy" evidence** — that transcript-diff compared default-int8 vs default-int8 (a no-op
flag), NOT f32 vs fc1-int8. See NEGATIVE_EVIDENCE 2026-07-12 for the full correction.

## Remaining levers — all need the model-bench + corpus-WER loop + owner sign-off

| lever | est. e2e | evidence in hand | why gated | validate before flip |
|---|---|---|---|---|
| ~~`FW_ENC_INT8_FC1` for tiny.en~~ **MOOT** | — | tiny.en already ships the strictly-more-aggressive FULL int8 (above); fc1-only is superseded/inert in the default config | n/a — not a lever | n/a |
| ~~tiny.en encoder int8 *calibration*~~ **DONE (shipped `a997f37`)** | ~1.47× encoder, LIVE | `calibrated_encoder_int8_model` includes tiny.en; policy `calibrated_model_budget_pass` (asserted by a unit test). Default-on, `FW_ENC_ATTN_OUT_I8I32=0` kills | not gated — shipped | — |
| **ToMe / layer-pruning** (encoder FLOP reduction) | large (turbo) | space mapped; tail-truncation already landed | changes output structurally | full WER + segment-timing corpus |
| **poly-exp variants / GPU** | — | poly-exp turbo shipped; GTX1070 = nouveau (no CUDA) | owner / infra | — |

## Import N+1 — DONE + flipped default-ON (no byte-exact lever remains)

The sync **export** N+1 is landed (`FW_SYNC_BATCH_QUERY`, ~1.32×). Its mirror, the **import**
path, was the last un-optimized IO site — byte-exact (no quality gate). **It is now COMPLETE.**

**CORRECTION 2026-07-12 (was stale): `FW_SYNC_BATCH_IMPORT` is DEFAULT-ON, not "default-OFF
pending a flip."** The runs/segments/events batch landed (`d2b5b14`/`8199711`/`40fbcdf`) and the
flip to default-on shipped in **`f38d83c` "flip FW_SYNC_BATCH_IMPORT default-ON — measured ~1.29×
import, byte-exact"** (verified against `sync.rs:56` = `… != Some("0")`, comment "**Default ON**",
`FW_SYNC_BATCH_IMPORT=0` kills). All three tables dispatch legacy vs batched through the SAME
`apply_{run,segment,event}_row` conflict logic (differing only in where `existing` comes from):
runs prefetch `WHERE id IN (…)`, composite tables prefetch `WHERE run_id IN (…)` + map by
`(run_id,idx)`/`(run_id,seq)`, each with an intra-chunk seen-map ⇒ byte-identical. Gate:
`sync::tests` 350/0 (now exercised through the batched path by default) +
`flush_{run,segment,event}_chunk_matches_per_line_reference` + full-CLI export→import A/B
byte-identical off-vs-on incl. the conflict/noop re-import path. **There is no pending byte-exact
lever — the soak+flip is already done.** The recipe + hazards below are retained as historical record.

- **Sites** (`src/sync.rs`): `import_table_runs` loop `SELECT … WHERE id=?1` **per line**
  (~:1202); `import_table_segments` `WHERE run_id=?1 AND idx=?2` (~:1384); `import_table_events`
  `WHERE run_id=?1 AND seq=?2` (~:1536). One query per JSONL line = N+1.
- **Recipe**: chunk the lines; per chunk collect keys → one `WHERE … IN (…)` → pre-fetch a
  `HashMap<key, full_row>`; process lines in original order against the map, applying the exact
  same identical-compare + `ConflictPolicy` (Reject/Skip/Overwrite) logic.
- **Hazard 1 — full row, not existence**: the per-line SELECT returns all columns for an 11-field
  identical-vs-conflict compare, so the map must hold full rows (not a `HashSet` of ids).
- **Hazard 2 — intra-chunk duplicate ids**: the per-line version's later duplicate SEES the
  earlier line's INSERT. A pre-fetch queried before any insert does not. Maintain a `seen` map
  updated on every insert/delete within the chunk so duplicate-id files stay byte-exact.
- **Composite keys**: segments/events key on `(run_id, idx)` / `(run_id, seq)`; if fsqlite lacks
  row-value `IN`, batch by `run_id` and index the map by the composite key.
- **Expected magnitude**: import is INSERT-dominated (the persist multi-row-INSERT reject proved
  per-row B-tree work isn't batchable), so batching only the SELECT setup nets **< the export's
  1.32×**. Real but modest.
- **Gate**: `sync::tests` round-trips + a NEW intra-chunk-duplicate-id test must stay byte-exact;
  put it behind a `FW_SYNC_BATCH_IMPORT` kill-switch/A/B arm mirroring `FW_SYNC_BATCH_QUERY`.

## Recipes (so the next session doesn't rediscover them)

- **Fast byte-exactness check, NO build** (~0.3 s/clip): prebuilt
  `/data/tmp/cargo-target/release/fw` + `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole` +
  `FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL=…/ggml-tiny.en.bin` +
  `FRANKEN_WHISPER_MODEL_DIR=…/models`, `fw transcribe --input <clip> --no-persist`,
  diff stdout with the `FW_*` flag off/on. Timing is NOT measurable this way (load-
  dominated); use it only to reject-or-gather byte-exact evidence.
- **Warm perf sizing** (needs ~6 min local build): `RCH_MIN_LOCAL_TIME_MS=999999999`
  `FRANKEN_WHISPER_MODEL_DIR=…` `cargo bench --bench native_engine_bench --
  encoder_window_tiny` (+ `e2e_tiny_jfk`, `decoder_token_step_tiny`), A/B the flag via
  external env on the same cached binary. **TWO GOTCHAS learned 2026-07-12:** (1) the
  `e2e_tiny_jfk` A/B needs an **idle box** — on a loaded host wall-clock swings ~22% (load
  2→26 mid-run) and buries any sub-15% lever. (2) Do NOT run `f32 then flag` each rep — the
  flag arm always runs second, so a warming/contending machine makes it *look* slower
  (this exact confound made `FW_ENC_INT8_FC1` look like an e2e regression when it likely
  isn't). Use ABBA / randomized order, and note both `encoder::forward` and the production
  `encoder::forward_from_full_mel_window` **ignore the thread-hint** (same ft rayon pool),
  so `encoder_window` IS representative of the encode *work* — the divergence was
  measurement, not a real code-path difference.
- **Corpus WER vs the original**: `legacy_whispercpp/whisper.cpp/build/bin/whisper-cli`
  is the reference (not on `$PATH`); tiny.en + turbo models + jfk/other clips live in
  `legacy_whispercpp/whisper.cpp/models/` and `sample_audio_files/`, `tests/fixtures/audio/`.
  **BASELINE ESTABLISHED + BLOCKER SHARPENED (2026-07-12):** the SHIPPING int8 `fw` is
  **byte-identical (normalized) to whisper.cpp on jfk** (`whisper-cli -m tiny.en -nt -t 8` vs
  `fw transcribe --no-persist`) — so the default int8 engine is faithful on real speech; the
  gated-lever WER baseline on jfk is **≈0**. **mp3-corpus limitation RESOLVED (`decode_to_wav` example, `e221630`):** `whisper-cli` can't read
  `.mp3`, but `cargo run --release --example decode_to_wav -- <mp3> <wav>` (built-in symphonia, no
  ffmpeg) makes any mp3 whisper-cli-readable. So `example_audio_track_01.mp3` now HAS a reference.
  **ENCODER-INT8 PROPER-NOUN WER (2026-07-12, positive):** the concern gating the shipped int8 was
  proper-noun safety — MEASURED on the track01 proper-noun clip (turbo): fw (shipping int8) vs
  whisper.cpp turbo = **271 vs 283 words (~96%), ~22 diff lines (~4% word variance)**, and the proper
  nouns are CORRECT (FrankenSearch, Twitter, XF, CAS×2, Daniel). No content-drop (turbo covers the full
  span). ⇒ the shipped int8 encoder is **proper-noun-faithful** on this clip; the ~4% is normal
  cross-quant/decode variance, not a quality bug. Still only 2 real-speech clips (jfk + track01) on box;
  a full corpus-WER for the remaining gated levers needs the **owner to supply more diverse speech**,
  but the mp3-corpus tooling is now in-tree and the int8 encoder is validated proper-noun-safe on the
  one proper-noun clip available.

## Recommendation

**⚠ CORRECTION 2026-07-13 — the headline speed numbers used whisper.cpp's DEFAULT decode (beam/best-of-5),
which is NOT matched-greedy. The fair numbers are lower — dramatically for tiny.en.** franken is
**greedy-only**; whisper.cpp's *default* is beam-5/best-of-5 (~5× the decode work AND higher quality). The
`whisper_cpp_ab.sh` harness correctly forces wc to greedy (`-bs 1 -bo 1`); my ad-hoc headline runs did NOT,
so they compared fw-greedy vs wc-DEFAULT-beam and partly credited franken for wc doing more work. Measured
both (track01, no_ts, matched threads):

| clip | fw / wc-**default** (old headline) | fw / wc-**greedy** (FAIR) |
|---|---|---|
| turbo   | 2.30× | **2.07×** (encoder-dominated ⇒ small correction; the int8 encoder is beam-independent) |
| tiny.en | 1.87× | **1.10×** (decode-dominated ⇒ HUGE correction — the "1.93×" was mostly wc's beam-5 tax) |

**So the honest speed claim is: matched-greedy, franken is ~2.1× on turbo but only ~1.1× on tiny.en.** The
one framing-independent, robust win is the **isolated encoder (2.29×, beam-independent)**. The old
"~1.4–2.3× every mode" numbers are the *vs-wc-out-of-box-default* framing (valid as a UX comparison, but wc
beam-5 is slower AND more accurate, so it's not a pure speed number). **Faithful** half stands: **0.0 % WER
vs ground truth on jfk (both models)**, ~92 % agreement on real speech (gap = filler/stylistic, proper nouns
match, zero int8 divergence; franken cleaner where wc loops). The byte-exact **perf frontier is CLOSED**;
the loop has run to completion — **redirect** to the owner-scoped items below or the correctness decision
(`FW_RETRY_FAILED_WINDOW`). (tiny.en/turbo TS/word-TS + sjobs headline cells above still carry the
vs-default numbers — apply the same matched-greedy correction; only the encoder 2.29× is framing-independent.)

**Validation COVERAGE + what's resource-blocked (so the owner knows what to supply to extend it):** the
numbers above cover **English** speech on the **two real on-box models** — `tiny.en` (74 MB, English-only)
and `large-v3-turbo` (1.5 GB, multilingual-capable). Two axes remain UN-measured and are **blocked on
missing on-box assets, not effort**: (1) **multilingual** — turbo *can* do it, but there is **no non-English
audio on box** (all clips — jfk / track01 / sjobs / test_10s_speech — are English); (2) the **intermediate
models** (base / small / medium) — only 562 KB *test stubs* are present, no real weights. To extend the
faster-and-faithful validation to those, the owner needs to drop in non-English speech clips and/or the real
base/small/medium ggml models; then the same harness (`decode_to_wav` → interleaved `fw` vs `whisper-cli`,
WER + word-agreement) applies unchanged.

Pause the autonomous *byte-exact* loop — further ticks only re-measure settled ground or
land sub-floor micro-levers the ledger reverts. **Rows 1 & 2 of the table above are no longer
the start point** (both MOOT/DONE per the §CORRECTION: `FW_ENC_INT8_FC1` is inert under the
shipped full int8; tiny.en calibration shipped `a997f37`). And the encoder FLOP-reduction row
is **measured dead on CPU** — `NEGATIVE_EVIDENCE` closes all three redundancy axes with data:
DEPTH (layer-pruning fatal at skip-1: `=31` mangles proper nouns + repetition-loops track01 (−27% words) though it's jfk-byte-identical; `=30` breaks even jfk — `7092` + 2026-07-12 update), SEQUENCE (ToMe frames not mergeable,
`4518`), SPECTRAL (weights near-full-rank, `4640`); Nyström/CountSketch/PQ/low-rank/Strassen all
rejected (`4552`). So the genuinely-remaining levers are **owner/infra only**: (1) a **Linux GPU
compute stack** (GTX 1070 is on nouveau → no CUDA/OpenCL/Vulkan — the encoder GEMM/SDPA is the
sole out-of-crate lever); (2) a **cheap multilingual DRAFT model** to unlock speculative decode
(verify amortization R(K)≈3.7× de-risked, but the draft-model-FREE **layer-skip self-draft is
MEASURED-DEAD** — `FW_DRAFT_ACCEPT_LAYERS` probe, NEGATIVE_EVIDENCE `6675`/`252`: k-of-4-layer early-exit
argmax matches the full-model argmax only **0% / 0% / 11.8%** (k=1/2/3) vs the 47% / 65% / 82% break-even,
because the distilled 4-layer decoder's layers are all load-bearing — so the drafter MUST be a real
separate model with a smaller logits head, not a self-skip); (3) **AVX-512-VNNI hardware** (int8 encoder GEMM
is 0.89× on this AVX2-no-VNNI box). No autonomously-landable byte-exact perf lever remains
(re-verified against current code 2026-07-12: encoder int8 maximal, import N+1 default-on, IO
swept, fresh shipped-tiny.en encoder profile = exp `__expf_fma` ~9% [poly-exp owns it: turbo-on,
tiny.en regressed-off] + rayon `__sched_yield` [contention-inflated] + int8-GEMM bulk — no new
hot spot).
