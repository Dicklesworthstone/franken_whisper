# Claim Coverage Audit — how much of our claimed ground rests on a live incumbent

**Date:** 2026-07-30 · **Auditor:** MagentaMeadow (cc lane) · **Bead:** bd-b4hp

Fleet policy under audit: **a perf KEEP requires a vs-incumbent ratio**, and a
competitive result requires the legacy incumbent to run *side-by-side with
franken in the same harness invocation* (README §"Campaign wins use the actual
incumbent"). This audit was not requested by a reviewer; it follows the
priority order, and it is published whether or not the number flatters us.

## The one-line answer

| | count |
|---|---:|
| **Perf KEEP claims held** (rows asserting a speed ratio) | **44** |
| …carrying a **live same-invocation vs-incumbent ratio** | **3** |
| …**not** carrying one | **41** |

**3 of 44.** Roughly **93% of our perf-claim ground rests on self-speedups or
non-interleaved comparisons**, not on the incumbent running beside us.

The three supported claims, all from `examples/incumbent_ab.rs`, all **tiny.en**:

| date | cell | ratio |
|---|---|---:|
| 2026-07-27 | tiny.en segment-timestamps, live whisper.cpp | 1.415× |
| 2026-07-28 | clean-start, independently split n=31 re-certification | 1.479272× |
| 2026-07-30 | current-source tiny.en text, 124.5 s / 300 s | 1.518913× / 1.5121× |

## The 41 unsupported claims split into two genuinely different problems

**17 — no incumbent arm can exist.** These are on surfaces `whisper.cpp` does
not have at all: the Bayesian backend router, the diarizer, SQLite run storage,
SRT/VTT export, NDJSON streaming, the YouTube renderer, CLI request building.
There is nothing to compare against; `whisper-cli` implements none of it. These
are **permanently unconvertible**, and the correct remedy is labelling, not
measurement — they must read "self-speedup / maintenance" and must never be
quoted as competitive wins.

**22 — convertible in principle.** These are on the shared ASR surface: mel
projection, GELU, the resampler, the i7/f16 GEMV kernels, beam KV reuse and
logits batching, the tiny.en no-carry decode policy. An incumbent arm is
*possible* for each. But most are **sub-lever kernels** whose honest incumbent
comparison is the whole-engine ratio they roll up into, not a per-kernel arm —
converting them individually would manufacture 22 ratios out of 1 real one.
The defensible conversion is at the engine level, which is the queue below.

## Also inventoried, and deliberately not counted as failures

- **30 non-perf KEEPs** — FEATURE / VALIDATION / ROBUSTNESS rows (all 10 ggml
  quant formats, `initial_prompt`, `--beam-size`, multilingual auto-detect,
  degenerate-input handling). They assert **no ratio**, so no incumbent arm is
  owed. Counting these as "unsupported" would inflate the failure in the
  opposite direction, which is its own dishonesty.
- **13 excluded** — rows self-labelled BLOCKED / NO VERDICT / REJECTED /
  "not a perf KEEP". They already decline to claim anything.

## Public exposure: the load-bearing claims are the supported ones

The README's only quantitative competitive assertions are the **1.52× / 1.51×
tiny.en no-timestamp** rows (lines 17, 34–35, 2624–2625). **Both are among the
3 supported claims.** No unsupported claim is currently user-facing — the 41
live in ledgers a user never reads. That is the one genuinely reassuring
finding here, and it is why the queue below is ordered by exposure rather than
by ratio size.

## Ranked conversion queue (by how load-bearing, not by how large)

1. **`large-v3-turbo` vs-incumbent ratio — ZERO coverage, flagship model.**
   Every supported claim is tiny.en. The README sells "a real in-process
   pure-Rust Whisper engine"; a user running the flagship turbo model has **no
   supported number at all**. Highest exposure of anything in this queue.
   *Status: the blocking clause was cleared on 2026-07-30 — the matched-greedy
   turbo comparator now measures **WER 0.034483** against the 0.10 gate on
   track01 (261 ref words vs 257, 9 edits), so the 287-vs-431 word divergence
   that aborted the cell does not reproduce at HEAD. Only a quiet host remains.*
2. **`whole_job` scope.** All three supported ratios are `transcribe_only`,
   which excludes one-time model load. A real user's whole job includes it, and
   load is ~35% of single-shot turbo wall. The scope exists in the harness and
   is unmeasured against the incumbent.
3. **The beam-search path — an unconverted *negative*.** The ledger's own
   honest row records native `retry+beam5` at **~3–4.6× SLOWER** than
   `whisper-cli -bs 5` on long-form. Converting this publishes a loss, which is
   exactly why it should be converted rather than left in a ledger: our public
   claim is greedy-only, and a user who turns on beam gets the opposite of the
   README's promise.
4. **The 22 shared-surface kernel rows.** Convert at the engine level (items
   1–2), not individually.
5. **The 17 no-arm rows.** Relabel as maintenance; do not attempt conversion.

## Method

`docs/PERF_LEDGER.md` + `docs/NEGATIVE_EVIDENCE.md` parsed into `## `-headed
sections; a section counts as a perf KEEP when it carries a KEEP/WIN verdict
**and** asserts a speed ratio. Same-invocation status requires an incumbent
token (`whisper.cpp` / `whisper-cli` / `incumbent`) **co-occurring** with
interleaving language (`same invocation`, `side-by-side`, `incumbent arm`,
`INCUMBENT_AB_`, `incumbent_bin_sha256`). Franken's own order-alternating A/A
interleaving does **not** count — an early pass over-counted 14 supported
claims precisely because it matched franken-vs-franken interleaving, and three
further false positives (a row self-labelled NON-CAMPAIGN, a harness-contract
row, a doc-structure header) were removed by inspection. The reported 3 is the
count that survives that filtering.
