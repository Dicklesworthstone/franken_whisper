# Claim Coverage Audit — how much of our claimed ground rests on a live incumbent

**Date:** 2026-07-30, re-verified at HEAD `ec96916` 2026-07-31 ·
**Auditor:** MagentaMeadow (cc lane), re-run by BlackThrush · **Bead:** bd-b4hp

**Reproduce with `python3 scripts/claim_coverage_audit.py [--detail]`.** The
first publication of this audit stated its method in prose but shipped no
script, so the headline could not be re-derived by a reader. It now can be, and
the re-run at HEAD reproduces 44 / 3 / 41 unchanged.

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

**Arithmetic correction (2026-07-31).** This section originally read "17" and
"22", which sum to **39**, not 41 — two rows fell through the hand split. The
script now derives the split by surface keyword and prints its per-row
assignment (`--detail`), so the parts sum to the whole by construction. The
corrected counts are **18** and **23**. Both extra rows land in the two buckets
below; neither changes the headline 3-of-44, and neither is user-facing.

**18 — no incumbent arm can exist.** These are on surfaces `whisper.cpp` does
not have at all: the Bayesian backend router, the diarizer, SQLite run storage,
SRT/VTT export, NDJSON streaming, the YouTube renderer, CLI request building.
There is nothing to compare against; `whisper-cli` implements none of it. These
are **permanently unconvertible**, and the correct remedy is labelling, not
measurement — they must read "self-speedup / maintenance" and must never be
quoted as competitive wins.

**23 — convertible in principle.** These are on the shared ASR surface: mel
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
   *Status: the exact frozen no-timestamp cell still reproduces the default-
   context mismatch: native/whisper.cpp produce 287/431 words and WER 0.479167.
   The matched predicate is cleared only by explicit no-context decoding on
   both arms. With native `DecodeParams.max_context=Some(0)` and
   `whisper-cli -mc 0`, both perform 319 single-token decode steps over five
   encoder windows and produce 279/279 words with 3 edits (WER 0.010753).
   Commit `4c4aaef` pins that parameter in the shared contract; a strict-RCH
   artifact and exclusive quiet-host timing verdict still remain.*
2. **`whole_job` scope — coverage is ZERO, not "tiny.en only".** An earlier
   revision of this row said the July 30 124.5 s / 300 s tiny.en ratios were
   `FW_BENCH_SCOPE=whole_job`. That is not what the ledger records, and the
   correction matters because it changes the item from "extend an existing
   measurement to turbo" to "no whole-job incumbent ratio exists at all":
   - `FW_BENCH_SCOPE`, `whole_job`, and `whole-job` appear **nowhere** in
     `docs/PERF_LEDGER.md` (zero grep matches across the file).
   - The July 30 row states its own scope at `PERF_LEDGER.md:174`: *"Transcribe
     work **excluding one-time model load on both sides**… franken is timed
     in-process with the model resident."* It then explains that full process
     wall was deliberately **not** used, because it would pit `whisper-cli`'s
     thin inference binary against franken's orchestrator.
   All three supported ratios (07-27, 07-28, 07-30) are therefore
   `transcribe_only`. Model load is ~35% of single-shot turbo wall, so this is
   not a rounding correction on the published number — it is a scope a real
   user's single-shot job includes and we have never certified against the
   incumbent.
3. **The beam-search path — an unconverted *negative*.** The ledger's own
   honest row records native `retry+beam5` at **~3–4.6× SLOWER** than
   `whisper-cli -bs 5` on long-form. Converting this publishes a loss, which is
   exactly why it should be converted rather than left in a ledger: our public
   claim is greedy-only, and a user who turns on beam gets the opposite of the
   README's promise.
4. **The 23 shared-surface kernel rows.** Convert at the engine level (items
   1–2), not individually.
5. **The 18 no-arm rows.** Relabel as maintenance; do not attempt conversion.

## Method

Implemented in `scripts/claim_coverage_audit.py`; run it to reproduce every
number on this page.

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

Both hand-removals are now rules in the script rather than judgement calls:
a ledger row's title starts with its ISO date, so document structure
("Levers", "Result classes") is excluded structurally (2 sections); and a row
that self-labels `NON-CAMPAIGN` is excluded from the supported set even though
it does quote a live incumbent run. Without the date rule the population reads
46, which is why the two published totals must be quoted together with the
rule that produced them.
