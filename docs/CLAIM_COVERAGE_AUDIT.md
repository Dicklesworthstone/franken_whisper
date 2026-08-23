# Claim Coverage Audit — how much of our claimed ground rests on a live incumbent

**Date:** 2026-07-30, re-verified after the flagship row on 2026-07-31 ·
**Auditor:** MagentaMeadow (cc lane), re-run by BlackThrush · **Bead:** bd-b4hp

> **Addendum 2026-08-23.** The 45 / 4 / 41 count below is a snapshot as of
> 2026-07-31 and predates the newest admissible row it helped force: the
> large-v3-turbo whole-job ratio improved to **2.992045×** (ToMe R=500,
> same-invocation incumbent, dual A/A nulls in `[0.98, 1.02]`, CI95 excluding
> 1.0 — `docs/PERF_LEDGER.md`). The README headline now quotes that newer
> row; note its WER figure is the **control arm's** cross-engine WER
> (0.010753), while the shipped R=500 arm's own cross-engine WER on the same
> clip was higher (0.025090). Re-run `scripts/claim_coverage_audit.py` at
> read time rather than trusting either snapshot.

**Reproduce with `python3 scripts/claim_coverage_audit.py [--detail]`.** The
first publication of this audit stated its method in prose but shipped no
script, so the headline could not be re-derived by a reader. It now can be.
After the first admissible flagship whole-job row, the current result is
45 / 4 / 41.

Fleet policy under audit: **a perf KEEP requires a vs-incumbent ratio**, and a
competitive result requires the legacy incumbent to run *side-by-side with
franken in the same harness invocation* (README §"Campaign wins use the actual
incumbent"). This audit was not requested by a reviewer; it follows the
priority order, and it is published whether or not the number flatters us.

## The one-line answer

| | count |
|---|---:|
| **Perf KEEP claims held** (rows asserting a speed ratio) | **45** |
| …carrying a **live same-invocation vs-incumbent ratio** | **4** |
| …**not** carrying one | **41** |

**4 of 45.** Roughly **91% of our perf-claim ground rests on self-speedups or
non-interleaved comparisons**, not on the incumbent running beside us.

The four supported claims all come from `examples/incumbent_ab.rs`:

| date | cell | ratio |
|---|---|---:|
| 2026-07-31 | large-v3-turbo, 124.5 s, whole job | 2.264127× |
| 2026-07-27 | tiny.en segment-timestamps, live whisper.cpp | 1.415× |
| 2026-07-28 | clean-start, independently split n=31 re-certification | 1.479272× |
| 2026-07-30 | current-source tiny.en text, 124.5 s / 300 s | 1.518913× / 1.5121× |

## The 41 unsupported claims split into two genuinely different problems

**Arithmetic correction (2026-07-31).** This section originally read "17" and
"22", which sum to **39**, not 41 — two rows fell through the hand split. The
script now derives the split by surface keyword and prints its per-row
assignment (`--detail`), so the parts sum to the whole by construction. The
corrected counts are **18** and **23**. Both extra rows land in the two buckets
below; neither changed the then-current 3-of-44 headline, and neither is
user-facing.

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

The README's quantitative competitive assertions are the **2.26×
large-v3-turbo whole-job** row and the **1.52× / 1.51× tiny.en transcribe-only**
rows. **All are backed by the 4 supported claims.** No unsupported claim is
currently user-facing — the 41 live in ledgers a user never reads. That is the
one genuinely reassuring finding here, and it is why the queue below is
ordered by exposure rather than by ratio size.

## Ranked conversion queue (by how load-bearing, not by how large)

1. **RESOLVED — `large-v3-turbo` vs-incumbent ratio.** The frozen matched
   no-context cell now has an admissible whole-job result:
   `2.264127×`, CI95 `[2.244706, 2.277732]`. Both arms performed 319
   single-token decode steps over five encoder windows and produced 279/279
   words with 3 edits (`WER=0.010753`). The exact work, actual-thread,
   identity, frequency, host-exclusivity, load, corrected-null-median, and
   retained 2x-null-margin gates all passed.
2. **RESOLVED — `whole_job` scope.** The July 31 flagship row is the first
   supported whole-job ratio: both fresh processes include startup, model and
   audio I/O, inference, serialization, and teardown. The prior audit was
   correct that the older tiny.en ratios were not whole-job:
   - `FW_BENCH_SCOPE`, `whole_job`, and `whole-job` appear **nowhere** in
     the pre-flagship `docs/PERF_LEDGER.md`.
   - The July 30 row states its own scope: *"Transcribe
     work **excluding one-time model load on both sides**… franken is timed
     in-process with the model resident."* It then explains that full process
     wall was deliberately **not** used, because it would pit `whisper-cli`'s
     thin inference binary against franken's orchestrator.
   The older supported ratios remain `transcribe_only`; the new turbo row does
   not retroactively change their scope.
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
row, a doc-structure header) were removed by inspection. The reported 4 is the
count that survives that filtering.

Both hand-removals are now rules in the script rather than judgement calls:
a ledger row's title starts with its ISO date, so document structure
("Levers", "Result classes") is excluded structurally (2 sections); and a row
that self-labels `NON-CAMPAIGN` is excluded from the supported set even though
it does quote a live incumbent run. Without the date rule the population reads
46, which is why the two published totals must be quoted together with the
rule that produced them.

---

**Addendum 2026-08-23.** Item 3 of the conversion queue (publishing the beam
path's negative result) remains open: the ledger records native
`retry+beam5` at roughly 3–4.6× slower than `whisper-cli -bs 5` on long-form,
and no engine-level conversion has landed since this audit. The greedy-only
public claim is unchanged and still matches the shipped default
(`FW_BEAM_SIZE` unset ⇒ greedy).
