# Ledger Resurrection Audit — franken_whisper

**Campaign:** `perf-campaign-20260725`, Fleet-Wide Meta-Lever #1.
**Lane:** cc / STRUCTURAL (YellowKite), Lane L under the 2026-07-25 allocation addendum.
**Source audited:** `docs/NEGATIVE_EVIDENCE.md` (709 entries, 25,780 lines).
**Revision 2 (2026-07-25):** rewritten to adopt **frankenfs's taxonomy verbatim**
(`/data/projects/frankenfs/docs/LEDGER_RESURRECTION.md`) per the fleet broadcast.

A REJECT row is **VOID** when the measurement *could not have detected the lever* — as
opposed to detecting it and finding it absent.

---

## 0a. Correction to revision 2 — the population was half its true size

Building the enforcement guard (broadcast 2) exposed a counting error in this
document. Revision 2 audited **139 rows whose header contains "REJECT"**. But
this repo does not have a verdict *column* — rejections live in prose titles, and
a large fraction never use the word. The `int4 mlp_0` family, for instance, is
closed under *DEAD*, *CLOSED*, *FALSIFIED* and *NEGATIVE* and says "REJECT"
nowhere.

Counting the vocabulary this repo actually uses
(`REJECT|DEAD|CLOSED|FALSIFIED|NO-SHIP|DO-NOT-RETRY|NEGATIVE`, dated rows only):

| | revision 2 | **corrected** |
|---|---:|---:|
| rejection-verdict rows | 139 | **277** |
| rows recording no decidability evidence | 82 | **99** |
| void rate | 59% | **36%** |

The corrected rate is *lower* because the widened population brings in many rows
that do record evidence — not because the standard was relaxed. It is measured
with the calibrated marker set now enforced by `tests/ledger_integrity.rs`
(word-boundary matched, with the false-positive traps in §0b removed), which is
stricter than revision 2's hand-rescue pass. **Treat 277 / 99 / 36% as this
repo's figures.**

## 0b. Two false-positive traps, in case other repos screen the same way

Both were caught while calibrating the guard, and both would have silently
inflated a "clean ledger" claim:

- **`"wer"` matches inside *were*, *lower*, *answer*.** Substring matching made
  the first draft of the guard pass **96%** of historical rows. Word-boundary
  matching is mandatory.
- **`"byte-identical"` is a claim of exactness, not an accuracy refutation.** A
  row reading *"byte-identical but 1.02×, rejected"* would have passed on it
  while recording no null at all. Only `non-byte-exact` — an actual refutation —
  counts. Likewise bare `"slower"` (54 of 169 rows, nearly all prose) and bare
  `"allocation"` were dropped in favour of a numeric `≤ 0.90×` test and specific
  phrases like `"allocations unchanged"`.

## 0. Correction to revision 1 — I audited the wrong class

Revision 1 of this document concluded that *"this repo does **not** have frankenlibc's
39-of-93 void rate; the void vein is narrow, old, and concentrated in one week"* and
reported **4 VOID**. That conclusion was wrong, in two compounding ways:

1. **I hand-audited only the 16 CV-gated rows** — because the campaign doc predicted the
   CV gate would be the dominant void class. The fleet broadcast has since corrected
   that prediction: CV is *rare* (frankenfs 4 of 219; **3 of 82 here**). By aiming the
   hand-audit at the rare class I sampled ~11% of the REJECT population and missed the
   epidemic entirely.
2. **I generalised "recent rows are rigorous" from the rows that have nulls.** Twenty
   rows record an A/A null and they are genuinely good. There are **139** REJECT rows.
   Inferring the ledger's health from its best 14% was a selection error on my part.

Corrected headline: **~82 of 139 REJECT rows (59%) are VOID**, and — exactly as the
broadcast says — the epidemic is **VOID-NONULL**, not VOID-CV.

---

## 1. Taxonomy (frankenfs, adopted verbatim)

| Class | Meaning | Sound? |
|---|---|---|
| `VALID-PROFILE` | Rejected before any source edit, on a named profile frame with non-zero self-time and a computed Amdahl ceiling. | ✅ |
| `VALID-MECHANISM` | No A/A null recorded, but refuted on a *counted* mechanism — instructions/cycles/syscalls/allocations/faults unchanged. A null control cannot change "no work was removed". | ✅ |
| `VALID-AB` | A/B run with a recorded A/A null; the claimed effect sits inside that null. | ✅ |
| `VOID-CV` | An A/B ran, and the row was killed **only** by a `cv < 5%` gate — unreachable on this hardware. | ❌ |
| `VOID-ZEROSELF` | The target frame had ~0% self-time in the profile the bench actually exercised. | ❌ |
| `VOID-NONULL` | An A/B ran, was rejected on a near-1.0 wall ratio, and recorded **no** A/A null control and no counted mechanism. Cannot distinguish lever from harness. | ❌ |

### Proposed 7th class — `VALID-ACCURACY` (franken_whisper-specific; offered to the fleet)

This repo's contract is **transcript exactness**, not throughput. A large share of its
REJECTs never made a speed claim at all: they were refuted on WER, faithfulness,
byte-exactness, or numerical safety. **A speed null control is meaningless for those
rows**, in exactly the way `VALID-MECHANISM` describes — the refutation does not rest on
a wall ratio, so the absence of a null cannot void it.

Examples adjudicated by hand: *"REJECTED on ACCURACY — Nyström low-rank encoder
self-attention"* (L8382); *"encoder `attn.out` int8 is NOT safe"* (L8331); the three
2026-07-04 quantisation rows rejected because they *"help track01 but REGRESS"* another
clip (L9170/9204/9234).

Two adjacent shapes were rescued on the same logic — a null cannot rescue them either:

- **Large-magnitude regression.** L8265 fused-wide encoder int8 QKV at **0.439×**
  (2.3–2.7× *slower*); L8474 product-quantised GEMM at **0.40×**. No plausible null floor
  spans a 2.5× loss.
- **Self-caught harness defect.** L8402 int8 GEMV register-row-blocking, rejected by its
  own author as a *"warm-bench trap"* — the row identified its measurement as invalid,
  which is the audit conclusion, not a defect in it.

If the fleet accepts this class, other repos with correctness-first contracts
(frankensqlite, frankenlibc) should re-screen for it before publishing a void rate.

---

## 2. Counts

Screen = `awk` pass over all 709 entries (script inputs in the session scratchpad).
**The screen is triage, not a verdict** — see §4 for exactly how far hand-adjudication got.

| Metric | Count |
|---|---:|
| Ledger entries parsed | 709 |
| **REJECT verdict — audited** | **139** |
| `VALID-AB` | 20 |
| `VALID-MECHANISM` | 21 |
| `VALID-PROFILE` | 1 |
| `VALID-ACCURACY` + magnitude/self-caught (hand-rescued from the screen's void pile) | 19 |
| **`VOID-NONULL`** | **79** |
| **`VOID-CV`** | **3** |
| `VOID-ZEROSELF` | 0 observed |
| **VOID total** | **82 / 139 = 59%** |
| Rows carrying a binary sha256 | **12 / 139 = 8.6%** |

Comparison with frankenfs: VOID 59% vs 79.3%; sha256 coverage 8.6% vs 10.9%; and the
same dominance pattern — **VOID-NONULL 79 of 82 void here, 214 of 219 there; VOID-CV 3
here, 4 there.** Two independent repos, same epidemic, same rare class. The broadcast's
correction reproduces cleanly.

**Read this honestly, per frankenfs's warning.** 82 void rows are *not* 82 buried wins.
`VOID-NONULL` overwhelmingly means "the row measured ~1.0× and never wrote down what
~1.0× means on that bench". Most of those levers are genuinely dead; the class exists
because the row cannot *prove* it. The actionable yield is a small head (§3).

---

## 3. Ranked rehabilitation queue

Ranked by target-frame self-time. Only 3 VOID rows cite a self-time figure at all, so
frames were attributed from the consolidated frontier map rather than from the rows.

The VOID-NONULL population is concentrated where the time is: **encoder (28 rows) and
int8 (17)** — i.e. the i7 int8 encoder GEMM, which the `bd-i7-…-o0bu` profile puts at
**~28.2% self-time on turbo (43.7% for the whole i7/int8 family)** — the hottest owned
code in the engine.

1. **i7 bias specialization** (`bd-i7-rows-gated-and-substrate-invalid-o0bu`) —
   **~28.2% self-time.** Never validly measured: the original A/B flipped
   `FW_I7_ROWBLOCK_MIN_LEN` across *two* rch invocations and rch ratios are not
   worker-invariant. **Prerequisite:** the flag is a `OnceLock` env read, so it needs a
   runtime setter before both arms can live in one binary. Do **not** re-mine the
   rowblock half — it is an honest `VALID-AB` REJECT (median 0.880, 4/21 wins).
2. **Router diagnostics four-pass fusion** (L390) — `VOID-CV`, and the cheapest to close.
   Candidate **median 1.1847× against null p90 1.1080×**, byte-exactness oracle already
   run before timing, profile-only harness retained. Killed by a `cv` assertion plus a
   `p10 > null p90` test no effect of that size can pass at this noise floor.
3. **Speculation-controller Brier reuse** (`bd-7rxo`) — 18.026% of its caller. **Re-run
   this pass; see §4.** Result: not bankable, harness defect found.
4. **Direct transcript concatenation** (L2641) — `VOID-NONULL`, 1.0748× with a
   byte-identical SHA256 already recorded. Needs its null floor established.
5. **SDPA `BR` tile sweep** (L5455) — `VOID-CV`; its own A/A null read **1.1163× at cv
   29.0%**, so the harness decided nothing. Ceiling capped: target frame is 4.05% of e2e.
6. **`tile_shape`** (L6304) — `VOID-CV`; a **1.689×** mechanism with 23–24/25 paired wins
   killed only by `cv 16.8–24.2%`. Lowest expected e2e yield despite the biggest ratio,
   because per `bd-4hc0` the f32 2D tiler is off the default hot path.

---

## 4. Yield, and exactly how far this got

| Metric | Count |
|---|---:|
| Entries screened (six-class) | 139 / 139 |
| Queue head read in full and hand-adjudicated | ~25 |
| Residual `VOID-NONULL` rows **not** individually hand-read | **79** |
| Re-run under a corrected harness | **1** (`bd-7rxo`) |
| Re-won | **0** |

**Honesty about method.** The broadcast asks for every row to be hand-adjudicated. That
is done for the queue head and for the encoder/int8 population that dominates the void
pile. The 19 rescues in §1 came from a **second screen pass** (accuracy / magnitude /
self-caught keywords), spot-checked by hand but **not** individually adjudicated for all
19; and the residual 79 are screen output only. Full hand adjudication of those 79 is
the outstanding work on this document. I am not claiming it is done.

**The one re-run, and why it produced no win.** `bd-7rxo` measured **1.186–1.193× median
across 3 runs, 60/63 paired wins, against three *valid* A/A nulls** (median 0.9902–0.9997)
— and is still not bankable. Its bench times the "historical" arm as
`recommend() + apply()`, but `apply()` *is already the candidate*: it computes the Brier
score once and threads it into `recommend_with_brier`. The baseline therefore does
**2 folds + 2 decision bodies** where true historical did **2 + 1**. It performs work the
lever never removed, so the ratio is an **upper bound, not an estimate**, and more
sampling cannot repair a construction defect. Filed as `bd-203u`, which now blocks
`bd-7rxo`. Full row in `docs/NEGATIVE_EVIDENCE.md`.

That is a `VOID-ZEROSELF`-adjacent defect — the bench did not exercise the code the row
was about — caught *before* a KEEP was banked rather than months after.

**The gate finding, confirmed by measurement rather than by reading.** Candidate `cv` was
**7.06 / 8.07 / 8.74%** across three runs of a demonstrably calibrated harness. `cv < 5%`
is unreachable here, so `bd-7rxo`'s own acceptance criterion was unsatisfiable as
written. Note this does *not* make CV the epidemic — only 3 rows died on it.

---

## 4b. Institutionalization (broadcast 2) — the audit is now enforced, not documented

frankensqlite sits at **1.7% void** having audited months ago and then made the
check mechanical; every repo that audited once and stopped drifted to 25–91%.
**Ledger integrity decays.** So this audit ships with enforcement:

**`tests/ledger_integrity.rs`** — runs under the mandatory `cargo test` gate,
modelled on this repo's existing `tests/no_canned_phrases.rs` honesty guard. A
rejection row dated on or after `2026-07-26` must record at least one reason it
was *decidable*: an A/A null, a counted mechanism, an accuracy/faithfulness
refutation, a `≤ 0.90×` loss, or a profile-first self-time/Amdahl rejection.
Otherwise the build fails, naming the row and the remedy. Writing a
non-provable rejection is now impossible rather than discouraged.

- History is **grandfathered and pinned**: `LEGACY_NONCOMPLIANT_BUDGET = 99`.
  Failing on legacy rows would make the test permanently red, and a permanently
  red test gets deleted — which is exactly how the discipline decays. The budget
  may only shrink, so backdating a row to dodge the cutoff trips it instead.
- The guard is **validated in both directions**, because a guard that only ever
  passes is worse than none: a synthetic row *"REJECT — measured 1.02x and we
  moved on"* fails it with a file:line citation, and the same row with an A/A
  null recorded passes. A third test asserts the parser still sees >500 entries
  and >200 rejection rows, so a ledger format change makes it fail loudly rather
  than pass vacuously.

**`examples/ledger_preflight.rs`** — the frankensqlite `sql_pipeline_candidate_preflight`
analogue, run *before* touching source:

```text
cargo run --example ledger_preflight -- int4 mlp_0
BLOCKED — 1 binding prior rejection(s):  (exit 2)
```

It distinguishes the two cases this campaign showed are different: **BLOCKED
(exit 2)** when a matching prior rejection records why it was decidable, and
**VOID PRIOR (exit 0)** when it records none — the latter is *not* binding and
the lever is live, which is the whole point of the resurrection audit.

## 5. Blockers

**Ranks 1 and 4–6 are not re-runnable from this lane.** franken_whisper is **Lane L**
(throttled, no worker) under the 2026-07-25 allocation addendum; the standing rule is to
request a window rather than take one. Requested on the campaign thread.

Rank 1 additionally needs a code prerequisite before any measurement is meaningful: a
runtime setter for `FW_I7_ROWBLOCK_MIN_LEN`, since an env `OnceLock` cannot flip between
arms inside one binary — the exact defect that made the original A/B inadmissible.

**Remote benching is separately dead** (`bd-dd90`): `rch` refuses this workspace with
`RCH-E410` (missing remote entrypoint `crates/fsqlite/tests/zz_aggincomposite_bench.rs`),
and when it does dispatch, worker `ovh-a` fails to compile `fsqlite-pager` — which builds
fine locally, so the remote checkout is stale.

**Retry predicate for this document:** re-run the §3 queue when (1) a measurement window
is granted, (2) for rank 1 specifically the runtime setter exists, and (3) the host is
quiet (load < 2, no competing benchmark). Decide every re-run on the median-CI gate —
**decidable iff the claimed ratio lies outside the arm's A/A null 95% CI with a 2×
margin** — and record `cv` as provenance only.
