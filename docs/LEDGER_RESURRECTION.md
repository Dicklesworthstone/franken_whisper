# LEDGER RESURRECTION — franken_whisper

Meta-Lever #1 audit of `docs/NEGATIVE_EVIDENCE.md`, per the 2026-07-25 fleet
performance campaign. Question asked of every REJECT row: **could the
measurement that produced this rejection have detected the lever at all?** A row
whose answer is "no" is VOID — it rejected the harness, not the lever.

Audited by: cc / STRUCTURAL lane. Date: 2026-07-25.

---

## 0. Headline (read this, not the raw screen numbers)

| Quantity | Value |
|---|---|
| Entries in `docs/NEGATIVE_EVIDENCE.md` | 709 |
| Entries whose header carries a REJECT verdict | 138 |
| REJECTs hand-audited this pass | 16 (the CV-gated subset — the class §2.3 predicts is void) |
| Confirmed **VOID** | 4 |
| Confirmed **SOUND** (decision survives the corrected gate) | 5 |
| Reclassified: not a REJECT at all (self-labelled blocker/surface row) | 2 |
| Open/unmeasured half of a live row over the top owned frame | 1 |

**The naive screen is misleading and I am not reporting it as a finding.** A
regex over the corpus says only 12% of REJECT bodies mention a null control and
only 8% cite self-time. That is *not* an 88% void rate. Hand-auditing shows the
recent rows describe their null control in prose the regex misses, and — more
importantly — that they **gate on candidate-p10 vs null-p90**, which is already
the §2 median-CI-style gate. The void vein in this repo is narrow, old, and
concentrated in one week (2026-07-10), not spread across the corpus.

---

## 1. The one real systematic finding

**This repo's `cv < 5%` gate is redundant where the row is sound, and decisive
only where the row is void.**

The campaign warned (§2.3) that `cv < 5%` is unreachable on this hardware. That
is confirmed here, from this repo's own recorded numbers — the honest A/A null
floors measured on these workers are enormous:

| Row | A/A null control (identical code both arms) | Implied floor |
|---|---|---|
| SDPA `BR` tile sweep (2026-07-10) | **1.1163×** at **cv 29.0%** | ±12% |
| `tile_shape` A/B (2026-07-10) | fails at **6.8–17.7%** | ±18% |
| i7 rowblock same-binary closeout (2026-07-14) | p10/median/p90 **0.735 / 0.978 / 1.107** | −27%/+11% |
| correction-evidence scan fusion (2026-07-24) | p10/median/p90 **0.902 / 0.995 / 1.050** | ±5–10% |

A gate demanding `cv < 5%` against floors of ±12–27% cannot pass. Where the row
*also* recorded a proper p10-vs-null-p90 comparison, the CV clause changed
nothing and the rejection stands. Where the CV clause was the *only* thing
standing between a measured effect and a KEEP, the row is VOID.

### 1b. The second finding — this repo's fallback gate is stricter than §2's

Where the CV clause was backstopped, the backstop was **candidate p10 > null
p90**. That demands the candidate's 10th percentile beat the null's 90th — i.e.
near-total non-overlap of two noisy distributions. §2's gate is on the
**median** against the null's CI. These are not the same test, and the gap is
where a real effect hides:

| Row | candidate p10 / **median** / p90 | null p10 / med / **p90** | p10 > null-p90? | median outside null envelope? |
|---|---|---|---|---|
| correction-evidence fusion (L352) | 0.9183 / **1.0394** / 1.1090 | 0.9019 / 0.9949 / **1.0504** | no | **no** → SOUND |
| router diagnostics fusion (L390) | 1.0209 / **1.1847** / 1.2825 | 0.9430 / 1.0063 / **1.1080** | no | **YES** → VOID |

L352 and L390 got the same verdict from the same harness, but they are not the
same result. L352's median sits *inside* the null envelope — genuinely
undecidable, correctly rejected. L390's median is **1.1847× against a null
median of 1.0063×**, clear of the null p90 — an ~18% directional effect that the
harness threw away on a CV assertion (it "correctly exited 101") plus a p10 test
no lever of that size could pass at this noise floor. **That is a rejected
harness, not a rejected lever.**

---

## 2. Hand-audited rows

| # | Entry (line) | Ratio claimed | Null floor at the time | Self-time of target frame | Binary sha? | Verdict |
|---|---|---|---|---|---|---|
| 1 | `tile_shape` A/B, 2026-07-10 (L6304) | **1.689× fc1 / 1.686× fc2, 23–24/25 wins** | null control itself FAILS at 6.8–17.7% | f32 2D tiler — see note | yes | **VOID** |
| 2 | SDPA `BR` tile sweep, 2026-07-10 (L5455) | not decidable | **null = 1.1163× @ cv 29.0%** | `matrixmultiply::gemm::gemm_loop` **4.05% e2e** | yes (`272102fd…`) | **VOID** |
| 3 | i7 **bias specialization** (bd-…-o0bu) | two-invocation rch ratio — inadmissible | none valid (cross-invocation) | **~28.2% self (turbo); 43.7% incl. family** | n/a | **VOID (unmeasured)** |
| 4 | i7 **rowblock coarsening** (same bead, closed 07-14) | cand p10/med/p90 0.748/**0.880**/1.161, 4/21 wins | null 0.735/0.978/1.107 | same ~28–43.7% frame | yes (`6803c12d…`) | **SOUND** (candidate is a *loss*) |
| 5 | correction-evidence six-to-one scan fusion, 07-24 (L352) | med **1.0394**, p10 0.918, 15/21 | null p90 **1.0504** | 96.23% of its caller | job-pinned | **SOUND** (p10 inside null) |
| 6 | router diagnostics four-pass fusion, 07-23 (L390) | p10/med/p90 **1.0209 / 1.1847 / 1.2825** | null p10/med/p90 **0.9430 / 1.0063 / 1.1080** | 21.63% stage share of its caller | job-pinned | **VOID — re-run** |
| 7 | TTY decode `HashSet`→adjacent-seq, 07-13 (L3079) | **1.0249×** | "inside the valid BASE/BASE null envelope" | — | — | **SOUND** |
| 8 | word-timestamp segment prealloc, 07-13 (L3139) | **1.0200×** | "inside the valid BASE/BASE null envelope" | — | — | **SOUND** |
| 9 | YouTube manifest JSON scratch, 07-15 (L2002) | **1.0069×**, 8/15 | self-labelled INVALID NULL | — | — | **SOUND** (sub-1.01 is undecidable per §2) |
| 10 | direct transcript concatenation, 07-14 (L2641) | **1.0748×**, byte-identical (sha `2916b65e…`) | "lost the tail gate" — floor not restated | `concat_segment_text` in `CorrectionDrift::compute` | yes | **RE-RUN** |
| 11 | wide-i7 K=64 unrolling, 07-10 (L4970) | candidate never executed | BASE/BASE "too biased and broad" | 43.717% family self-time | yes | not a REJECT (self-labelled SURFACE/BLOCKER) |
| 12 | speculation controller Brier scan (bd-kdg7.1 → bd-7rxo) | none — no arm ever ran | none | **18.026% of `apply`** | n/a | not a REJECT (BLOCKED / NO VERDICT) |

Rows 13–16 of the CV-mentioning subset are ledger-integrity audits and
surface/coordination rows, not lever rejections; they are excluded from the
verdict counts above.

**Note on row 1.** The `tile_shape` mechanism is real and large (1.689×, 23–24/25
paired wins) and the rejection was driven purely by a `cv` gate the hardware
cannot meet — so the *row* is VOID. But its e2e value is capped by a separate,
independently-established fact: per `bd-4hc0`'s correction, the f32 2D tiler is
**off the default hot path** (all 32 linears run int8; only conv2 reaches the
tiler). A resurrected 1.689× on a frame that barely executes is not an e2e win.
This is exactly why §1 ranks by self-time rather than by claimed ratio.

---

## 3. Resurrection queue, ranked by target-frame self-time

1. **i7 bias specialization** — `~28.2%` self-time on turbo (`dot_maddubs_i7_m2n4`
   14.63% + `matmul_bias_i7_quantized` 9.91% + `matmul_bias_i8` 2.74% +
   `quantize_act_i7` 0.88%); 43.717% for the whole i7/int8 encoder family. The
   **hottest owned code in the engine.** Never validly measured: the original A/B
   flipped `FW_I7_ROWBLOCK_MIN_LEN` across *two* rch invocations, and rch ratios
   are not worker-invariant. **Hard prerequisite:** the flag is a `OnceLock` env
   read, so it cannot flip inside one binary — it needs a runtime setter first,
   exactly like the existing `set_sdpa_poly_exp` / `set_sgemm_tile_balanced`.
2. **Speculation controller Brier reuse** (bd-7rxo) — 18.026% of the profiled
   `apply` caller. Uniquely actionable: the production change and a §2-compliant
   harness (one pinned binary, A/A identity null, order-alternating 21 pairs,
   up-front evidence-parity oracle) are **already landed** at `fd3bdd5`; only the
   measurement is outstanding. Its declared gate is `candidate_cv < 0.05`, which
   §2.3 says to replace with the median-CI gate.
3. **Router diagnostics four-pass count/calibration fusion** (row 6) — 21.63%
   stage share of its caller, **median 1.1847× clear of null p90 1.1080×**, and
   the byte-exactness oracle (full serialized diagnostics JSON) was already run
   before timing. The single best ratio-to-effort item in the queue: the
   candidate source was removed, but the profile-only harness was retained, so
   re-landing the fold and re-deciding it on the median-CI gate is a small job.
4. **Direct transcript concatenation** (row 10) — 1.0748× with a byte-identical
   SHA256 proof already in hand. Needs its null floor restated; 1.075× is
   plausibly outside a ±5% floor but not a ±12% one.
5. **SDPA `BR` tile sweep** — void, but the target frame is only 4.05% of e2e, so
   even a full resurrection is capped near 4%.
6. **`tile_shape`** — void with a large mechanism, but off the default hot path
   (see note above). Lowest expected e2e yield despite the biggest ratio.

---

## 4. Yield this pass

- Audited: 16 hand / 138 screened.
- Void: 4.
- Re-run: 0 completed this pass — **blocked, and the blocker is named in §5.**
- Re-won: 0.

Honest statement: this repo does **not** have frankenlibc's 39-of-93 void rate.
Its recent ledger discipline is genuinely good — every 2026-07-2x row carries a
same-worker A/A null, a pinned job id, and a byte-exactness oracle run *before*
timing. The failure here is subtler than "no null control": it is a **decision
rule** that is too strict for the measured noise floor, applied on top of a `cv`
assertion that cannot pass at all. That combination discards real effects
(L390's ~18%) while correctly discarding fake ones (L352). The remedy is §2.3's
median-CI gate, not more sampling.

The value is concentrated in three places: one very high-self-time item never
validly measured (#1), one measurement already fully teed up (#2), and one
sizeable effect thrown away by the gate (#3).

---

## 5. Blocker that stopped the re-runs (fixed this session, but see the residue)

Both re-run candidates need a working bench. At the start of this session
**the crate did not compile at all**: the `frankensqlite` async migration
(`54020c68`, `a0ab400a`, 2026-07-25) turned `fsqlite::Connection`'s `open` /
`query` / `execute` / `*_with_params` into `async fn`, and this repo's fully
synchronous `src/storage.rs` + `src/sync.rs` still called them directly — 235
compile errors across the two files. Fixed this session by a `BlockingConnection`
facade (see `docs/PERF_LEDGER.md`).

Second-order finding worth knowing before you run any test or bench here:
driving those futures synchronously is **stack-hungry**. `fsqlite`'s statement
futures nest deeply (statement dispatch → DML → triggers → nested statement
execution), and in a debug build the poll chain overflows a default libtest
thread stack — the suite aborted with SIGABRT after only 3 of 202 storage tests.
The same 202 tests pass **202/0** under `RUST_MIN_STACK=67108864`. Boxing the
outermost future helps but is not sufficient on its own. If you see a
"has overflowed its stack" abort in this repo, that is what it is; it is not a
logic bug in the code under test.

**Residual blocker, not fixed here:** remote benching is still unavailable.
`rch` refuses this workspace with `RCH-E410` (missing remote source entrypoint
`crates/fsqlite/tests/zz_aggincomposite_bench.rs`), and when it does dispatch,
worker `ovh-a` fails to build `fsqlite-pager`. That is bead `bd-dd90`, and it is
the same frankensqlite-side breakage the campaign flagged as "blocking
franken_whisper's benchmarking". Until it clears, every A/B here must run local,
which is admissible for single-binary paired micro-benches (§2.2) but not for the
32-thread encoder gates.

**Retry predicate for this document:** re-run the §3 queue once (1) `bd-dd90`
clears or a local single-binary harness is accepted for the item, AND (2) for
item #1 specifically, a runtime setter for `FW_I7_ROWBLOCK_MIN_LEN` exists so
both arms can live in one binary. Re-decide every re-run on the median-CI gate —
**decidable iff the claimed ratio lies outside the arm's A/A null 95% CI with a
2× margin** — and record `cv` as provenance only.
