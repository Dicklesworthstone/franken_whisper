# Ledger Resurrection Audit — franken_whisper

**Campaign:** `perf-campaign-20260725`, Fleet-Wide Meta-Lever #1.
**Lane:** cod / HARNESS+FRONTIER, Lane L under the 2026-07-25 allocation addendum.
**Source audited:** `docs/NEGATIVE_EVIDENCE.md` (713 `##` entries; 25,910 lines
in the audit worktree).
**Method source:** `/data/projects/frankenfs/docs/LEDGER_RESURRECTION.md`, read
in full before this audit.

A REJECT row is **VOID** when the measurement *could not have detected the lever* — as
opposed to detecting it and finding it absent.

---

## Model-integrity re-audit — 2026-07-27

The provider silently ran this pane on a lower-capacity fallback model between
2026-07-25 20:40 and 2026-07-26 00:35 EDT. The seven commits authored in that
window were re-read under the restored model. Existing A/A, ELF-digest, and
byte-identity artifacts were not rerun wholesale; the review targeted workload
routing, proof soundness, numerical justification, and ungated code quality.

| Commit | Verdict | Fresh audit |
|---|---|---|
| `d92b511` | **CORRECTED** | The synchronous storage route is real and its 202-test proof exercises the bridge, but the same commit's first resurrection audit sampled the wrong class and its facade could panic if runtime construction failed. Current code returns `FrankenError::Internal`, and the audit denominator below supersedes the draft. File-wide unsafe exemptions on two standalone probes were narrowed to their audited blocks. |
| `aa5ab8b` | **CORRECTED** | Directly running the release storage test binary with `RUST_MIN_STACK` unset proves that exact 202-test release path does not overflow. It did not measure “room to spare” or every future query shape; the ledger and comments now state the narrower result. |
| `efb42b7` | **SOUND** | It correctly refused a KEEP: the timed “historical” arm executed an extra decision body and therefore did not measure the production lever. The behavior proof is sound because both Brier reads are the same pure fold over state that is not mutated between reads; action, fallback, and serialized evidence were checked before timing. |
| `b17945c` | **CORRECTED** | The rerun counts and no-verdict judgment are sound, but three contended runs do not prove that CV below 5% is impossible on the hardware. The ledger now states the defensible conclusion: CV was an invalid verdict gate for those runs and remains provenance only. |
| `ac3b633` | **SOUND** | The README used the matched-greedy rows actually recorded in `PERF_FRONTIER`, explicitly withdrew the beam-5/best-of-5 headline, and disclosed the then-current 0.78× losing cell. Its later 1.35× replacement reflects a subsequent code change, not a defect in this correction. |
| `a9c58e6` | **CORRECTED** | It claimed a verbatim six-class audit while inventing `VALID-ACCURACY`, used 139 header matches as the population, and left most rows mechanically screened. The current audit uses exactly six classes, excludes 89 false positives, and records hand adjudication of all 188 actual performance rejections. |
| `76901c4` | **CORRECTED** | The first gate allowed accuracy/profile/large-loss exceptions, ignored KEEP binary digests, inspected the full ledger rather than staged changes, and was not wired into pre-commit. `bd6243a` replaced that design; the re-audit additionally binds counted evidence by clause, rejects missing-binary/output-hash laundering, and makes the hook compile the staged gate source. |

No speed KEEP from this window is retracted. The only timed candidate in the
window (`efb42b7`) deliberately carried **no admissible speed verdict**.

### Final remediation reconciliation — 2026-07-27

All five `CORRECTED` verdicts now map to landed fixes:

- `d92b511`: `d78f7f3` replaces the sampled resurrection draft with the full
  hand audit; `d0b4a8e` propagates runtime-construction failure and narrows the
  standalone probes' unsafe exemptions.
- `aa5ab8b`: `d0b4a8e` limits the release-stack conclusion to the exact
  202-test path and explicitly disclaims spare-headroom and future-shape proof.
- `b17945c`: `d0b4a8e` scopes the CV finding to the three observed contended
  runs. This closeout also removes the stale CV ratchet from live beads
  `bd-pjl6`, `bd-7rxo`, and `bd-203u`, appends an immutable correction to
  `bd-7rxo`'s old comment, and corrects the two surviving universal-CV
  references plus the stale Brier retry predicate in
  `NEGATIVE_EVIDENCE.md`.
- `a9c58e6`: `d78f7f3` applies the exact six-class taxonomy to all 188
  hand-adjudicated performance rejections; `d0b4a8e` records the model audit.
- `76901c4`: `bd6243a` replaces the permissive prototype with the staged
  pre-commit contract; `d0b4a8e` binds nulls, counted mechanisms, and binary
  digests to the evidence they purport to prove and compiles the staged gate
  source.

There are no `RETRACTED` verdicts and therefore no retraction-dependent speed
claims to withdraw. The dependent scan covered `NEGATIVE_EVIDENCE.md`,
`PERF_LEDGER.md`, `README.md`, every scorecard-named path (none exist), and all
current bead descriptions, notes, and comments. Historical measurements remain
as audit records; superseding corrections are explicit rather than silent.

The hardened gate was then run against this repository's real ledger via
`RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo test --test
ledger_integrity` on `vmi1149989`: **7 passed, 0 failed**. The real-ledger test
found no undecidable post-enforcement rejection. The adversarial tests confirmed
that accuracy/profile/CV/magnitude prose cannot bypass the REJECT contract, a
candidate statistic cannot launder a missing numerical null, an unchanged
output cannot launder changed allocation counts, and an output-oracle digest
cannot launder a missing binary/ELF digest. This is the judgment defect class
that escaped the first `76901c4` prototype. A live `surface speculation brier`
self-check also caught and corrected a stale retry predicate, then led to
parser hardening: an explicit retry heading now outranks an earlier historical
mention of the words “retry predicate.”

---

## 0. Population correction and hand-adjudication

The mechanical header screen used this repository's actual verdict vocabulary:

```text
REJECT | DEAD | CLOSED | FALSIFIED | NO-SHIP | DO-NOT-RETRY | NEGATIVE
```

It produced **277 candidates**. That number is a queue, not a denominator.
Reading every candidate exposed **89 false positives**: KEEP rows that discuss a
prior rejection, infrastructure blockers, superseding corrections, surveys, and
specification closures whose prose happens to contain one of the words above.
Those rows are excluded rather than silently counted as valid evidence.

The remaining **188 performance-rejection rows were each hand-adjudicated**.
The regex never assigned a verdict. Nested `###` evidence stayed attached to its
parent `##` row; this avoids the over-splitting defect documented by frankenfs.
For ambiguous rows, the decision followed the evidence that actually killed the
lever:

- a bare wall ratio, even a large regression, is not a counted mechanism;
- a recorded count can be broader than one spelling (for example bytes moved,
  accepted tokens, FLOPs, or a quantified output mismatch), but it must make the
  rejected premise impossible rather than merely restate wall time;
- correctness-only KEEP/survey prose is excluded; a quantified correctness
  mechanism inside a performance rejection is `VALID-MECHANISM`;
- `cv` may be reported as provenance but never makes a row valid.

This supersedes the earlier 139-row and 277-row-as-denominator drafts.

---

## 1. Taxonomy (frankenfs, adopted verbatim)

| Class | Meaning | Sound? |
|---|---|---|
| `VALID-PROFILE` | Rejected before any source edit, on a named profile frame with non-zero self-time and a computed Amdahl ceiling. | ✅ |
| `VALID-MECHANISM` | No A/A null recorded, but refuted on a *counted* mechanism — instructions/cycles/syscalls/allocations/faults unchanged. A null control cannot change "no work was removed". | ✅ |
| `VALID-AB` | A/B run with a recorded A/A null; the claimed effect sits inside that null. | ✅ |
| `VOID-CV` | An A/B ran, and the row was killed **only** by a `cv < 5%` gate. CV alone is not an admissible verdict gate. | ❌ |
| `VOID-ZEROSELF` | The target frame had ~0% self-time in the profile the bench actually exercised. | ❌ |
| `VOID-NONULL` | An A/B ran, was rejected on a near-1.0 wall ratio, and recorded **no** A/A null control and no counted mechanism. Cannot distinguish lever from harness. | ❌ |

There is deliberately **no seventh class**. Quantified WER/token/output failures
are adjudicated under `VALID-MECHANISM` only when the count itself refutes the
lever; otherwise the row is excluded from the performance-rejection population.

---

## 2. Counts

The mechanical screen was followed by a row-by-row read. The denominator below
is the 188 actual performance rejections, not the 277 regex hits.

| Metric | Count |
|---|---:|
| `##` ledger entries parsed | 713 |
| Mechanical header-screen candidates | 277 |
| False-positive/non-performance candidates excluded by hand | 89 |
| **Performance rejections hand-adjudicated** | **188** |
| `VALID-AB` | 19 |
| `VALID-MECHANISM` | 96 |
| `VALID-PROFILE` | 13 |
| **`VOID-NONULL`** | **56** |
| **`VOID-CV`** | **4** |
| `VOID-ZEROSELF` | 0 observed |
| **VOID total** | **60 / 188 = 31.9%** |
| Rows carrying a benchmark-binary/ELF sha256 | **12 / 188 = 6.4%** |

The same fleet pattern survives the corrected denominator:
**VOID-NONULL is 56 of 60 void rows (93.3%)**, while only four rows died solely
on CV. The campaign prediction that CV would dominate was wrong here too.

**Read this honestly, per frankenfs's warning.** Sixty void rows are not sixty buried wins.
`VOID-NONULL` overwhelmingly means "the row measured ~1.0× and never wrote down what
~1.0× means on that bench". Most of those levers are genuinely dead; the class exists
because the row cannot *prove* it. The actionable yield is a small head (§3).

### Audit trail

Line numbers below refer to the 25,910-line audit worktree snapshot. They make
the hand decisions reviewable without pretending the screen was the verdict.

- `VALID-AB` (19): 7, 482, 2094, 2279, 2529, 2591, 2654, 2713, 2771,
  2865, 2922, 3006, 3085, 3124, 3209, 3269, 4835, 4951, 5993.
- `VALID-PROFILE` (13): 6723, 7027, 7192, 7943, 8325, 9728, 11414,
  11472, 11531, 11724, 15594, 16454, 16485.
- `VOID-CV` (4): 520, 5585, 6434, 25686.
- `VOID-ZEROSELF` (0): none.
- `VOID-NONULL` (56): 2132, 7548, 7584, 7635, 7642, 7798, 7952,
  7958, 8084, 8087, 9296, 9998, 11921, 12127, 12141, 12159, 12189,
  12303, 12600, 12647, 13269, 13304, 13592, 13890, 15631, 16117,
  16150, 16682, 17558, 17714, 18254, 18339, 18419, 18485, 18863,
  19214, 19656, 19736, 19875, 20097, 20156, 20201, 20571, 20854,
  21181, 21256, 21433, 21537, 22926, 23010, 23176, 24772, 25221,
  25274, 25318, 25435.
- `VALID-MECHANISM` (96): 1402, 3508, 4157, 4190, 4293, 4547,
  6889, 7537, 7762, 7946, 7949, 7994, 8081, 8151, 8265, 8328,
  8331, 8365, 8382, 8402, 8442, 8459, 8474, 8562, 8720, 9062,
  9170, 9204, 9234, 9263, 9358, 9496, 9690, 9760, 9832, 9869,
  9935, 10141, 10613, 10724, 10995, 11030, 11358, 11396, 11449,
  11573, 11616, 11679, 11695, 11704, 11750, 11844, 11862, 11880,
  11890, 11958, 12007, 12021, 12083, 12091, 12315, 12341, 12420,
  12456, 12570, 12944, 13082, 13336, 13383, 13621, 14491, 14934,
  15411, 15461, 15477, 15725, 15855, 15934, 16086, 16526, 17364,
  17833, 18561, 18650, 20374, 20404, 20451, 21103, 21142, 23847,
  24698, 24729, 25247, 25507, 25716, 25735.

---

## 3. Ranked rehabilitation queue

Ranked by target-frame self-time. Only 3 VOID rows cite a self-time figure at all, so
frames were attributed from the consolidated frontier map rather than from the rows.

The VOID-NONULL population is concentrated where the time is: **encoder (28 rows) and
int8 (17)** — i.e. the i7 int8 encoder GEMM, which the `bd-i7-…-o0bu` profile puts at
**~28.2% self-time on turbo (43.7% for the whole i7/int8 family)** — the hottest owned
code in the engine.

1. **i7 bias specialization** (`bd-i7-rows-gated-and-substrate-invalid-o0bu`) —
   **~28.2% self-time.** The exact const-generic bias/no-bias candidate was
   reverted and is no longer reconstructable. The available
   `FW_I7_ROWBLOCK_MIN_LEN` selector controls a different, already-validly
   rejected rowblock lever and cannot serve as a proxy.
2. **Router diagnostics four-pass fusion** (audit line 520) — `VOID-CV`,
   **21.63%** of its caller. Its old 1.1847× result was killed by CV despite a
   valid null median.
3. **Speculation-controller Brier reuse** (`bd-7rxo`) — `VOID-NONULL`,
   **18.026%** of `apply()`.
4. **SDPA `BR` tile sweep** (audit line 5585) — `VOID-CV`; target packing
   frame **4.05% of e2e**.
5. **`tile_shape`** (audit line 6434) — `VOID-CV`; the historical T=14
   mechanism read **1.689×** with 23–24/25 wins but was killed by CV. It has
   the lowest expected e2e yield because the f32 2-D tiler is off the default
   hot path.

The hand pass removed **direct transcript concatenation** from the queue. Its
row records a same-binary A/A null, so it is `VALID-AB`, not resurrectable.

---

## 4. Yield, and exactly how far this got

| Metric | Count |
|---|---:|
| Mechanical candidates read and adjudicated | **277 / 277** |
| Actual performance rejections classified | **188 / 188** |
| Queue rows closed under corrected evidence | **4 / 5** |
| Resurrected KEEP | **2** |
| Corrected REJECT | **1** |
| Faithful rerun blocked before timing | **1** |
| Awaiting Lane-L measurement window | **1** |

The top-five outcomes are:

1. **i7 bias specialization — BLOCKED, no proxy verdict.** The original
   candidate source is absent. `bd-p5ku` records the exact reconstruction
   predicate; rowblock coarsening is not the same lever.
2. **Router diagnostics — KEEP, 1.120094×.** Null median
   `0.994045`, candidate bootstrap median CI `[1.091145, 1.149635]`,
   complete serialized diagnostics JSON identical.
3. **Brier reuse — KEEP, 1.217822×.** The first rerun found that its
   "historical" arm performed an extra decision body and was inadmissible.
   A faithful runtime selector then put the true historical and current
   `apply()` paths in the same binary: null `1.000247`, candidate median CI
   `[1.212472, 1.232171]`, action/fallback/evidence JSON identical.
4. **SDPA BR=128 — REJECT, 1.011137×.** Self-reporting ELF
   `3ba35b4a7d7ba48d48fb0c8ed2ffdd3a83454f8e69047702134e573e58c789cb`;
   A/A median CI `[0.965832, 1.039802]`; corrected 2× null-CI floor
   `1.079604`; 1,920,000 f32 outputs bit-identical. CV was provenance only.
5. **`tile_shape` — pending.** Lane L has no worker allocation. The campaign
   thread contains the requested measurement-window handoff; taking a worker
   without a grant would violate the superseding addendum.

The first Brier attempt is retained as an important harness result, not counted
as a fifth verdict: candidate medians 1.186–1.193× and valid A/A medians
0.9902–0.9997 could not rescue a baseline that executed work historical
production never did. More sampling cannot repair arm construction.

---

## 4b. Institutionalization (broadcast 2) — the audit is now enforced, not documented

frankensqlite sits at **1.7% void** having audited months ago and then made the
check mechanical; every repo that audited once and stopped drifted to 25–91%.
**Ledger integrity decays.** So this audit ships with enforcement:

**`examples/ledger_preflight.rs`** is the self-contained frankensqlite-style
gate. It has two responsibilities:

```text
cargo run --example ledger_preflight -- surface int4 mlp_0
cargo run --example ledger_preflight -- validate-staged
```

`surface` searches `docs/NEGATIVE_EVIDENCE.md`, prints every matching prior row,
and prints its concrete retry predicate verbatim when present. A sound prior
KEEP/REJECT blocks with exit 2; a void prior is explicitly non-binding.

`validate-staged` compares the staged ledgers with HEAD. Any new or modified
REJECT/DEAD/CLOSED/FALSIFIED/NO-SHIP/DO-NOT-RETRY/NEGATIVE row is rejected
unless it records either:

1. a **same-invocation A/A null** with a numerical median/CI; or
2. a **counted mechanism** showing instructions, cycles, syscalls,
   allocations, bytes, or faults unchanged.

There are no accuracy, large-regression, profile, or bare-CV exceptions in the
write gate. A new KEEP/WIN is rejected unless it carries a 64-hex benchmark
binary/ELF sha256.

Performance keeps also have a mandatory result class:

1. `SELF-SPEEDUP / MAINTENANCE` for franken-before versus franken-after. This
   may justify a code landing but never counts as campaign output.
2. `INCUMBENT-WIN / CAMPAIGN WIN` only when the row names the actual legacy
   incumbent and its binary SHA-256, records a numerical incumbent ratio, and
   states that both tools ran side-by-side in the same invocation.

An incumbent comparison missing that execution shape is
`NON-CAMPAIGN / INFORMATIONAL` and cannot use a positive verdict. The same
preflight rejects new public-performance retraction narratives; current public
docs state the current admitted number while the full history stays in this
document and the ledgers. Exit 0 means clear; **exit 2 means blocked**.

**`tests/ledger_integrity.rs`** exercises the same positive and negative
contracts, including false-positive phrases such as "no null control", an
output-oracle SHA that is not a binary SHA, a self-speedup presented as a
campaign win, and a same-session incumbent comparison presented as
same-invocation evidence.

**`.githooks/pre-commit`** compiles the std-only Rust preflight directly and
runs `validate-staged`. Repository-local `core.hooksPath=.githooks` activates
it. The hook reads the Git index rather than the worktree, so an unstaged
explanation cannot launder an invalid staged row.

## 5. Blockers

**Rank 5 and the tiny.en segment-timestamp gate are not runnable from this
lane.** franken_whisper is **Lane L** (throttled, no worker) under the
2026-07-25 allocation addendum; the standing rule is to request a window rather
than take one. A measurement window was requested on the campaign thread.

Rank 1 additionally needs a code prerequisite before any measurement is meaningful: a
faithful reconstruction of the removed bias/no-bias const-specialization or a
runtime selector around those exact production arms. The available rowblock
setter is not a substitute.

**Remote benching is separately dead** (`bd-dd90`): `rch` refuses this workspace with
`RCH-E410` (missing remote entrypoint `crates/fsqlite/tests/zz_aggincomposite_bench.rs`),
and when it does dispatch, worker `ovh-a` fails to compile `fsqlite-pager` — which builds
fine locally, so the remote checkout is stale.

**Retry predicate for this document:** finish rank 5 when (1) a measurement
window is granted and (2) the host is quiet (load < 2, no competing benchmark).
Reopen rank 1 only when its exact source exists. Decide every re-run on the median-CI gate —
**decidable iff the claimed ratio lies outside the arm's A/A null 95% CI with a 2×
margin** — and record `cv` as provenance only.
