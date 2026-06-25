# franken_whisper — Performance Lever Ledger

> Head-to-head, MEASURED optimization log for the native Rust engine. Owned by
> swarm agent **BlackThrush** (franken_whisper-cc). Every entry records a real
> criterion measurement; ~0-gain or regressing levers are REVERTED, not kept.

## Measurement protocol

- **Harness:** `benches/native_engine_bench.rs` (criterion).
- **Build/run:** per-crate via `rch exec` with
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cc` (never
  workspace-wide).
- **Baseline vs candidate:** criterion `--save-baseline` / `--baseline`. rch
  repo-convergence keeps a repo on one worker, so save/compare runs land on the
  same hardware; large effects (≫ worker variance) are the bar for keeping a
  lever.
- **Conformance gate:** every numeric kernel change ships with a **bit-exact
  parity test** against the pre-change reference, so a "win" cannot silently
  alter output. The mel output is conformance-checked against whisper.cpp's exact
  encoder input.
- **What the original is:** whisper.cpp's exact algorithms (this engine is a
  faithful Rust port). A kernel lever's "gain" is the measured speedup of the
  Rust port over its own faithful-port baseline while preserving bit-exact output
  — i.e. doing whisper.cpp's identical math, faster.

## Hermetic vs model-gated benches

| bench | hermetic? | status |
|---|---|---|
| `native_engine/mel/mel_30s` | yes | **measured** |
| `native_engine/f16_gemv/*` | yes | available |
| `encoder_window_{tiny,large}` | no (model+jfk.wav) | tiny unlocked locally; large needs `large-v3-turbo` |
| `decoder_token_step_{tiny,large}` | no | tiny unlocked locally |
| `logits_gemv_large` | no (large model) | blocked: model absent |
| `e2e_tiny_jfk` | no (model) | tiny unlocked locally |

> `tests/fixtures/native/jfk.wav` is gitignored; copied locally from
> `legacy_whispercpp/whisper.cpp/samples/jfk.wav` (mono 16 kHz, 11 s) to unlock
> the model-gated benches for measurement. The `large-v3-turbo` model is not
> present, so the large-shape levers remain blocked (concrete blocker).

---

## Levers

### L1 — log-mel FFT twiddle precompute (bit-exact)  — `src/native_engine/mel.rs`

**Hypothesis.** whisper.cpp's recursive `fft` recomputes `cos`/`sin` twiddles
per butterfly per frame, and the odd-`N` base case `dft(25)` (reached 16×/frame,
3000 frames) recomputes ~1250 f64 transcendentals per call — ~60 M `sin`/`cos`
per 30 s of audio. These are pure functions of `(k, j, n)` and can be precomputed
into f32 tables once, evaluated bit-for-bit identically thereafter.

**Change.** Precompute level twiddles `[400,200,100,50]` and the `n=25` DFT
`25×25` cos/sin table once (cached `OnceLock`, shared read-only across mel worker
threads); thread them through `fft`/`dft`. Arithmetic and accumulation order
unchanged → bit-exact.

**Conformance.** New test `fft_twiddle_table_is_bit_exact_vs_inline_reference`
asserts byte-for-byte `Vec<f32>` equality vs an inline-transcendental copy of the
original recursion across 10 transform widths × 64 random seeds.

**Measurement (worker vmi1149989, criterion; baseline + candidate on the SAME
worker via rch repo-convergence → valid A/B; baseline = pre-edit code):**

| bench | baseline (mel-pre) | candidate | change | speedup |
|---|---|---|---|---|
| `mel_30s` | 269.06 ms | 29.23 ms | **−89.1%** (p=0.00) | **≈9.2×** |

**Verdict: KEEP.** A 9.2× reduction on the always-on log-mel frontend, far above
any plausible worker variance, with **byte-identical output** (parity test green
— see below). The transcendental-elimination hypothesis is confirmed: the
`dft(25)` base case was the dominant cost.

**Honesty note — what "vs the original" means here.** This ratio is
franken_whisper's mel frontend vs **its own prior faithful-port baseline**, not a
direct timing of OpenAI Whisper's Python mel or whisper.cpp's C mel. The output
remains bit-exact to whisper.cpp's mel (the port's contract), so this is "do
whisper.cpp's identical math, 9.2× faster." A true head-to-head wall-clock vs the
C++/Python originals needs the original-vs-franken harness (bd-zk43 / bd-0hnz);
the large-shape kernels also need the `large-v3-turbo` model staged (bd-ms0x).

**Conformance gate (CONFIRMED GREEN):** `cargo test -p franken_whisper --lib
native_engine::mel` → **7/7 passed** incl.
`fft_twiddle_table_is_bit_exact_vs_inline_reference` (0.32 s). Clippy
`-D warnings` initially flagged the new `n % 2 == 0` (`manual_is_multiple_of`);
fixed forward in **b0577d9** (`n.is_multiple_of(2)`, the codebase idiom) →
clippy green (`Finished dev`, exit 0). Both commits on `origin/main` + `master`.

> **Commits:** `656f55c` (L1) + `b0577d9` (clippy fix-forward).

### L2 — log-mel FFT per-call allocation elimination (deferred)  — bd-02do

The recursive `fft` still `vec!`-allocates even/odd split + child-output buffers
at every recursion node (~60 allocs/frame × 3000 frames). Secondary to the
twiddle win (allocator churn ≈ single-digit ms vs the ~240 ms transcendental
cost just removed). Tracked in bd-02do as a follow-up via per-thread scratch
buffers.

**Status: MEASURED, NOT LANDED (deferred).** Pre-verified bit-exact (standalone
scratch-FFT harness, 418,800 outputs, 0 mismatches). Measured via a standalone
local same-process A/B (stable host — the rigorous way given the 5.6× worker
variance below) over a realistic 3000-frame `N_FFT=400` pass:

| FFT pass (3000 frames, 1 thread) | time | speedup |
|---|---|---|
| alloc (current) | 28.5 ms | — |
| scratch (L2) | 23.4 ms | **1.21× (stable across runs)** |

**Decision — not landed.** The 1.21× is real at the FFT-kernel level, but the
FFT is only part of `mel_30s` (≈1.1× there) and `mel_30s` is itself a small
fraction of end-to-end transcription ⇒ **e2e gain ≈ 0**. Landing it also forces
`compute_frame_column` past the 7-arg `clippy::too_many_arguments` limit
(struct-refactor or `#[allow]`) — added complexity in a freshly-clean file for
no e2e benefit. Per the swarm's own "REVERT ~0-gain" rule, **deferred** until/
unless a real workload shows the mel frontend on its critical path. Design +
measurement preserved here and in the scratchpad so it can be landed in minutes
if that changes.

### L3 — sparse mel-filterbank projection (bit-exact)  — `src/native_engine/mel.rs`

**Hypothesis.** Real whisper mel filterbanks are sparse triangles: each of the 80
filters is nonzero over only **~5 of the 201** FFT freq bins. The projection loop
ran densely over all 201 bins per filter regardless — ~97.5% of the multiply-adds
were `power[k] * 0.0`. Skipping the leading/trailing zeros is **bit-exact**: for
the finite non-negative `power` an FFT of real audio produces, `power[k] * 0.0 ==
+0.0`, which never changes a running f64 sum (and the accumulation order over the
nonzero range is unchanged).

**Change.** Precompute each filter's `[start, end)` nonzero range once per
`log_mel` (bundled with the bank in `SparseMelFilters`, keeping
`compute_frame_column` under the 7-arg clippy limit); project only over that
range.

**Conformance.** New test `sparse_projection_matches_dense_bit_exact` asserts
byte-identical f64 sums (range-restricted vs full 201-bin dense) across 16
filters × 64 random non-negative power spectra. The existing mel tests
(silence/determinism) stay green (output unchanged). The hermetic `mel_30s`
(dense synthetic bank) is unaffected; new bench `mel_30s_realistic` (sparse
triangular bank, the production case) captures the win.

**Measurement (standalone local same-process A/B — rigorous given 5.6× rch worker
variance — over a realistic 80×201 triangular bank, 3000 frames):**

| projection (3000 frames) | time | speedup |
|---|---|---|
| dense (all 201 bins/filter) | 37.5 ms | — |
| sparse (~4.9 nonzero bins/filter) | 2.9 ms | **12.78×** |

Bit-exact check in the same harness: **0 / 240,000** mismatches. Since the dense
projection (37.5 ms) is *larger* than the post-L1 FFT pass (~28 ms), eliminating
it is **≈2× on the whole mel frontend for real (sparse-bank) models** —
a genuine real-workload win, unlike L2. **Verdict: KEEP.**

### L4 — frame-batched SIMD FFT (bit-exact)  — `src/native_engine/mel.rs`

**Hypothesis.** After L1+L3 the FFT is the dominant mel cost. Frames are
independent and identically-shaped, so they vectorize *vertically*: put one frame
per SIMD lane (`Simd<f32, 8>`, structure-of-arrays) and run one batched FFT over
8 frames. IEEE-754 f32 lane ops are bit-identical to scalar f32 (no FMA
contraction), so lane `L` equals the scalar FFT of frame `L` — **bit-exact**,
not an approximation. (This is a *vectorization* axis, orthogonal to L1/L3's
arithmetic-redundancy elimination — the "bit-exact floor" is lower than L3
implied.)

**Change.** `fft_simd8` / `dft_simd8` mirror the scalar recursion over
`Simd<f32, 8>` with the same precomputed twiddles (splatted). The mel worker
batches fully-valid frames (full `N_FFT` window) 8-at-a-time; the partial-window
tail + noise-floor frames keep the scalar path. After the batched FFT each lane
is transposed back and fed to the shared, tested `power_and_project` — so the
columns are byte-identical to the scalar path. Needs `#![feature(portable_simd)]`
(crate is nightly; stays `#![forbid(unsafe_code)]` — std::simd is safe).

**Conformance.** New test `fft_simd8_matches_scalar_bit_exact` asserts
byte-identical output per lane vs the scalar FFT (32 rounds × 8 frames × 802
bins); existing silence/determinism mel tests stay green.

**Measurement (standalone local same-process A/B, 3000-frame `N_FFT=400` pass —
rigorous given 5.6× rch worker variance):**

| FFT pass (3000 frames) | time | speedup |
|---|---|---|
| scalar (per-frame) | 26.7 ms | — |
| SIMD f32×8 (baseline x86-64) | 6.3 ms | **4.22×** |
| SIMD f32×8 (AVX2) | 4.5 ms | **5.62×** |

Bit-exact: **0 / 2,400,000** mismatches. Since the FFT dominates the post-L3 mel
frontend, this is **~2.5–3× on the whole mel frontend** on top of L1+L3.
**Verdict: KEEP.**

**In-tree cumulative result (criterion `native_engine/mel`, post L1+L3+L4):**

| bench | time | notes |
|---|---|---|
| `mel_30s` (dense synthetic bank) | **12.8 ms** | L1+L4 only (dense bank can't use L3); was 269 ms pre-L1 |
| `mel_30s_realistic` (sparse triangular bank = **production**) | **3.95 ms** | full L1+L3+L4 stack |

So a real model's 30 s log-mel frontend now runs in **~4 ms** (from a 269 ms
dense/transcendental-heavy starting point — a **~68× cumulative** reduction on the
hermetic frontend, all bit-exact). e2e share remains bounded by encoder/decoder.

### L5 — vertical-SIMD `layer_norm` (bit-exact)  — `src/native_engine/nn.rs`

**Hypothesis.** `layer_norm` runs in every encoder + decoder block. Its per-row
f64 mean/var reductions can't use *horizontal* SIMD (that reorders the f64 sum →
not bit-exact), but the L4 *vertical* trick applies: one row per `f64x8` lane, so
each lane reduces its own row in the original ascending order. IEEE-754 f64 lanes
+ correctly-rounded `sqrt`/division are bit-identical to scalar f64 ⇒ **bit-exact**
(unlike `gelu`/`softmax`, whose `tanh`/`exp` have no bit-exact SIMD form).

**Change.** Factor the per-row body into `norm_rows`, which gathers 8 rows into a
structure-of-arrays, computes mean/var/inv-std/affine in `f64x8`, and scatters
back; the `< 8`-row tail stays scalar. Both the serial and band-parallel paths
call it, so SIMD stacks with the existing thread fan-out. Reuses the L4
`#![feature(portable_simd)]` gate (still `#![forbid(unsafe_code)]`).

**Conformance.** New test `layer_norm_simd_matches_scalar` asserts byte-identical
output vs an independent scalar per-row f64 reference across row counts
{1,7,8,9,20,33} (covers SIMD groups + tail); existing layer_norm tests stay green.

**Measurement (standalone local same-process A/B, `[1500, 384]` encoder-window
shape; rigorous given 5.6× rch worker variance):**

| layer_norm `[1500,384]` | time | speedup |
|---|---|---|
| scalar per-row | 1.20 ms | — |
| vertical `f64x8` (baseline x86-64) | 0.61 ms | **1.97×** |
| vertical `f64x8` (AVX2) | 0.47 ms | **2.33×** |

Bit-exact: **0 / 576,000** mismatches. ~2× on a real per-layer activation op
(runs ×4 encoder + ×N decoder layers), bit-exact. New `layer_norm_1500x384`
bench makes it a standing in-repo instrument. **Verdict: KEEP** (modest e2e share
— still encoder/decoder-GEMM-bound — but a real, measured, bit-exact win and the
last nn kernel amenable to bit-exact vectorization).

### L6 — re-tune `layer_norm` PAR_THRESHOLD post-SIMD  — REJECTED (~0-gain)

**Hypothesis.** L5's SIMD made `layer_norm`'s compute ~2× cheaper, so the
`thread::scope` spawn cost might now dominate at the encoder shape `[1500,384]`,
arguing to raise `PAR_THRESHOLD` and run it serial-SIMD (a pure bit-exact
scheduling knob).

**Measured (standalone, same host, 8 workers):**

| shape | serial-SIMD | parallel-SIMD | winner |
|---|---|---|---|
| `[1500,384]` (encoder) | 0.70 ms | 0.79 ms | serial **1.0–1.13×** (within noise) |
| `[3000,384]` | 1.42 ms | 1.21 ms | parallel **1.17×** |

**Verdict: REJECTED.** The crossover already sits right around the production
encoder shape, so the existing `PAR_THRESHOLD = 1<<16` is well-tuned; raising it
would buy ≤1.1× at `[1500,384]` (noise) while *hurting* larger shapes. Per
REVERT-~0-gain, not shipped. (The slow in-tree `layer_norm_1500x384` = 3.3 ms on
ovh-b was worker variance, not spawn overhead.)

---

## Measurement infrastructure findings (2026-06-24, BlackThrush)

These shape what is measurable and how the ratios above must be read.

1. **Worker variance ≈ 5.6×.** `mel_30s` (identical code) measured **29 ms**
   (vmi1149989), **63 ms** (ovh-a), **164 ms** (vmi1152480). rch assigns workers
   per invocation and exposes **no pinning flag**, so **cross-run criterion
   `--baseline` is invalid** unless both runs land on the same worker. L1's 9.2×
   is trustworthy precisely because baseline + candidate both ran on vmi1149989.
   **Rule:** only same-worker (single-`rch exec`) A/B is admissible.

2. **Real-workload benches are unmeasurable via `rch`.**
   `encoder_window_*`, `decoder_token_step_*`, `e2e_tiny_jfk`, `logits_gemv_large`
   all SKIP on remote workers: the ggml model and `jfk.wav` are **gitignored**
   (`*.wav`, model dirs) so rch does not sync them to the worker. The native
   engine never downloads. ⇒ The big head-to-head workloads can only be measured
   **locally** (assets present) with `$FRANKEN_WHISPER_MODEL_DIR` pointed at
   `legacy_whispercpp/whisper.cpp/models`. Blocker bead: **bd-ms0x** (large model)
   + new bead for the rch-sync gap.

3. **No built `whisper.cpp` comparator.** `whisper-cli`/`main` is not built on
   this host (only source under `legacy_whispercpp/whisper.cpp`). A true
   wall-clock head-to-head vs the original requires building it first
   (cmake) — harness work tracked under bd-zk43 / bd-0hnz (IcyWren).

4. **Hermetic f16_gemv baselines** (ovh-a, for future levers):
   `1280×1280 = 419 µs (3.9 Gelem/s)`, `384×384 = 137 µs (1.07 Gelem/s)`. The
   small 384×384 (tiny.en per-token Linear) is ~4× lower throughput — a possible
   future lever, but `gemv_f16` is already SIMD + band-parallel, so a bit-exact
   gain there is uncertain.

**Bit-exact-lever feasibility map.** The mel twiddle win was a sweet spot:
constant (data-independent) transcendentals, precomputable exactly. The other
hot kernels are NOT: `softmax`(exp), `gelu`(tanh), `layer_norm`(reduction) all
have **data-dependent** transcendentals / order-sensitive f64 sums — any speedup
(approx exp/tanh, reordered reduction) changes output bits and breaks the
whisper.cpp conformance contract. Encoder GEMM is FrankenTorch's (external
crate). So further *bit-exact* native-engine levers are limited; the largest
remaining honest wins require the local-measurement unblock (item 2) and the
`whisper.cpp` comparator (item 3).
