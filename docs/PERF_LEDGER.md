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

### L7 — x86-64-v3 build baseline (AVX2/FMA)  — `.cargo/config.toml`  **[e2e win]**

**Hypothesis.** The build used the Rust default target (`x86-64`, SSE2 only),
leaving AVX2/FMA unused by *all* code — the SIMD native engine AND, crucially,
**FrankenTorch's sgemm, which is ~99% of e2e** (encoder + decoder GEMM/GEMV). The
first profile of the real workloads exposed this: e2e_tiny_jfk = 708 ms = mel
~4 ms + **encoder 263 ms (37%) + decoder 441 ms (62%, ~15 ms/token)** — all
GEMM/gemv-bound. `#![forbid(unsafe_code)]` rules out runtime `#[target_feature]`
dispatch, so a build-wide CPU baseline is the only safe way to enable these
instructions.

**Change.** `.cargo/config.toml` → `rustflags = ["-C", "target-cpu=x86-64-v3"]`
(AVX2+FMA+BMI, Haswell-2013+).

**Measurement (local same-host A/B, tiny.en; first lever to move e2e):**

| `native_engine_bench` | SSE2 (default) | x86-64-v3 | speedup |
|---|---|---|---|
| `encoder_window_tiny` | 263 ms | 204 ms | **1.29×** |
| `decoder_token_step_tiny` | 122 ms | 102 ms | **1.20×** |
| **`e2e_tiny_jfk`** (full 11 s transcription) | 708 ms | **633 ms** | **1.12×** |

**Conformance.** Transcription-level (per `conformance-contract.md`), not
bit-exact — AVX2/FMA changes f32 rounding but `native_engine_e2e` is **6/6 green**
under the flag (transcription unchanged). **Verdict: KEEP.** First and only lever
to move the e2e-dominant GEMM. **Trade-off:** raises min CPU to AVX2 (2013+);
revert = delete `.cargo/config.toml` (or use `x86-64-v2`). The bit-exact
kernel levers (L1/L3/L4/L5) stack *on top* — they make the non-GEMM parts faster
within this baseline.

### L8 — vectorized gelu/softmax (AVX2 minimax exp/tanh)  — MEASURED, REVERTED (~0 e2e)

**Hypothesis.** Scalar `libm` `tanh`/`exp` in `gelu`/`softmax` looked like ~30%
of the encoder (a single isolated gelu over `[1500,1536]` is 15.2 ms scalar vs
4.3 ms vectorized = **3.56×**, with an accurate `exp_simd` at 7.9e-8 rel error).

**Measured in-tree (clean v3 A/B, `e2e_tiny_jfk`):** **632.6 ms (v3) → 647 ms
(v3 + vectorized gelu/softmax)** — **~0 gain, marginally negative.** The isolated
3.56× did NOT translate: gelu/softmax are a *small* fraction of the
GEMM-dominated encoder/decoder (my ~30% estimate was wrong — the FrankenTorch
sgemm dominates), so vectorizing them moves e2e by noise. Conformance was green
(200/200 lib tests incl. an accuracy test, native_engine_e2e 6/6), so it was
*correct*, just not *worth it*.

**Verdict: REVERTED** (commit b42ce64 → reverted) per the swarm's "REVERT ~0-gain"
rule. Lesson recorded so it isn't re-attempted: **isolated-kernel speedups must be
validated at e2e before landing** — the encoder/decoder are GEMM-bound, so only
the GEMM (FrankenTorch, external) or the build baseline (L7) move e2e here.

---

### L9 — decoder GEMV PAR_THRESHOLD 1<<19→1<<21 (spawn-bound MLP)  — `src/native_engine/nn.rs`  **[e2e win]**

**How it was found.** The 2026-06-25 whisper.cpp head-to-head (bd-zk43) showed
franken's DECODER is ~2× slower than whisper.cpp (the encoder/mel already win).
`decoder_attrib` (tiny.en, 400 steps, real load) pinpointed it: `mlp_fc_gelu` =
**5.14 ms/tok (35%, 0.23 GFLOP/s)** — absurd for 1.18 M MACs → **spawn-bound, not
compute-bound**. The MLP GEMVs (`[384,1536]`/`[1536,384]` = 590 k MACs) sit *just*
over the old `1<<19` (524 k) threshold, so each spawned 8 `thread::scope` threads
per token; 590 k split 8 ways is ~20 µs compute/thread vs tens of µs spawn/join.
(whisper.cpp avoids this with a persistent thread pool.)

**Fix.** Raise `PAR_THRESHOLD` to `1<<21` (2 M) in both GEMV paths, so the
per-token mid-size Linears run serial while the logits GEMV (20 M) and large-model
Linears (6.5 M) stay parallel. Pure scheduling knob → **bit-identical**.

**Measured (local v3 A/B):** `decoder_attrib` `mlp_fc_gelu` 5.14→**2.81 ms/tok
(−45%)**, total 14.67→12.32 ms/tok (−16%); **`e2e_tiny_jfk` 614→571 ms = −9.5%
(criterion p<0.05, "improved")**. Narrows the whisper.cpp gap 1.37×→1.27×.
**Verdict: KEEP.** Follow-up (same tick): the *other* decoder subs that looked
spawn-bound in `decoder_attrib` do NOT translate to the e2e — both MEASURED and
REJECTED:
- `project_qkv` serial (was 1.64 ms/tok in attrib): e2e **566 vs 571 ms, p=0.55
  (~0)** → reverted, kept concurrent (helps large models).
- `cross_attn` 1<<13→1<<14 (tiny serial; was 2.93 ms/tok in attrib): no-ts e2e
  **+2.7%, p<0.05 (REGRESSED)** → reverted, parallel path is genuinely faster.

Lesson: **`decoder_attrib`'s tight 400-step loop over-states per-call spawn cost**
vs the real e2e (decode interspersed with mel/encode). Only the MLP GEMV
threshold (L9, validated on the e2e) was a real spawn win; a blanket persistent
thread pool is NOT obviously worth it. The remaining franken-vs-whisper.cpp
decoder gap (1.27×) is now compute-bound (GEMV/sgemm/softmax), not spawn-bound.

---

### L10 — m=1 GEMV fast path in `nn::matmul` (skip ft sgemm for tq=1 attn)  — `src/native_engine/nn.rs`  **[e2e win]**

**How it was found.** With spawn ruled out (L9 + follow-ups), the decoder gap is
compute. `nn::matmul` routed *everything* through `ft_kernel_cpu` sgemm — including
the per-token decode attention matmuls, which at tq=1 are GEMV-shaped
(`[1,d]×[d,tk]` scores, `[1,tk]×[tk,d]` out). Standalone (x86-64-v3) showed ft
sgemm pays huge packing/dispatch overhead at m=1: `[1,64]×[64,1500]` **sgemm 46 µs
vs direct gemv 4.5 µs (10.2×)**; `[1,1500]×[1500,64]` **48 vs 6.3 µs (7.6×)**.
(GGML/whisper.cpp use a dedicated dot here — this is a real slice of the decoder
gap.)

**Fix.** Add an `m == 1` branch to `nn::matmul`: row-broadcast SAXPY accumulation
over k (`out += a[k]*b[k,:]`, LLVM → AVX2 FMA), skipping sgemm packing entirely.
Helps every m=1 caller (cross_attn + self_attn). NOT bit-identical (different
summation order, max abs diff ~1e-6/2.7e-5) → relies on the transcription-level
contract.

**Measured (local v3):** `e2e_tiny_jfk` 571→**561 ms (ts)** / 543→**534 ms
(no-ts)** = **−1.7%**; whisper.cpp gap 1.21×→**1.19×** (no-ts). **Conformance
GREEN** (native_engine_e2e 6/6). **Verdict: KEEP.** Modest at e2e (the attn
matmuls are a small slice; the mlp/logits use the separate f16 GEMV path), but a
free, correct win and the right structural fix.

---

### L11 — rayon persistent-pool `gemv_f16` (re-parallelize the mlp w/o spawn)  — `src/native_engine/nn.rs`  **[e2e win]**

**The insight.** L9 serialized the per-token mid GEMVs because `std::thread::scope`
*per-call spawn* dominated their compute under load. But serial leaves 7 of 8 cores
idle on the mlp — whisper.cpp uses a PERSISTENT thread pool (no per-call spawn) and
keeps the parallelism. franken used `thread::scope` everywhere (no persistent pool).

**Fix.** Add `rayon` (already in-tree via ft-kernel-cpu) and dispatch `gemv_f16`'s
parallel path via `par_chunks_mut` over output-row bands (rayon's global pool — no
per-call spawn), and drop the threshold back `1<<21`→`1<<19` so the mlp (590 k) +
logits (20 M) re-parallelize while the tiny `[384,384]`=147 k stay serial.
**Bit-identical** (disjoint output-row bands, each row's `dot8` order unchanged;
standalone maxdiff 0).

**Measured.** Standalone (contended host) rayon vs serial gemv: `[1536,384]` 1.40×,
`[384,1536]` 1.35×. In-tree: **`e2e_tiny_jfk` 561→542 ms (ts) / 534→523 ms
(no-ts) = −3.4% / −2.1%**; **conformance GREEN** (native_engine_e2e 6/6). whisper.cpp
gap 1.19×→**1.17×** (no-ts). **Verdict: KEEP.** rayon's persistent pool is the
correct structural answer to the per-call-spawn problem L9 worked around; supersedes
L9's serial-mlp compromise (threshold restored, dispatch via the pool).

*Band-size follow-up (MEASURED, REJECTED):* finer chunks (`workers*4`, min 64
rows) to let rayon work-steal on a contended host — hypothesis that a 1-chunk/core
split stalls when a core is busy with another process. no-ts e2e **+3.7%
(REGRESSED)**: the extra rayon task + per-chunk scratch-alloc overhead outweighs
the work-steal benefit at these sizes. `band = out/workers` is optimal; kept.

---

### L12 — rayon persistent-pool cross-attn head dispatch  — `src/native_engine/decoder.rs`  **[e2e win]**

**Insight.** Extending L11 to the cross-attention wrapper. The no-timestamps decode
path (record off — the apples-to-apples vs whisper.cpp's `dtw=0`) parallelized
cross-attn over heads with `std::thread::scope` **per token** (6 head-threads ×
~28 tokens). Like the mlp (L9/L11), that per-call spawn was the bottleneck, not the
compute (serializing it had REGRESSED +2.7%, so parallelism is needed — just
without the spawn).

**Fix.** Dispatch the head bands via rayon's persistent pool
(`band_starts.into_par_iter()`), each band scattering into a private buffer →
disjoint-merge. **Bit-identical** (every position written by exactly one head;
compute_head/scatter capture only shared refs).

**Measured (local v3, no-ts e2e):** **523→477–491 ms = −6 to −8.8%** (contention-
dependent); **conformance GREEN** (native_engine_e2e 6/6). The ts path is
unchanged (it uses the serial `record` branch, not this parallel path). whisper.cpp
gap **1.17×→~1.07–1.10× (NEAR PARITY)**. **Verdict: KEEP.**

---

### L13 — rayon cross-attn for the RECORD (timestamps) path  — `src/native_engine/decoder.rs`  **[e2e win]**

**Insight.** L12 only sped the no-ts path; the realistic default (`timestamps:true`,
DTW word alignment) took the serial `record` branch because per-head softmax
`scores` must land in `recorded` in head order. But the *compute* can still be
parallel — only the recording needs ordering.

**Fix.** Parallelize `compute_head` over heads via rayon (persistent pool), collect
in head order, then push `scores` + scatter SERIALLY. `compute_head` never touches
`recorded` → Sync; ordering + disjoint scatter unchanged → **bit-identical** (DTW
timestamps green).

**Measured (local v3, ts e2e):** **542→504 ms = −7%**; **conformance GREEN**
(native_engine_e2e 6/6). **Verdict: KEEP.** Now both decode paths (ts + no-ts) get
parallel cross-attn.

### L14 — cap Rayon default pool to native default_threads()  — `src/native_engine/mod.rs`

**How it was found.** Current head (`a9ecb3b`) ran on a 64-way host. The native
engine's own default is capped at 16 threads, and its glue kernels are tuned
around 8-16 workers, but Rayon defaulted to all 64 host threads when
`RAYON_NUM_THREADS` was unset. A same-binary surface sweep showed the issue:
loaded `tiny.en` JFK at `threads=8` had median-after-warmup **0.624 s** with the
default pool, while `RAYON_NUM_THREADS=16` measured **0.547 s**. The 4/8/12/16
sweep showed 16 was the best tested cap; 4 regressed badly.

**Fix.** Before the first native inference kernels run, initialize Rayon's
global pool to [`default_threads()`] (16 on this host) when the operator has not
already set `RAYON_NUM_THREADS`. Explicit `RAYON_NUM_THREADS` remains an override;
if another embedding app already initialized Rayon, `build_global`'s error is
ignored and behavior remains unchanged. This is pure scheduling: no numeric
order inside any output row changes.

**Measured (local same-host, current-head A/B, `native_ab tiny.en 9 <threads>`,
discard run 0):**

| loaded-model path | baseline median | L14 median | speedup |
|---|---:|---:|---:|
| 4 threads | 0.603520 s | 0.540470 s | **1.117×** |
| 8 threads | 0.624235 s | 0.535540 s | **1.166×** |

Decoder attribution agreed directionally: 13.064→11.878 ms/token, mainly from
`logits_gemv` and `cross_attn` moving to the right-size persistent pool. Output
proof: baseline and L14 `native_ab` JSON outputs are byte-identical at both 4
and 8 threads.

**OpenAI Whisper boundary (same host):** one-shot CLI comparator improved from
**3.20×** to **4.23×** faster than OpenAI Whisper CLI. Loaded API boundary is
mixed: L14 beats OpenAI loaded API at 4 threads (**1.078×**) but still loses at
8 threads (**0.784×**, franken 1.275× slower). **Verdict: KEEP.** This is the
first post-L13 in-crate e2e win; it narrows but does not eliminate the loaded
OpenAI 8-thread gap.

---

## ⇒ Session arc (2026-06-25, BlackThrush): built the comparator, closed 1.37×→~1.08×

Building `whisper-cli` (bd-zk43) exposed the real gap as the **in-scope decoder**
(not the encoder, which already wins 204 vs 242 ms). FIVE bit-identical/
transcription-green wins followed — all whisper.cpp/GGML techniques franken lacked
(spawn-bound dispatch → persistent pool; sgemm-for-gemv → dedicated dot):

| lever | what | e2e |
|---|---|---|
| L9 | mlp GEMV spawn threshold | no-ts ~590→543 ms |
| L10 | m=1 gemv (skip sgemm packing) | no-ts 543→534 ms |
| L11 | rayon persistent-pool gemv_f16 | no-ts 534→523 ms; ts 561→542 |
| L12 | rayon persistent-pool cross-attn (no-ts) | no-ts 523→**477–491 ms** |
| L13 | rayon cross-attn (ts/record path) | ts 542→**504 ms** |

**franken_whisper tiny.en jfk vs whisper.cpp: no-ts 1.37×→~1.07–1.10× (near
parity); ts (realistic, with word timestamps) 614→504 ms (−18%)** — all
conformance-green. Remaining to *win outright*: bd-4hc0 (encoder
`matrixmultiply→gemm`, out-of-scope) would cut the encoder ~2×.

## Conformance-level finding — bit-exact was stricter than required (BlackThrush)

`docs/conformance-contract.md`: **"Compatibility is *not* byte-for-byte identical
output"** — the contract is **transcription-level** (exact/normalized text +
≤50 ms timestamp tolerance + speaker/confidence bands), enforced by
`tests/conformance_harness.rs`. All L1/L3/L4/L5 levers were **bit-exact** (zero
risk, correct), but that is *stricter* than the contract requires. Implications
for future levers:

- **rFFT / split-radix mel is contract-permitted** (no approval needed) — but mel
  is already ~4 ms post-L1/L3/L4, i.e. **<2% of e2e** (encoder/decoder-bound), so
  a further ~2× there is REVERT-~0-gain. Not pursued.
- **INT8-quantized GEMV — MEASURED, REJECTED.** Accuracy is fine (int8 vs f32
  max rel error 0.4%; whisper.cpp Q8_0 confirms int8 preserves WER), but a SAFE
  `std::simd` int8 GEMV (widen i8→i32, no VNNI) clocks **0.24× — ~4× SLOWER** than
  the f16/f32-dot path at both baseline and AVX2 (`int8_gemv.rs`). The int8 speed
  win needs `vpdpbusd` (VNNI) intrinsics, which are **unsafe → forbidden by
  `#![forbid(unsafe_code)]`**; the f16 path already uses hardware `f16c` dequant
  safely. **DEAD under the safe-code constraint.**
- **Approximate-transcendental `gelu`/`softmax` (SIMD `exp`/`tanh`)**: legal under
  the contract, but they're small vs the GEMM (GEMM-bound e2e) and carry
  transcription risk needing local-e2e verification → marginal EV.

- **Explicit FMA (`mul_add`) in the gemv `dot8` — MEASURED, REJECTED (regression).**
  The decoder is 62% of e2e and runs `gemv_f16`/`dot8` (separate mul+add, since
  Rust doesn't auto-contract). Hypothesis: explicit `mul_add` under the +fma
  baseline (L7) would speed the decoder core. Standalone (logits shape
  51864×384, x86-64-v3): explicit `mul_add` dot = **0.791× — SLOWER** than the
  current mul+add. LLVM already lowers the 8-accumulator mul+add optimally (and
  contracts where it helps); forcing `mul_add` hurts. The decoder gemv is already
  optimal; **REJECTED**.

- **Vertical-layout gemv (bd-n0m3) — MEASURED, REJECTED (~0-gain).** Hypothesis:
  store the logits f16 weight interleaved `[OUT/8, INP, 8]` so the gemv vertically
  vectorizes 8 output rows into f32×8 accumulators (no per-row horizontal
  reduction) — a different organization than the current per-row `dequant+dot8`.
  Standalone with real f16c dequant (logits shape 51864×384, x86-64-v3):
  current 4154 µs vs vertical 4046 µs = **1.03×** (max abs diff 4e-6,
  transcription-level). The current per-row dequant+dot8 is already within 3% of
  the alternative organization → not worth the load-time relayout + kernel
  rewrite. Confirms the decoder gemv is mature regardless of layout; **REJECTED**.

- **Encoder QKV-projection fusion — MEASURED component (1.14×), net ~0 at e2e,
  NOT PURSUED.** Encoder attention does Q/K/V as 3 separate `matmul_bias` calls on
  the same LHS `h` (encoder.rs:426-428); `matrixmultiply` re-packs `h` per call, so
  fusing into one `[1500,384]×[384,1152]` saves 2 re-packings — standalone measured
  **1.14×** on the QKV proj (16884→14791 µs, contended; bit-identical since sgemm
  output columns are independent). But integration negates it: the fused output
  `[1500,1152]` must be split back to q/k/v `[1500,384]` (3 strided copies ≈
  6.9 MB/layer ≈ 1.4 ms/4 layers), eating most of the saving; and QKV is only
  ~20-30% of the encoder → net **~0–0.5% e2e** (within bench noise). Classic
  component-win-vanishes-at-integration (cf. L8). Deferred as not worth the change.
  NB: the win is *matrixmultiply's per-call repacking overhead* — another cost the
  `gemm`/faer swap (bd-4hc0) removes structurally, reinforcing that lever.

- **Decode-loop full-vocab logsumexp vectorization — MEASURED, REJECTED (~0).**
  `compute_logprobs` (decode.rs) runs a log-softmax over ALL 51 864 logits per
  token — ~1.45 M scalar `libm` `exp` over the decode — which *looks* like a fat
  lever. Vectorized the logsumexp loop with an 8-wide minimax `exp_simd`
  (`nn::logsumexp_over_finite`, ~7.9e-8 rel). Clean back-to-back A/B (no-ts e2e,
  `--baseline`): **−0.32%, p=0.46 — "no change"** (a spurious −1.8% on one ts run
  was contention noise). Reason: modern `libm` `expf` is ~5–7 ns, so the loop is
  only ~7–10 ms total (~1.5%), within bench noise, and `compute_logprobs`'s
  output `Vec` (needed by the ts timestamp-pairing) isn't the bottleneck either.
  **REVERTED** (conformance was 6/6 green, so it was *correct*, just ~0). Don't
  re-attempt: the per-token full-vocab `exp` is not a real e2e cost here.

**Net (measured, not assumed):** `#![forbid(unsafe_code)]` (no VNNI) + the
e2e-dominant GEMM living in FrankenTorch (external crate `ft-kernel-cpu`, which
hardcodes `matrixmultiply 0.3` with no feature knob) cap the kernel-level wins in
this crate. The lever space is **exhaustively exhausted by measurement**: 5
shipped (L1/L3/L4/L5 mel bit-exact + **L7 x86-64-v3 = the 1.12× e2e win**), 5
measured-and-rejected (L2 ~0-e2e, L6 ~0-gain, L8 ~0-e2e, INT8 0.24×, gemv-FMA
0.791×). e2e is encoder-GEMM-bound (external) + decoder-logits-bandwidth-bound
(40 MB f16/token, fundamental). Further e2e wins require FrankenTorch-side GEMM
work (`matrixmultiply` → `gemm`/faer, ~1.5–3×) or lifting `#![forbid(unsafe_code)]`
for VNNI int8 — **both out of `franken_whisper`'s crate**.

## ⇒ Biggest remaining e2e lever, MEASURED: the GEMM has 3.75× headroom (bd-4hc0)

The e2e wall is the encoder GEMM, delegated to `ft_kernel_cpu::matmul_tensor_
contiguous_f32`, which uses **`matrixmultiply 0.3`**. Standalone A/B (x86-64-v3,
rayon) for the encoder MLP shape `[1500,384]×[384,1536]`:

Full per-shape profile (standalone same-run A/B; ratios are the signal — absolute
GFLOP/s drops under box contention, e.g. the uncontended fc1 run hit
187→701 GFLOP/s = 3.75×):

| encoder GEMM shape | `gemm`/faer vs `matrixmultiply` |
|---|---|
| attn Q/K/V/out `[1500,384]×[384,384]` | **3.14×** |
| MLP fc1 `[1500,384]×[384,1536]` | **2.24× – 3.75×** (uncontended) |
| MLP fc2 `[1500,1536]×[1536,384]` | **1.46×** (larger K → smaller gap) |

So EVERY encoder GEMM is faster on `gemm`/faer — `matrixmultiply` is consistently
the bottleneck. The GEMM is ~most of the GEMM-bound encoder (~32% of e2e), so it
is **~1.5–3.75× off achievable** (shape-dependent; weighted ~2–3×). Swapping `ft-kernel-cpu`'s `matrixmultiply`→`gemm` is **~2× encoder
→ ~1.2× e2e** for franken_whisper, and benefits every FrankenTorch user.
`ft-kernel-cpu` already calls `matrixmultiply` via `unsafe`, so `gemm`'s unsafe
API is fine there; `franken_whisper`'s `#![forbid(unsafe_code)]` blocks calling
`gemm` directly (and `faer`'s safe API is a heavy dep), so the clean fix lives in
**ft-kernel-cpu** (out of `franken_whisper-cc`'s scope). **bd-4hc0 (P0).** This
turns "the GEMM is external, untouchable" into "the GEMM has a measured 3.75×,
here's exactly where."

## Measurement infrastructure findings (2026-06-24, BlackThrush)

These shape what is measurable and how the ratios above must be read.

1. **Worker variance ≈ 5.6×.** `mel_30s` (identical code) measured **29 ms**
   (vmi1149989), **63 ms** (ovh-a), **164 ms** (vmi1152480). rch assigns workers
   per invocation and exposes **no pinning flag**, so **cross-run criterion
   `--baseline` is invalid** unless both runs land on the same worker. L1's 9.2×
   is trustworthy precisely because baseline + candidate both ran on vmi1149989.
   **Rule:** only same-worker (single-`rch exec`) A/B is admissible.

2. **Real-workload benches are unmeasurable via `rch` — RESOLVED via local builds
   (bd-7xbq closed).** `encoder_window_*`, `decoder_token_step_*`, `e2e_tiny_jfk`,
   `logits_gemv_large` SKIP on remote workers: the ggml model and `jfk.wav` are
   **gitignored** (`*.wav`, model dirs) so rch does not sync them. **Working
   path (proven):**
   ```
   RCH_MIN_LOCAL_TIME_MS=99999999 \      # forces rch to build LOCALLY (no offload)
   CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cc-local \
   FRANKEN_WHISPER_MODEL_DIR=.../legacy_whispercpp/whisper.cpp/models \
   cargo test -p franken_whisper --release --test native_engine_e2e
   ```
   Built locally in **5m52s** (this host's nightly compiles `ft-kernel-cpu` fine —
   the `ovh-a` `stdarch_neon_dotprod` failure is worker-specific drift) and ran
   **6/6 gated pipeline tests that actually transcribed jfk** via the native
   tiny.en engine (no SKIP) — i.e. **transcription conformance is verifiable
   locally**. This is the gateway for any non-bit-exact lever AND the e2e
   head-to-head. `large-v3-turbo` still absent (bd-ms0x).

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
