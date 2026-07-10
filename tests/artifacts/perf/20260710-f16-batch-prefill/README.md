# PARKED — f16 batch GEMV scheduler: row-morsel is the wrong scheduler for fc2 prefill

**Status: PARKED. Patch + bench saved, NOT applied.** This is the re-attack of the family
reopened by the 2026-07-10 ledger-integrity audit (`bd-f16-gemv-weight-stationary-reopen-ugyh`).
Blocked on `rch` being unable to sync this repo (below). Filed by `cc_fw`.

Files here:
* `scheduler_knob.patch` — adds `nn::set_batch_gemv_row_morsel(bool)` (AtomicBool + setter,
  default unchanged) so a bench can flip the scheduler **inside one binary** and time the
  **real** `nn::gemv_f16_batch` both ways. An env var is read once per process and cannot do
  that, and a two-invocation A/B is inadmissible.
* `f16_batch_prefill.rs` — the bench. Both arms call the real `nn::gemv_f16_batch`; they
  differ only by the setter. True interleaving (`paired()`, arms alternated *inside* one
  measured routine), full-output `black_box` consumption, a null control (arm-vs-itself), and
  a bit-exactness assert.

Apply with:
```
cd /data/projects/franken_whisper
git apply tests/artifacts/perf/20260710-f16-batch-prefill/scheduler_knob.patch
cp tests/artifacts/perf/20260710-f16-batch-prefill/f16_batch_prefill.rs benches/
# then add   [[bench]] name = "f16_batch_prefill" harness = false   to Cargo.toml
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench -p franken_whisper --bench f16_batch_prefill
```

---

## What the audit reopened, and where the code actually runs

The 2026-07-09 REJECT of a "weight-stationary token-block tile" benched
`[1500,1280]×[1280,1280]` — the **cross-K/V** shape — on the premise that `cross_attn_k/v`
route to `gemv_f16_batch`. They do not: `a674b49` (2026-07-02) flipped
`cross_proj_f32_enabled()` **default-ON**, sending cross-K/V through f32 sgemm.

`gemv_f16_batch` has exactly **one** production caller (`decoder.rs:345`; `nn.rs:5558` is a unit
test), reached only by `WeightMat::F16` linears at `tq>1` without `w_i8`. Per `decoder.rs:312`,
**`mlp_2` (fc2)** is excluded from the int8 batch path because it carries a `w_i8_block` copy —
*"those stay on the f16 batch path."* So the real consumer is **fc2 at prefill**:
`out = n_state = 1280`, `inp = n_mlp = 5120`, `tq` = prompt length.

## The real defect (structural; no bench needed to see it)

`gemv_f16_batch` has two schedulers, and both are proven bit-identical to per-token `gemv_f16`
(`gemv_f16_batch_equals_per_token_gemv`, `gemv_f16_batch_row_morsel_equals_per_token_gemv`), so
choosing between them **cannot change results**:

| scheduler | loop | weight traffic | intensity (flop per weight byte) |
|---|---|---|---|
| **column-band** (`compute_band`) | `o` outer, `t` inner | streamed **once** | `tq` |
| **row-morsel** (`gemv_f16_batch_rows`) | `t` outer per band, `o` inner | **re-streamed once per band** | `tq / workers` |

Row-morsel is selected when `work = tq·out·inp ≥ COMPUTE_BOUND_MACS (1<<26)`. **That gate
conflates "compute-bound" with "big weight."** It was tuned for cross-K/V (weight
`1280·1280·2` = **3.3 MB**, 2.4 GFLOP — genuinely compute-bound, so the extra weight passes are
free). fc2's weight is `1280·5120·2` = **13.1 MB**, 4× larger, at a far smaller `tq`.

For fc2, `out < 1<<14` ⇒ `gemv_worker_count` caps at 8; but `work ≥ 1<<26` promotes to
`avail.min(16)`. `row_band = ceil(tq/workers)`:

| prefill `tq` | `work ≥ 1<<26`? | `row_band` | bands | weight traffic | M2col fires? (`row_band ≥ 2`) |
|---|---|---|---|---|---|
| 10 | no → column-band | — | — | **13.1 MB** | n/a |
| **12** | yes → row-morsel | **1** | 12 | **157 MB** (12×) | **NO** |
| **24** | yes | 1 | 24 | **315 MB** (24×) | **NO** |
| 50 | yes | 4 | 13 | 170 MB (13×) | yes |
| 1500 (cross-K/V, *not a live consumer*) | yes | 94 | 16 | 53 MB (16× of 3.3 MB) | yes |

Two things fall out, both provable from the source:

1. **For `tq` in `[11, 2·workers)` the row-morsel path is the worst of both worlds:**
   `row_band = 1`, so the weight is re-streamed **once per token**, *and* M2col's
   `local_t + 2 <= rows` never fires, so the cvtph-halving tile is **inert**. At `tq=12` that is
   **157 MB of weight traffic instead of 13.1 MB** — a 12× blowup — for 0.157 GFLOP.
2. **The landed M2col win (`dot_f16c_2col`, 1.212×, default-on) is inert on its own real
   consumer** whenever `tq < 2·workers` (= 32 at `avail ≥ 16`). Combined with the audit's
   finding that its claimed beneficiary (cross-K/V) no longer uses this kernel at all, its
   effective payoff is much narrower than the ledger records.

**The lever is therefore not "weight-stationary token-blocking"** (the thing that was rejected —
and rightly so, at `tq=1500`). It is: **do not route fc2-class shapes to the row-morsel
scheduler.** The column-band path is *already* weight-stationary. The gate should compare the
re-streamed weight bytes (`bands · out · inp · 2`) against the compute, not `tq·out·inp` alone.
Concretely: require `row_band ≥ C` (so bands are fat and M2col actually fires), or gate on
arithmetic intensity `tq/workers`.

This is the same class of bug as `F32_2D_TALL_MAX_K` in `ft-kernel-cpu` (gate keyed on `k`
alone when the real invariant was B-bytes re-streamed per thread). Same fix shape.

## Expected magnitude — small, and stated honestly

Analytic bound (a **profile is blocked**, see below). turbo decoder has 4 layers; fc2 prefill
runs once per window. At `tq=12`, row-morsel moves `4 × 157 MB ≈ 628 MB` of weight per window
vs `4 × 13.1 MB ≈ 52 MB` for column-band. At ~50 GB/s that is ~12.6 ms vs ~1.0 ms, i.e.
**~11.6 ms/window saved**. Against ~2.9 s/window (turbo, track01) that is **≈ 0.4% e2e**.

Real but modest. It is a pure scheduling bug, **bit-exact**, and needs no transcript/WER gate —
only a perf number.

## BLOCKER — why no number is attached

`RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench -p franken_whisper
--bench f16_batch_prefill` **fails closed**, twice, with:

```
WARN rch::transfer: sync_to_remote: retryable error on attempt 1/3: timed out after 30000ms
[RCH] remote required; refusing local fallback (remote execution failed)
```

Root cause: **this working tree is 52 GB** (`target` 2.2 G, `legacy_whispercpp` 1.7 G, a 736 MB
`perf.data`, plus more). rsync cannot complete inside rch's 30 s `sync_to_remote` window. By
contrast `frankentorch` (11 GB, all deps in-tree) syncs and benches fine — every remote run this
session (`hz2`, `vmi1149989`, `vmi1227854`, `ovh-a`) was in that repo.

`RCH_REQUIRE_REMOTE=1` behaved exactly as designed: it refused the local fallback, and the disk
delta across both attempts was ~1 MB (my own output files). **No local build occurred.**
Without it, `rch` logs *"Remote execution failed …, running locally"* — the fall-open that eats
disk. Use it on every invocation.

## Unblock (any one)
- Raise rch's `sync_to_remote` timeout, or give it an rsync exclude list for
  `target/`, `legacy_whispercpp/`, `perf.data`, `*.wav`, model dirs.
- Shrink the working tree (owner call — I delete nothing).
- Run the bench on a host with disk headroom, locally, with the command in the header.

## Related
- `bd-f16-gemv-weight-stationary-reopen-ugyh` (the reopened lever)
- `bd-transcript-gate-unrunnable-xu9g` (umbrella measurement blocker)
- `docs/NEGATIVE_EVIDENCE.md`, 2026-07-10 "LEDGER-INTEGRITY AUDIT of all 10 do-not-retry families"
- `frankentorch 8e3e7c9d` — the `F32_2D_TALL_MAX_K` analogue (gate keyed on the wrong invariant)
