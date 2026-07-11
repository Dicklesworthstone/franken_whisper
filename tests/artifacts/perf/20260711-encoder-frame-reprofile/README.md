# Fresh default-turbo re-profile (2026-07-11) — SDPA flat-par REJECT (now ADMISSIBLE) + infra recipe

> **HONESTY CORRECTION (same day) — this was NOT a novel re-ranking. Read `project_turbo_e2e_frame_table`
> first.** That memory ALREADY documents (a) SDPA's ~37.8% encoder share, (b) the rayon/crossbeam
> decomposition, and (c) **"SDPA flat-vs-nested" as an existing REJECT (`59b77db`/`a410602`) that was
> "measurement-blocked"** pending "a QUIET ≥32-physical-core box." My net-new contribution is narrower
> and real: I **admissibly CONFIRMED that measurement-blocked reject** (0.997×, byte-exact, thinkstation1
> 64c satisfies the retry-condition via build-remote/run-local), and **verified the profiling-infra
> recipe** (§0). The "~37% rayon aggregate" in §2 is NOT a real increase over the frame table's 11.25% —
> it is a symbol-attribution artifact of my thin-LTO `release` build leaving `bridge_producer_consumer`
> un-inlined (the frame table's `release-perf` build inlines it into the loop bodies). Same engine, same
> reality; do not cite "37%" as evidence the engine changed. The FLAT #1 frame is still `dot_maddubs_i7`
> (i7 GEMM); "SDPA #1" holds only by PHASE/region, which was already known.

## (original writeup below — the SDPA-37% / rayon framing is superseded by the correction above)

# Fresh default-turbo re-profile (2026-07-11) — frame re-ranking + SDPA flat-par REJECT

Profile-before-optimizing pass. Built a debuginfo'd `e2e_probe` remotely, ran the DEFAULT
large-v3-turbo engine locally on `jfk_x8.wav` (88 s), sampled with `perf`. Three results:
a corrected owned-frame ranking, an admissible byte-exact REJECT of the first lever it pointed
at, and a verified profiling-infra recipe.

## 0. Profiling-infra recipe (VERIFIED — saved a future 8.5-min wasted build)
`rch` retrieves example binaries from `target/release/` and `target/debug/` but **NOT** from a
custom profile dir like `target/release-perf/`. A `--profile release-perf` remote build finishes
fine but the binary never syncs back (confirmed: "1 files, 2304 bytes" retrieved; `ssh` to workers
is denied, so no manual scp). And plain `--release` is `strip=true` → useless flames. Fix: force
debuginfo into the retrievable `release` profile via env overrides forwarded through the rch config
allowlist (`[environment] allowlist = ["CARGO_PROFILE_RELEASE_STRIP","CARGO_PROFILE_RELEASE_DEBUG",
"CARGO_PROFILE_RELEASE_LTO"]`):
```
CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=line-tables-only CARGO_PROFILE_RELEASE_LTO=thin \
RCH_CONFIG_DIR=$SCRATCH/rchcfg RCH_DISABLE_CONFIG_CACHE=1 RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
  rch exec -- cargo build --release -p franken_whisper --example e2e_probe
# -> target/release/examples/e2e_probe (9.3 MB, debuginfo, retrieved). Run:
FRANKEN_WHISPER_MODEL_DIR=legacy_whispercpp/whisper.cpp/models \
  perf record -F 1999 -o perf.data -- target/release/examples/e2e_probe large-v3-turbo jfk_x8.wav 1
```

## 1. Engine phase spans (FRANKEN_WHISPER_PERF_SPANS=1 — ground truth, excludes model load)
Per 30 s window (3 windows for jfk_x8): `encoder_window` ~1450 ms, `decode_loop` ~560 ms (58 tok),
`cross_kv` ~33 ms, `prefill` ~40 ms ⇒ **encoder ~72% of per-window compute, decode ~27%.**

Encoder sub-op breakdown (sum over 32 layers, one window; stable across windows):
```
  attn_sdpa   ~37%   <-- SINGLE BIGGEST encoder sub-op
  mlp_fc      ~20%   } int8 i7 maddubs GEMMs (nn.rs, ISA-ceilinged)
  mlp_proj    ~20%   }
  attn_out    ~12%   i8xi32 GEMM
  conv_stem  ~1.8-2.4% | ln/resids ~1.5-2.5% each | pos_emb ~0%
```
So **SDPA ≈ 0.37 × 0.72 ≈ 26% of e2e** — and SDPA is in **frankentorch (clean, non-conflicting).**

## 2. Flat self-time (perf, `sort symbol`; see perf_flat_top22.txt) — the RE-RANKING
```
16.30%  dot_maddubs_i7_m2n4                     (i7 int8 GEMM — MLP/attn linears)
11.57%  matrixmultiply::sgemm_kernel            (SDPA's Q@K/P@V + conv; NOT the int8 linears)
 3.63%  matrixmultiply::gemm::gemm_loop         (sgemm packing)
 2.46%  ft_kernel_cpu::sdpa_forward_f32{closure} (SDPA softmax/body)
~37% AGGREGATE: rayon bridge_producer_consumer (10.72+4.32+3.00+2.96+1.17+0.83 ≈ 23%)
              + crossbeam Stealer::steal (9.36%) + crossbeam_epoch try_advance (2.71%)
              + rayon wait_until_cold (1.91%)   = RAYON/CROSSBEAM MACHINERY
 2.42%  encoder::load_linear_transposed          (ONE-TIME model load; perf samples it)
```
**This re-ranks bd-o0bu** (which had SDPA 11.46%, rayon 11.25%). The real picture: **SDPA is #1**,
and **~37% of self-time is parallelism machinery** — both far bigger than believed. The int8 i7
GEMM (`dot_maddubs_i7`, 16.3%) is #2, at its VNNI hardware ceiling (Zen3, `#![forbid(unsafe_code)]`,
[[project_isa_baseline]]) and in the actively-edited `nn.rs` — unmineable this session. The f32
2-D tiler (tile_shape/gemm-swap lane) is confirmed absent — matmul_tensor_contiguous_f32 does not
appear; the default linears are int8 (last turn's gate-correction holds).

## 3. LEVER TESTED & REJECTED (byte-exact, admissible): SDPA nested→flat par split
Hypothesis: SDPA's few-heads-many-cores branch (`num_bh=20 < 32 threads`, `lib.rs:4692`) nests
`par_chunks_mut` inside `par_chunks_mut`; the flat profile's ~23% `bridge_producer_consumer`
looked like nested-iterator setup. Flattening to ONE `par_iter` over disjoint `split_at_mut` block
slices (safe under `#![deny(unsafe_code)]`) is BIT-EXACT (same blocks, same k-accumulation order).

Implemented behind `set_sdpa_flat_par` (default off) + a single-binary ABBA microbench at the exact
turbo encoder shape (20 heads, seq 1500, d 64, poly on, 32 threads, admissible on thinkstation1
64c). Harness preserved: `sdpa_flat_par_ab.rs.txt`. Result (min-of-25):
```
A nested  10.868 ms   B flat  10.901 ms   A/B = 0.997x   cv 3.5-4.1%   bit-exact: true
```
**REJECT — a wash (0.997×, within cv).** Rayon handles the nested split efficiently; the inner
bridge is not the cost. The ~37% machinery is therefore NOT the SDPA nesting — it is the aggregate
of every encoder par-dispatch plus the latency-bound DECODE per-token spin (the closed avenue,
[[project_decode_overthreaded_rayon_lead]]). The frankentorch flag+branch were **reverted** (no
measured benefit ⇒ no dead knob in a shared crate); the harness + numbers live here.

## 4. Frontier implication
The remaining owned e2e headroom is concentrated in SDPA's internal **sgemm** (Q@K/P@V via
matrixmultiply, ~15% of e2e) — i.e. the `gemm`-crate swap lane ([[project_gemm_backend_bd4hc0_corrected]]),
which is owner-gated (shared-crate dep + WER gate) and DOES reach the default path (unlike the f32
linear tiler). The int8 i7 GEMM is at its ISA ceiling. No byte-exact, non-owner-gated, unconflicted
lever with material e2e impact was found this pass; the accessible owned frames are at their ceiling.
