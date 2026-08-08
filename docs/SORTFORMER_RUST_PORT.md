# Native Streaming Sortformer Rust Port Contract

Status: authenticated native L1-L8 parity is complete for the pinned NVIDIA
recommended streaming profile, together with a production-shaped library
session and explicit evaluation-only CLI; broader L9 accuracy, L10 resource,
greater-than-four-speaker, and product-routing certification remain open
Fit verdict: **CONDITIONAL GO**
Promotion authority: explicit evaluation only; `Auto` and the transcribe
diarization resolver remain unchanged

## 1. Decision and no-claim boundary

The full end-to-end NVIDIA Streaming Sortformer v2.1 graph is the primary
learned-model candidate for a native Rust port. It directly owns speech
segmentation, overlap, speaker change, speaker activity, and speaker-count
behavior up to four speakers. A speaker encoder such as ReDimNet or ECAPA does
not replace those stages and remains a compact hybrid or low-resource ablation.

The existing native acoustic diarizer remains available as a deterministic
native-acoustic comparator, low-memory fallback candidate, and regression
control; it is not the learned-model graph oracle. The pinned NeMo graph plus a
source/runtime/exporter-bound public activation pack is now the L1-L8 external
graph oracle; the older final-output adapter remains only an identity-checked
discrete-output diagnostic oracle. No automatic routing change, broad accuracy
claim, or speed claim is authorized until the ordered parity and public
evaluation gates below pass. The explicit `sortformer-diarize` command reports
`evaluation_only`, defaults to a fully hash-verified release cache populated by
explicit `fw pull sortformer`, and does not affect the currently resolved
transcribe behavior. Explicit receipt/package paths remain available for
offline evaluation, but report no release-transport policy because hash
authentication alone cannot establish how caller-selected files arrived.
Unknown recordings report
`four_lane_capped_output_true_speaker_count_unknown`; the active lane count is
not treated as a certified true-speaker count.

The port remains a conditional fit rather than an unconditional fit because:

- the 117.7-million-parameter graph is a tier-2 port with a substantial
  FastConformer front end and stateful streaming overlay;
- the upstream model has a fixed four-speaker output capacity and cannot by
  itself establish that an unconstrained recording contains more speakers;
- the custom NVIDIA Open Model License requires an explicit distribution
  policy; and
- the accepted four-fixture parity pack and ten-row development diagnostic are
  not a sealed-corpus product decision gate.

No existing native diarization code should be deleted to make room for this
route. A failed parity, resource, capacity, or public-accuracy gate leaves the
current production behavior unchanged.

## 2. Primary upstream and runtime pins

| Item | Frozen identity |
|---|---|
| Model | `nvidia/diar_streaming_sortformer_4spk-v2.1` |
| Hugging Face revision | `fafaab5faa1617a0ca52d38dd3dc4bd636800d3d` |
| `.nemo` bytes | `471367680` |
| `.nemo` SHA-256 | `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8` |
| `model_config.yaml` SHA-256 | `2865d469c4d2aac54aa5b8a956b2423c053806dd20d5bf5d08675942a1acface` |
| `model_weights.ckpt` bytes | `471352898` |
| `model_weights.ckpt` SHA-256 | `eca9773c2dab91dd41fbaa4473cebb9d00811d67788ce2de609dadc6e499cdf4` |
| Pinned externally-derived 990-entry state inventory SHA-256 | `f4f219cf4ac6f755247b56d19e425db3d6a7c23c4509176549b363b63abdf532` |
| Canonical 992-record topology-projection SHA-256 | `2c32b0b9e48bb296e66615b038827d0fdde4b4fda2ce044a6c30cd317456c8d7` |
| Reviewed converter source SHA-256 | `3ce885d1dcb0aeeebf2bb73c165f501a1d240e01ad70354c65cf43d8a3c6d8ce` |
| Canonical conversion receipt SHA-256 | `407c642f3d51b399514f6a35227b1c80886387472a44fb78f01b824d26318fb0` |
| Converted package bytes | `491570584` |
| Converted package SHA-256 | `487fa30cb0aa9799c77bd9985e6787962c3991fab8d4d576a4f1221d45298f6a` |
| NeMo source revision | `40ace43c7cf151af78dc22027c02feeca7e06b6a` |
| Python | `3.12.12` |
| NeMo package | `3.1.0+40ace43c7c` |
| PyTorch / torchaudio | `2.7.1` / `2.7.1` |
| NumPy | `2.4.6` |
| safetensors | `0.8.0` |
| Historical conversion-oracle adapter SHA-256 | `8f376c979b7eaca41dc0a438d9aaa41c1c723052b97c45eb2acc59b6d6f00bde` |
| Current runtime-comparison adapter | `franken-whisper-sortformer-oracle-v3` |
| Current runtime-comparison adapter SHA-256 | `d8ced65ea4fa48e7f238005bf81659f57b9b575ddf6e04a75a835313ac0bf4eb` |
| External model-contract SHA-256 | `7ac048e3372fe4c622840beddfbeef42944d961408360324cb7276a69c8542c5` |
| Qualified runtime-fingerprint SHA-256 | `3713fd3f024c1cef7d860706baf0dbaaf18058c03c26331da6254687693d564c` |

### 2.1 Converted-weight admission evidence

The conversion receipt permanently binds the historical conversion-oracle
adapter digest above. Runtime comparison adapters are independently versioned
and hash-bound by the public comparison protocol. This separation keeps
immutable package provenance stable while making any runtime adapter change
produce a new comparison protocol identity.

On 2026-08-05, the frozen converter completed both an initial publication and
an exact-artifact retry on Darwin 25.2.0 arm64. Each exited zero with 974
tensors and the converter, topology-projection, receipt, and package identities
listed above; the resulting directory was mode `0700` and both files were mode
`0600`. The final focused Rust verifier run passed 31 tests with only the
deliberately operator-local test ignored. That exact ignored test was then run
locally against the admitted receipt/package pair and passed 1/1. The loader
rejects directories and symlinks before opening; on Linux, Android, and Apple
targets its post-precheck open is nonblocking, does not follow the final
symlink, rechecks the regular-file type, and binds the opened device/inode to
the precheck. This proves L0 converted-weight admission only; it is not a
model-forward or diarization accuracy result.

The archive configuration says `nemo_version: 2.6.0rc0`; that string is not
the executable reference. The intended oracle behavior is the pinned NeMo
source revision and package/runtime set above. It becomes accepted seam
authority only after `bd-y4ip.10` authenticates the executed source/runtime and
exporter. A future source or package change creates a new evidence row and
cannot silently inherit these tolerances.

The relevant installed source files at the pinned revision were independently
hashed:

| Source file | SHA-256 |
|---|---|
| `nemo/collections/asr/data/audio_to_diar_label.py` | `f9b0d23bd52da417ac18418ea1c83aa1119f59e6b37d3b2b3159c8cb2f036234` |
| `nemo/collections/asr/models/sortformer_diar_models.py` | `4978dba1a02b414893123f66905a1e523d5bb65766903269b325746c67f6920a` |
| `nemo/collections/asr/modules/sortformer_modules.py` | `3d136c245e3bf7a88c47fdd2eae1edb9189bbeddc3ff779cb5679a29d890b7eb` |
| `nemo/collections/asr/modules/conformer_encoder.py` | `a8b6f712cdf75a3be768848e8242ea9412ca7ff31ba2dda6b9602bcefc627cec` |
| `nemo/collections/asr/modules/transformer/transformer_encoders.py` | `a2859c86c8389f1954d5c8be04dc2bc422452517ef15e069cf42bfab5d304759` |
| `nemo/collections/asr/modules/transformer/transformer_modules.py` | `2564d95365cfafd486b1a3d10e2e2f438702907076f3716dd4c42d568b3bcc72` |
| `nemo/collections/asr/modules/audio_preprocessing.py` | `c061f521e14978d22ad57fa5ddf08f1103c2d1f1a4e01aca6698bfad007e8e7c` |
| `nemo/collections/asr/parts/preprocessing/features.py` | `4290ed2d697362a68a6158fb8b7b8d1e2306b223b83172c63fc6b5d31b28ee69` |
| `nemo/collections/asr/parts/preprocessing/segment.py` | `a598d91b94110e0c12a1ba4a57894ce89109e597fa8e909cf7b5b6e7bb9369af` |
| `nemo/collections/asr/parts/submodules/conformer_modules.py` | `99bb846c51db028d6d30b3d844af22826068aeaa0e48eb586489a31a9cbacf9d` |
| `nemo/collections/asr/parts/submodules/multi_head_attention.py` | `4999fd0d679fd7315ba275f7311fe6608c48e492bd337f2e220c99b8b9729c69` |
| `nemo/collections/asr/parts/submodules/subsampling.py` | `4fbc689f3f66e4630b286196315a02b315ad53e8049c164fe40dd11168cf0834` |
| `nemo/collections/asr/parts/submodules/causal_convs.py` | `7cf505c8caef44a37a7dec10b51eb2d60ec2f1efc3a2badc3c20c37e427cbd42` |
| `nemo/collections/asr/parts/utils/speaker_utils.py` | `6c247bdda26fd010190e1c96f8399f77a5265a180086e134d9b167b3c8019dc0` |
| `nemo/collections/asr/parts/utils/vad_utils.py` | `7beb57efff5e08407f9f16afe9c0da7d0e2ddb9bd62e2a37424693e48c5f0437` |
| `nemo/collections/asr/parts/mixins/diarization.py` | `5365e416ecab192cf59f1b9d6554ebce0ed3bdb2fee7575966ac1e3fca1a1408` |
| `nemo/collections/common/parts/transformer_utils.py` | `47f5e337230e7b4e176877f01c2ae85f75c024942dc567f27d8429c3e60e67c0` |

These hashes identify the source semantics inspected for the port. They do not
prove that arbitrary Python code or model bytes are trustworthy.

The receipt verifier pins Python, NeMo, PyTorch, torchaudio, NumPy,
safetensors, librosa 0.11.0, Lhotse 1.33.0, SoundFile 0.14.0, SciPy 1.18.0,
OmegaConf 2.3.0, Hydra 1.3.2, and Lightning 2.4.0. That list is not yet an
audited transitive closure. `bd-y4ip.10` must bind every package and executed
source that influences an accepted seam or eliminate it from seam authority by
exporting the exact decoded PCM plus exact intermediate tensors. A repeated
version document alone does not close that gap. Every accepted future oracle
row must also authenticate the adapter executable against the pinned digest
above before its self-reported identity is trusted, authenticate the installed
NeMo source bytes rather than only package metadata, and remove or contain its
executable/audio/model hash-then-open windows. The converter's model input is
now one owned verified byte stream, but the external final-output adapter has a
separate lifecycle. Those are `bd-y4ip.10`/`bd-y4ip.7` gates, not evidence
supplied by this document.

## 3. License and artifact-distribution boundary

The model is under the custom
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
not the repository's Rust source license. The model card and license were
reviewed for the dedicated converted-weight release. Each distributed package
is accompanied by an unmodified snapshot of the NVIDIA Open Model License, the
required NVIDIA notice, and the identity-bound conversion receipt. A different
upstream revision or conversion recipe requires a new review and release
identity.

The reference snapshot retrieved on 2026-08-06 had raw-payload SHA-256
`13c9c998e24abd5211cff4b5c912902f566bd710294da98580be7b3376626f04`,
HTTP `Last-Modified: Mon, 03 Aug 2026 17:46:28 GMT`, and ETag
`4b001-658281e31650b`. The pinned NeMo source tree is Apache-2.0; its `LICENSE`
payload has SHA-256
`43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1`.
The pinned `nemo/collections/asr/parts/preprocessing/features.py` also contains a Ryan Leary MIT
notice, which must be preserved in any distribution that carries derived
source. These identities establish an auditable review input, not legal
approval.

The distribution policy is therefore:

- no original or converted weights are committed to Git, embedded in the
  binary, or copied into test fixtures;
- the exact converted safetensors package is distributed only by the dedicated
  `sortformer-v2.1-f32-v1` GitHub model release, never by a source or binary
  archive;
- that release contains exactly the weights, the canonical conversion receipt,
  the NVIDIA Open Model License snapshot, and the required notice;
- `fw pull sortformer` downloads only those compiled manifest URLs, verifies
  every size and SHA-256, and admits the four files together into an
  identity-versioned per-user cache;
- explicit caller-supplied receipt/package paths remain an offline evaluation
  route, but do not inherit release-transport provenance; and
- runtime loading never deserializes the pickle-based `.ckpt` from the `.nemo`
  archive.

This is a repository policy, not legal advice and not a claim that every future
form of redistribution is permitted.

### 3.1 Native proof snapshot (2026-08-08)

The safe Rust f32 graph now loads the authenticated 491,570,584-byte package,
runs the frozen streaming schedule through all 17 FastConformer and 18
Transformer blocks, updates the speaker cache/FIFO state, and emits L7 activity
and L8 anonymous speaker turns. `SortformerSession` checkpoints cancellation at
chunk and neural-block boundaries, uses checked bounded allocations, and clamps
user-visible tail turns to the physical recording duration. The standalone
`sortformer-diarize` command exercises that same library path after the existing
audio normalizer; it does not participate in `DiarizationEngine::Auto`.

The current local Apple M4 Pro evidence snapshot is:

| Evidence | Result |
|---|---:|
| Public fixture | VoxConverse `mevkw`, samples 0..400000, 25.0 s, overlap, two reference speakers |
| L5 production probability drift | max abs `6.854534149e-7`; relative L2 `9.072629352e-8` |
| Reconstructable-prefix L7/L8 comparison | byte-exact against authenticated public outputs before physical-duration clamping |
| Authoritative transcript-free DER / JER | `0.058904110` / `0.065850064` |
| DER components | miss `0.92 s`; false alarm `0.80 s`; confusion `0.00 s` |
| Native inference / audio / RTF | `31.749242 s` / `25.0 s` / `1.269970` |
| Session materialization after package admission | `3.001861 s` |
| Pre-optimization real CLI public input | VoxConverse `syiwe`, `69.146125 s`, three inferred active lanes |
| Pre-optimization CLI inference / RTF | `108.945964 s` / `1.575590` |
| Pre-optimization CLI session materialization after package admission | `3.112265 s` |
| Optimized release CLI inference / RTF | `2.960639 s` / `0.0428171` |
| Release package admission / session materialization / combined model load | `1.904567 s` / `0.428503 s` / `2.333070 s` |
| Release end-to-end wall time | `6.13 s` |
| Release peak RSS | `1,368,850,432` bytes |
| Recommended-profile public parity pack | Four complete/declared public fixtures, including 102-second `mevkw` and one four-speaker row; 4,540 authenticated L1-L8 tensors |
| Recommended-profile whole-recording L5 drift | `mevkw`, 102.0 s: max abs `1.072883606e-6`; relative L2 `8.214150678e-8` |
| Recommended-profile whole-recording L7/L8 | Native activity and all 16 anonymous turns byte-exact against the authenticated NeMo source output |
| Strict default-score DER / JER | `0.029816514` / `0.038068192` with zero boundary collar and overlap included |
| Strict default-score components | miss `2.32 s`; false alarm `1.28 s`; confusion `0.04 s` |
| Contended local release parity runtime | inference `16.401269 s` / audio `102.0 s` / RTF `0.160797`; retained only as parity-run timing, not the throughput baseline |
| Historical archive-profile comparison | VoxConverse `mevkw`, 102.0 s: native DER/JER `0.021214713430` / `0.029991623791`; recommended-profile NeMo DER/JER `0.019846022241` / `0.029477961362` under the protocol scorer |
| Whole-recording native resource row | wall `14.940 s`; RTF `0.146470588235`; approximate whole-tree sampled RSS `1,363,558,400` bytes |
| Whole-recording NeMo resource row | wall `18.482 s`; RTF `0.181196078431`; approximate whole-tree sampled RSS `2,209,021,952` bytes; frozen 2 GiB cap failed |
| Historical archive-profile L8 discrepancy | Four one-frame (80 ms) boundary differences among 16 anonymous turns; resolved as a streaming-profile mismatch, not a Rust/source parity loss |

These are real local runtime and accuracy observations. The optimized release
row is about 23.4 times faster than real time on this one Apple M4 Pro input;
it is not broad accuracy certification or a same-invocation speedup claim over
an incumbent. Peak RSS remains about 1.27 GiB. The production forward no longer
retains parity-only subsampler/block traces or computes FastConformer Q/K/V twice, but
the before/after timings above were separate invocations and therefore are not
a ledger-qualified optimization win. A frozen multi-record public evaluation,
same-invocation baseline comparison, 90-minute resource row, and further
profile-guided optimization remain required.

The historical 102-second archive-profile row remains a valid loss report, not
a tolerance change. That invocation compared the converted archive defaults
(`188/1/1/0/188/188`) against NVIDIA's recommended runtime profile
(`340/1/40/40/300/188`). The lane identities matched, but boundaries differed
at native/reference
`13840/13760` ms (start), `64800/64880` ms (end), `74480/74400` ms (end), and
`101840/101760` ms (end). The accepted native profile now matches the published
recommended geometry, and the complete recommended-profile source pack proves
all 16 turns byte-exact without changing a numeric or discrete gate. The strict
default-score row above and the protocol rows below intentionally use different
scorer policies: the former has zero collar and includes overlap; protocol v7
uses a 250 ms reference-boundary collar and excludes overlap.

The balanced ten-record development comparison was reproduced by protocol v7
on 2026-08-08 with the exact release executable
`e7f3991525ca5b9aa4b8535a4ebf4d2ec92b5fa618f6af362480d132c2d43f4c`.
The protocol SHA-256 is
`f6d62452f4291c453f861a5613f33bd2e935945825a7fd4f4e559f47048874d8`,
the result SHA-256 is
`e4fc7ca6b461cc0e0648631822b58e0246d2319f749cf8f7d5936a3a159758b5`,
and the deterministic-accuracy SHA-256 is
`84ff4a04394537595d6f2038a6e5a70c6b2afe90fe710c75bbb26b870997884b`.
The path-free bundle and evidence file SHA-256 values are respectively
`ebef44364e844f9eb3432758793814f8eda22185fa503050879390fa8cbbdc13`
and `59d94e06df362a4bea7049ee4780638db637a0c00210519469420a992b81bcb6`.
The exact adapter was `franken-whisper-sortformer-oracle-v3` with executable
SHA-256
`d8ced65ea4fa48e7f238005bf81659f57b9b575ddf6e04a75a835313ac0bf4eb`.
The ten Williams rows completed in balanced order. Native acoustic, ECAPA, and
fused ECAPA completed 10/10 attempts; native and external Sortformer each
completed every one of the eight four-speaker-capacity-eligible attempts and
retained two typed capacity skips, with zero failed attempts in every lane.
The common-complete intersection therefore contains eight recordings.

On those eight available Sortformer recordings, native and external
Sortformer respectively recorded micro DER `0.014704749721` and
`0.014435836015`, macro JER `0.015993908493` and `0.015972975758`, overlap F1
`0.859849181534` and `0.861681374826`, and full-timeline exact speaker-count
rate `1.0` for both. Native reported an exact inferred count on 8/8; the
external diagnostic does not expose an independently resolved count estimate.
Their aggregate RTF values were `0.058895178751` and `0.056356249758`.
Approximate sampled whole-process-tree peak RSS was 1,427,947,520 bytes for
native and 2,422,849,536 bytes for external NeMo: native passed the frozen
2 GiB cap and external NeMo honestly failed it. Every lane passed the 500 ms
cancellation cap with a 27 ms retained maximum. ECAPA-only and fused ECAPA
also completed 10/10 after the turn builder stopped extending a secondary
overlap across unmarked speech and coupled its end to an adjacent primary's
analysis-frame midpoint clipping.

The artifact pair passed independent structural and bundle-identity
verification, including rejection of a self-consistently rehashed bundle
substitution. It is mode-0600, aggregate-only, path-free, diagnostic-only,
development-uncertified evidence with `production_route_changed=false` and no
superiority permission. It does not certify deployment accuracy, solve the
greater-than-four-speaker capacity boundary, excuse the external RSS loss, or
authorize `Auto` routing.

The historical one-record resource row was reproduced by protocol v6 in five fresh workers,
and all five lanes completed, so its common-complete intersection contains one
recording. The comparison executable SHA-256 is
`2816def9153aeed644b86aa8c480a046a8b18a4a3414fe4bc73926988142ee0d`,
the protocol SHA-256 is
`af046e2f7060590d6d94421f404040a75a006ddcaaef37e79bf92e888a1cd04b`,
the aggregate result SHA-256 is
`418a5a6337851ae6ff6cffd2f485a71d04be40faebc2862ca7f19fcf8e07452b`,
the bundle-file SHA-256 is
`0860d2b5112b6c01813a6b7ecaec84a73e3e4ca26317cd27a7bd4798bcf05bf9`,
and the evidence-file SHA-256 is
`655953efe564a9ae5697e8017876c961e96c5b1f8deb1b0aaad5f131aca409ac`.
The real artifact pair validated and the retained tamper regression rejected a
self-consistently rehashed bundle-identity substitution. Every lane retained a
37 ms cancellation probe through the same bounded observer path used by real
attempts. Native acoustic, ECAPA, fused ECAPA, native Sortformer, and external
Sortformer respectively recorded RTF `0.009794`, `0.023725`, `0.023804`,
`0.080059`, and `0.125304`, with sampled whole-process-tree RSS of 22,544,384,
249,888,768, 251,822,080, 1,359,282,176, and 2,213,494,784 bytes. The external
lane honestly failed the frozen 2 GiB cap; the native lane passed it. Native and
external Sortformer DER were `0.021214713430` and `0.019846022241` on this row.
RSS remains an approximate sampled process-group sum whose sample starts are at
least 50 ms apart, not an exact high-water mark. Protocol v6 additionally checks
cancellation and the attempt deadline inside platform scans, rejects silent
omission of matched live Linux group members, and treats a complete zero-only
scan as missing rather than measured zero. This v6 row supersedes the v5
resource/cancellation evidence. It is still only one development observation,
not a complete ten-row Williams schedule, a multi-condition accuracy gate, or
production-route authorization.

The exact-score L6 compression test and complete recommended-profile chained
run pass inside the frozen numeric and discrete envelopes. The earlier `iqtde`
archive-profile investigation correctly exposed a weak-cache K=66 cutoff tie:
libc++ `nth_element` selected frame 27 where a deterministic full sort selected
a different bit-equal identity. The accepted Rust path now uses a safe,
index-based translation of the pinned LLVM libc++ 15.0.7 `nth_element`
algorithm and retains a fail-closed geometry test. No tolerance, fixture, or
gate was changed. This closes same-host L6 identity for the accepted profile;
other standard-library/runtime/CPU tiers still require their own evidence.

### 3.2 Initial hardware and memory tier

The f32 reference target is the current 64-bit little-endian
`aarch64-apple-darwin` development host: Apple M4 Pro, 64 GiB physical memory,
and Rust `1.99.0-nightly (9f36de775 2026-07-19)`. The reference code must not
require Apple-only instructions, but passing on this host authorizes no x86-64,
other ARM, operating-system, or lower-memory support claim.

This is a desktop/server memory-tier port, not an embedded or low-memory route.
The immutable f32 tensor payload is about 468.7 MiB after adding the positional
buffer; the safetensors container, decoded tensors, activations, streaming
state, and allocator overhead make that number an invalid RSS estimate. The
first child-only debug CLI measurement reached 1,378,844,672 bytes on a 69.15 s
public recording. That single row does not certify a 90-minute ceiling, so
evaluation continues to require at least 4 GiB of available memory and treats
that floor as provisional. Product routing and any low-memory fallback choice
remain owned by the later integration gate.

## 4. Model and state census

Pinned live-model introspection produced:

| Census | Value |
|---|---:|
| Trainable parameters | 117,693,960 |
| Parameter tensors | 937 |
| State tensors | 990 |
| State elements | 117,744,681 |
| Float32 state tensors | 973 |
| Int64 state tensors | 17 |
| Float32 state elements | 117,744,664 |
| Float32 state bytes | 470,978,656 |
| Total state payload bytes | 470,978,792 |
| Conformer layers | 17 |
| Transformer blocks | 18 |
| Linear modules | 266 |
| Layer-normalization modules | 121 |
| Dropout modules | 124 |
| Conv1d modules | 34 |
| Conv2d modules | 5 |
| BatchNorm1d modules | 17 |

Approximate element ownership from the same introspection is 109,565,969
encoder elements, 8,007,552 transformer elements, 137,864 Sortformer-module
elements, and 33,296 preprocessor elements. The f32 weights alone are about
449 MiB. Peak process RSS and activation high-water marks are not inferred
from that number; they remain measurement gates.

There are two named non-persistent buffers outside `state_dict`:
`preprocessor.dtype_sentinel_tensor`, an empty f32 tensor, and
`encoder.pos_enc.pe` with shape `[1, 9999, 512]`: 5,119,488 f32 values, about
19.5 MiB. The converted package must export the exact positional values or
bind and test a deterministic Rust regeneration algorithm; silently omitting
them from the audit is not acceptable.

All 17 int64 state tensors are
`encoder.layers.{0..16}.conv.batch_norm.num_batches_tracked`. They are
training-only counters. The initial package census is therefore 992 audited
source records: export the 973 f32 state tensors plus the positional buffer as
974 destination tensors, and explicitly receipt the 17 counters plus the empty
dtype sentinel as 18 drops. Before the safetensors header, the expected
destination tensor payload is 122,864,152 f32 elements or 491,456,608 bytes.

The conversion receipt must enumerate every tensor and record, at minimum:

- source key and destination key;
- source and destination dtype;
- source and destination shape and logical layout;
- the typed conversion transform;
- source-value and destination-value SHA-256;
- byte length and element count; and
- the pinned model, source, config, converter, and receipt-schema identities.

The loader must reject missing tensors, unexpected tensors, duplicate names,
duplicate or unknown tensor-entry fields, wrong shapes, unsupported dtypes,
non-finite materialization, checksum drift, and unknown package-schema
versions. A matching total byte count is not sufficient.

Version 1 permits only a byte-preserving
`identity_contiguous_f32` export: the destination name, shape, layout, value
hash, element count, and byte count must equal the f32 source record. The 17
training-only i64 counters and the empty runtime dtype sentinel are the only
legal drops. Transpose, reshape, split, concatenate, cast, quantization, and
prepacking require a new receipt schema and new parity evidence.

The trust chain is deliberately one-way. The Rust binary compiles the reviewed
converter-source, canonical topology-projection, receipt, package-length, and
package SHA-256 roots. Its public loader accepts only receipt and package paths;
there is no caller-supplied digest parameter. The topology digest is recomputed
from all structural receipt fields, so a renamed or reshaped f32 tensor cannot
hide behind matching aggregate counts. The authenticated receipt binds every
source and destination value hash. The loader independently proves each
exported destination's exact name, shape, dtype, raw payload, finite values,
compact lexicographic layout, metadata absence, and complete census; it does
not re-derive dropped source hashes from the executable checkpoint.

The frozen offline profile in `scripts/convert_to_safetensors.py` produced this
package from one owned, hash-verified copy of the exact `.nemo` byte stream. It
verified both archive members, installed `direct_url.json` revision metadata,
17 selected source files before and after export, the listed package-version
tuple, the original insertion-order state inventory, every tensor's contiguous
CPU representation, the frozen topology-projection and package identities, and
all aggregate censuses.
It also binds and rechecks its own source inode and digest, instantiates the
pinned graph without `restore_from` temporary extraction, publishes owner-only
mode-0600 outputs, and permits retries only by reusing exact artifacts. The
licensed model bytes and receipt remain outside Git. The exact converted
package and canonical receipt are now redistributed in the dedicated,
hash-pinned `sortformer-v2.1-f32-v1` GitHub release beside the NVIDIA Open
Model License agreement and required notice; `fw pull sortformer` verifies the
separate distribution manifest before cache publication.

## 5. Frozen forward graph

### 5.1 Input and frontend

- mono 16 kHz finite PCM bounded to `[-1, 1]`;
- signed 16-bit WAV decoding maps each sample to `sample / 32768.0`;
- pre-emphasis preserves the first sample and then uses
  `x[t] - 0.97 * x[t - 1]`;
- constant center padding for STFT;
- 400-sample non-periodic Hann window centered in a 512-point FFT, 160-sample
  hop, 257 one-sided bins, and squared magnitude;
- 128 log-mel features, frame splicing 1, no feature normalization;
- the exact stored Slaney mel buffer with shape `[1, 128, 257]`;
- natural logarithm after adding `2^-24`;
- archive dither value `1e-5`, but evaluation mode disables dither; `pad_to`
  remains 16, masks frames beyond the declared valid length to zero, and the
  canonical L1 comparison crops back to valid frames; and
- batch size one, CPU float32, no autocast, no quantization.

The converted package must retain `preprocessor.featurizer.window` with shape
`[400]` and raw little-endian f32 SHA-256
`7d6b2ab4944b0b65650e1bba1132821fd1d2ed000df84dbd893316788d0ef062`, plus
`preprocessor.featurizer.fb` with shape `[1, 128, 257]` and SHA-256
`82663f1145f6965d8b27a85f32a44fa4f3bffef9bd0d6c2d1902b334a012367b`.
Rust loads and verifies these stored buffers rather than regenerating them.

The first oracle seam is the exact framed log-mel tensor and its valid length.
The effective valid-frame count is `floor(samples / 160)`: centered STFT
creates one extra frame, but NeMo marks it invalid and the model crops to the
declared length. The existing Whisper frontend is not reusable as-is because
its window periodicity, padding, transform length, logarithm, clamping, and
normalization semantics differ. Every frontend fact must still be captured by
numeric fixtures rather than accepted from source inspection alone.

The initial Rust L1 implementation is deliberately whole-file and bounded to
115,200,000 samples (two hours at 16 kHz), which includes the 90-minute target
class. It materializes the complete `[128, T]` log-mel tensor, although
pre-emphasis is computed on demand rather than stored in a second waveform
buffer. L10 must measure peak RSS on a 90-minute public or synthetic input.
True incremental pre-emphasis, STFT overlap continuity, and concatenation parity
are separate seams; this bound certifies no live-streaming frontend.

### 5.2 FastConformer encoder

- depthwise-striding subsampling by 8 with 256 convolution channels;
- 17 layers, model width 512, eight attention heads;
- feed-forward expansion factor 4;
- relative-position attention with x-scaling and untied biases;
- convolution kernel 9 with batch normalization; and
- evaluation-mode dropout and stochastic-depth behavior.

Subsampling is three symmetric stride-2 stages: a `1 -> 256` 3-by-3 Conv2d,
then two grouped depthwise 3-by-3 stages with pointwise `256 -> 256`
convolutions and ReLU after each stage. The result flattens `256 * 16 = 4096`
coordinates before the `4096 -> 512` affine.

Each Conformer block has two half-residual feed-forward modules,
Transformer-XL relative attention, and a Swish convolution module. Despite its
class name, the kernel-9 `CausalConv1D` receives integer padding 4 and is
symmetric four-left/four-right in this graph. Relative attention uses learned
`pos_bias_u` and `pos_bias_v`, a bias-free positional affine, relative shift,
and division by `sqrt(64)`. The pinned graph does not use PyTorch SDPA.

The archive encoder uses regular attention with `att_context_size = [-1, -1]`.
Evaluation never enters the training-only causal-context mutation, so
FastConformer's `_create_masks` returns no attention mask. It also returns an
inverted length mask (`true` means padding), but the relative-position
self-attention branch does not consume that pad mask; the convolution module
does. Attention is therefore entirely unmasked and bidirectional within the
concatenated cache, FIFO, and chunk sequence. The accepted batch-one
synchronous path normally has physical length equal to declared length, so it
does not introduce a padded tail at that seam.

The port must reproduce the NeMo ordering of feed-forward, attention,
convolution, residual, scaling, and normalization operations. Existing Whisper
attention helpers are not accepted as evidence of equivalent conventions.
FastConformer scales its input by exactly `sqrt(512)`. Every PyTorch LayerNorm
uses epsilon `1e-5`. Each convolution BatchNorm1d uses stored evaluation running
statistics and affine parameters with epsilon `1e-5`; training momentum is not
an inference input. Dropout is identity in evaluation, and stochastic depth is
disabled (`stochastic_depth_drop_prob = 0`).

### 5.3 Projection, transformer, and speaker head

- learned projection from 512 FastConformer coordinates to width 192;
- 18 transformer blocks at width 192, inner width 768, eight heads;
- ReLU feed-forward activation;
- post-layer-normalization inside each block, with no final transformer layer
  normalization operation; and
- four sigmoid speaker-activity outputs.

Transformer attention uses eight 24-dimensional heads, divides both Q and K
by `sqrt(sqrt(24))`, and adds `-10000` for padded positions rather than
negative infinity. The inference head is exactly ReLU, evaluation-identity
dropout, `192 -> 192`, ReLU, evaluation-identity dropout, `192 -> 4`, then
sigmoid. The checkpoint's `384 -> 4` `hidden_to_spks` tensor is dead in the
accepted inference path. It remains in the complete conversion receipt but
must not be wired into the Rust graph by name intuition.

`mask_future` is absent and therefore false, so the transformer's diagonal
future mask is absent. `form_attention_mask` returns valid key positions as
`[batch, 1, 1, length]`, with zero for valid keys and `-10000` for padded keys,
then broadcasts them across heads and queries when adding to attention logits;
it adds no causal triangle and no separate padded-query mask. The archive
sets `pre_ln = false`, which is why the otherwise named
`pre_ln_final_layer_norm` path does not instantiate a final layer norm.

### 5.4 Streaming state machine

The archive sets `streaming_mode = true`; `async_streaming` is absent, and the
source defaults it to false. The accepted path is therefore synchronous
stateful chunk emulation. It reads and preprocesses the entire waveform first,
skips global peak-amplitude normalization because streaming mode is set, and
crops features to the processed signal length. It is not evidence of a true
incremental-waveform frontend; live audio continuity requires separate frontend
and state seams.

The accepted native path and activation truth pack use NVIDIA's published
recommended synchronous-streaming profile, which is also the profile pinned by
the external comparison adapter. The raw `.nemo` archive carries lower-context
construction defaults (`188/1/1/0/188`); those defaults are not the documented
deployment profile and are no longer treated as the runtime oracle. Keeping the
two configurations separate is essential: an earlier native run used the
archive defaults while the external adapter used the recommended profile and
therefore produced four apparent 80 ms boundary differences that were not a
numeric or top-k parity failure.

| Quantity | Output-frame count |
|---|---:|
| Chunk | 340 |
| Left context | 1 |
| Right context | 40 |
| FIFO capacity | 40 |
| Configured cache update | 300 |
| Speaker cache | 188 |
| Silence cache frames per speaker | 3 |

For a chunk beginning at feature index `start`, the feature loader uses
`left = min(8, start)`, `end = min(start + 340*8, feature_length)`, and
`right = min(320, feature_length - end)`, then transposes and pre-encodes.
After factor-8 subsampling it uses `round(left/8)` left frames and
`ceil(right/8)` right frames. `drop_extra_pre_encoded` is zero.

The nominal first input buffer is 30.40 seconds (340 central frames plus 40
right-context frames at the 80 ms output stride), and the output stride is 80
ms. FIFO movement is:

```text
min(max(configured_update, chunk_len - fifo_capacity + current_fifo),
    current_fifo + chunk_len)
```

The first full chunk moves 300 frames into cache construction and retains 40 in
FIFO. A steady full chunk combines 40 FIFO frames with 340 current frames,
moves 340, and again retains 40. A short final chunk may move fewer than the
configured 300 frames because the pop is capped by the physically available
FIFO-plus-chunk total; Rust performs the capacity subtraction with checked
arithmetic so a tail shorter than 40 frames cannot underflow. An interior
feature chunk contains 3,048 pre-encoding frames and subsamples to 381 encoder
frames (one left-context, 340 central, 40 right-context); with a full cache the
recurrent encoder sequence can reach 569 frames. First-full, steady, large-tail,
and sub-40-frame-tail behavior are explicit regressions rather than
extrapolations.

Initial speaker-cache and FIFO embeddings have shape `[batch, 0, 512]`.
`spkcache_preds` and `fifo_preds` are initially `None`, as are
`spkcache_lengths`, `fifo_lengths`, and `spk_perm`; these Options must not be
substituted with empty tensors without a seam proving equivalence. `fifo_preds`
and `spkcache_preds` both become tensors at the first update because the
300-frame first pop already exceeds the 188-frame speaker cache. Mean silence is a zero
`[batch, 512]` tensor and the silence count is zero-valued i64 `[batch]`.
Lengths otherwise derive from physical shapes. Rust may own checked scalar
lengths as an implementation invariant, but parity must not invent oracle
length tensors.

Cache compression has 44 non-silence slots per speaker
(`188/4 - 3`). Its strong, weak, and minimum-positive counts are respectively
`floor(44*0.75) = 33`, `floor(44*1.5) = 66`, and
`floor(44*0.5) = 22`. A frame is silent when `sum_s p_s < 0.2`; the running
silence embedding is
`(old_mean*n_old + sum(new_silent_embeddings)) / (n_old + new_count)`, with the
denominator clamped to at least one.

For speaker lane `j`, the base score is:

```text
ln(max(p_j, 0.25)) - ln(max(1 - p_j, 0.25))
  + sum_k ln(max(1 - p_k, 0.25)) - ln(0.5)
```

`p_j <= 0.5` forces negative infinity. Once a lane has at least 22 positive
scores, its active non-positive scores also become negative infinity. There is
no evaluation-time noise or speaker permutation. Frame rows at indices 188 and
later receive `+0.05`. Per lane, `torch.topk(sorted=false)` first selects 33
rows for a `-2*ln(0.5)` boost, then selects 66 rows from the already boosted
scores for a further `-ln(0.5)` boost. Three positive-infinity silence
placeholder rows are appended per speaker.

The boosted matrix is flattened in speaker-major order and a global
`topk(188, sorted=false)` is taken. Winners at negative infinity receive the
sentinel index 99,999; indices are then numerically sorted, reduced modulo the
augmented score-row count (the physical rows plus three placeholders), and any
remainder at or beyond the original physical row count is disabled together
with the final three silence rows. Gathering uses row zero for disabled entries,
then replaces their embeddings with mean silence and predictions with zero. The resulting cache is
grouped by speaker and chronological only within each lane, not globally; one
physical frame may be selected once for each speaker lane. Exact top-k ties
remain an oracle-fixture question.

The output mask order is also fixed. FastConformer returns embeddings and
lengths; `encoder_mask = arange < length` supplies transformer key validity;
the sigmoid head runs; predictions are multiplied by
`encoder_mask.unsqueeze(-1)` to zero padded query rows; and
`apply_mask_to_preds` applies the FastConformer lengths again before central
chunk slicing and cache/FIFO mutation. A final crop to `ceil(signal_length/8)`
exists only in the distributed cross-rank padding branch. The accepted
batch-one, non-distributed path does not take that branch because whole-file
features were already cropped.

The NeMo state type also includes an optional speaker-permutation field.
Speaker permutation is training-only in the pinned path. Evaluation passes
`permute_spk = false`, so the accepted inference state has no active
`spk_perm`. The Rust evaluation state may omit non-evaluation support but must
prove that the oracle field stays absent; later training or adaptation support
would be a different contract.

### 5.5 Post-processing and capacity

- at most four contiguous arrival-ordered labels `speaker_0` through
  `speaker_3`;
- each 80 ms probability is repeated eight times onto a 10 ms grid;
- onset and offset are 0.5; onset uses strict `>` and offset strict `<`, so an
  exact equality preserves the current state;
- onset/offset padding and minimum on/off durations are zero;
- an activity lane still open at the end uses the final 10 ms sample index as
  its endpoint, creating a one-frame-short source convention before adapter
  clipping and validation;
- the adapter relabels non-empty lanes by first onset and clips a terminal end
  that exceeds document duration by at most the accepted 79 ms tolerance;
- accepted turns are aligned to 80 ms, except the final end may equal document
  duration;
- overlap is represented by concurrent labeled turns, not an overlap flag; and
- speech, overlap, and speaker-change stages are derived exactly from the final
  turns.

A caller-provided pre-inference requirement above four speakers is
capacity-ineligible for this model. Before any product integration, such a
request must remain on the currently resolved route; the future integration
layer may choose a non-Sortformer path only after that path passes a separate
greater-than-four-speaker accuracy, capacity, privacy, and resource gate;
otherwise it preserves the current route or returns typed capacity-ineligible.
It must never clamp, drop, or merge speakers. A reference annotation loaded or
discovered for scoring only marks that completed row capacity-ineligible and
must never retroactively select a route.
The model itself cannot determine that an otherwise unconstrained recording
actually contains five or more speakers: it can only activate zero through
four lanes. Unknown-capacity production inputs therefore require a separate
capacity sentinel or a candid capped-output status before this route can be
certified.

Known timestamp intervals do not change the neural forward pass during parity
work. A later product adapter may use them as hard or soft constraints when
mapping the four opaque arrival-ordered streams to caller-provided speaker
references. Contradictory hard intervals must fail closed; they must not tune
or mutate model weights during a call. Soft intervals cannot override hard
evidence, and the output contract must distinguish an anonymous lane from a
lane identified through accepted reference evidence.

## 6. Operator inventory and FrankenTorch boundary

The port uses one Sortformer-specific CPU facade over FrankenTorch. It may
reuse general kernels only when the exact shape, layout, dtype, reduction, and
mask contract is proven at a seam.

| Operator family | Initial route | Required proof |
|---|---|---|
| Dense affine / batched GEMM | explicit CPU-only FrankenTorch matmul | scalar or oracle tensor differential |
| Conv2d and depthwise Conv2d | FrankenTorch kernels where layouts match | impulse, padding, tail, and random tensor cases |
| Conv1d / depthwise Conv1d | safe reference lowering first | exact padding, groups, dilation, and tail cases |
| Batch normalization | evaluation-mode FrankenTorch apply | epsilon, running-stat, affine, and layout parity |
| Layer normalization | FrankenTorch layer norm | axis, epsilon, and accumulation parity |
| Relative-position attention | safe model-specific composition first | Q/K/V, position bias, mask, softmax, and output seams |
| Transformer attention | safe model-specific composition first | per-block Q/K/V and output seams |
| ReLU / Swish / sigmoid | safe reference, then general kernel | special values and full-tensor drift |
| Softmax and masks | FrankenTorch only after mask equivalence | fully masked, tail, and long-context cases |
| Top-k, stable sort, gather, optional permutation | safe Rust model logic | tie, placeholder, disabled, and eval-absence fixtures |
| Cache/FIFO mutation | safe Rust bounded state machine | empty, first-full, steady, tail, and cancellation fixtures |
| Log-mel frontend | pinned-buffer Rust implementation | authenticated synthetic parity plus public source seams; native public-stage parity remains required |

No new tensor runtime is justified. No model-specific fused kernel is justified
before the f32 whole-forward gate passes and profiling identifies a measured
hot path. Quantization and fusion belong to `bd-y4ip.12`, not the reference
implementation.

## 7. Oracle floor pilot

The final-output external adapter still does not authorize intermediate
activations. A separate frozen exporter provides both the synthetic L1 pack
and the licensed-public L1-L8 source pack. Its source SHA-256 is
`b3020f1e6c136343adecabc3209f3b1ef70f40a7e36d2b2ed9b25fbbd439b6dd`;
the canonical receipt SHA-256 is
`ac3dab6f7ad48ccaeeee0ba8e1f4932b5377736e28fa471fa7d43020922df2a9`;
and the 282,716-byte safetensors package SHA-256 is
`294edcc0a9d80fa9470c2cd45f2c1556a47a56b7c98ba444984f764a1f398a8b`.
The package and receipt stay operator-local. Git contains only the exporter,
compiled trust roots, verifier, aggregate metrics, and documentation.

The truth pack covers deterministic non-human silence, impulse, exact integer
tone-cycle, and partial-tail fixtures. Each captures decoded PCM, input and
valid lengths, pre-emphasis, windowed frames, complex STFT, power, mel energy,
unmasked physical log-mel, valid log-mel, and the padded/masked frontend output,
plus the exact Hann and mel buffers. The exporter requested five runs in each
configured one-thread and eight-thread regime, producing 44 fixture-stage
observations and 45 all-pairs comparisons per observation. Every captured
source replay is byte-exact, so the measured synthetic source floor and its
tolerance are both zero. The receipt binds the requested regimes; it does not
claim direct observation of operating-system worker utilization.

That zero source floor is not silently reused as a cross-implementation Rust
tolerance. The independent radix-2 Rust FFT has a different valid floating-
point operation order from PyTorch. The diagnostic comparator therefore keeps
the source floor separate and predeclares binary cross-kernel ceilings of
`2^-12` maximum absolute drift, `2^-17` mean absolute drift, and `2^-19`
relative L2 drift; synthetic silence must remain byte-exact. Exceeding any
ceiling is a test loss, not a reason to regenerate goldens or loosen the gate.

The compiled Rust operator test passed on 2026-08-06 (`1 passed; 0 failed`) and
reported these valid-log-mel comparisons:

| Fixture | Values | Bit-different | Max absolute | Mean absolute | Relative L2 |
|---|---:|---:|---:|---:|---:|
| `silence_320` | 256 | 0 | `0` | `0` | `0` |
| `impulse_480` | 384 | 59 | `2.765655518e-5` | `2.073744933e-7` | `2.232957057e-7` |
| `tone_640` | 512 | 323 | `1.392364502e-4` | `2.987362677e-6` | `8.583203051e-7` |
| `partial_tail_321` | 256 | 150 | `1.716613770e-5` | `6.364425644e-7` | `4.189396136e-7` |

The comparator independently loads the Rust Hann/mel buffers from the admitted
L0 model package and the expected values from the activation pack. It currently
reconstructs and compares only `log_mel_f32`; the other captured frontend stages
are authenticated source-replay evidence, not yet native Rust stage parity.

The version-2 public pack uses four frozen VoxConverse development fixtures: an
exact two-chunk one-speaker case, the complete 102-second overlap-bearing
three-speaker `mevkw` recording, a second complete three-speaker recording, and
a complete four-speaker recording. The descriptor, source version, license
acknowledgement, recording/annotation hashes, clip PCM hashes, exact sample
intervals, and NVIDIA recommended `340/1/40/40/300/188` streaming profile are
bound in the receipt. No transcript is used.

Five runs at one PyTorch intra-op thread and five runs at eight threads captured
4,540 named L1-L8 seams. To avoid a multi-gigabyte replay artifact, every full tensor
retains its shape, element/byte count, and baseline SHA-256 while tensors larger
than 4,096 values store a deterministic endpoint-inclusive stratified probe.
Small and discrete tensors remain complete. The resulting metadata-free package
is 72,590,196 bytes with SHA-256
`4ec66cf29e4286fed21fdf3d9c170293aafb26ba9783b9e0eea4d245b4630a6d`;
the 5,092,023-byte canonical receipt SHA-256 is
`8dd949aeccc0754338c3c777e8ef596f043387a2a38543f0a91353d06f70234f`.
The reviewed exporter source SHA-256 is
`af752ee007d46eb010d69109cc8c6f4f753f0304d30add401e114066a4a2f877`.
Both stay operator-local.

All discrete probes and cache/FIFO Option transitions were byte-exact. Numeric
thread-regime drift appeared in 873 probes; each accepted source tolerance is
zero when exact and otherwise the smallest power of two at least twice the
measured baseline-to-replay floor. The rule was frozen before Rust neural
results existed. That pack completes source truth rather than substituting for
native parity; the subsequent native results are recorded below. Broad
accuracy/resources and automatic routing remain separate gates.

The independent Rust/PyTorch L2 comparison does not misuse a zero source floor
as a cross-kernel equality requirement. Before running the native L2 path, it
freezes per-seam ceilings of `2^-10` maximum absolute drift and `2^-16`
relative L2 drift. The effective limit is the maximum of that predeclared
cross-kernel ceiling and the independently authenticated source-replay floor.
These constants are compiled into `src/sortformer_inference.rs`; a native loss
must be fixed or reported and must not be followed by loosening the ceiling.
The immediately following FastConformer block-input seam multiplies L2 by
`f32(sqrt(512))`; before that comparison its absolute ceiling is therefore
frozen separately at `2^-5`, while its scale-aware relative-L2 ceiling remains
`2^-16`.

The local Apple Silicon public operator run on 2026-08-06 passed in 132.05
seconds (`1 passed; 0 failed`). It compared all six L2 seams for all 17
streaming transitions (102 seam probes) and the first two reconstructable L3
block-00 input seams for each of the four fixtures (8 seam probes). The worst
L2 result remained `3.232955933e-4` maximum absolute drift and
`1.559432089e-6` relative L2. The worst L3 handoff result was
`7.324218750e-3` maximum absolute drift and `1.467437671e-6` relative L2,
below the frozen `2^-5` and `2^-16` limits respectively. The L3 harness
prepends exact prior pre-encode embeddings while that state is independently
reconstructable. It deliberately stops after the first cache-compression
transition because subsequent speaker-cache selection depends on L5
probabilities; claiming those later L3 inputs before implementing L3-L6 would
be circular. Thus this is complete L2 evidence and bounded L2-to-L3 handoff
evidence, not FastConformer or whole-model parity.

Before executing the first native FastConformer operator comparison, the
block-00 `feed_forward1` seam freezes cross-kernel ceilings of `2^-8` maximum
absolute drift and `2^-14` relative L2 drift. This seam includes the preceding
LayerNorm, both affine projections, and Swish, but excludes the subsequent
half-step residual. As with every earlier gate, a loss cannot be followed by a
tolerance increase.

The resulting local Apple Silicon run passed all eight reconstructable
block-00 `feed_forward1` probes in 133.66 seconds (`1 passed; 0 failed`). The
worst observed maximum absolute drift was `1.434326172e-3`; the worst observed
relative L2 was `1.249860390e-6`. Both remain below the predeclared limits.
This proves the first LayerNorm and FFN operator seam only; it does not yet
prove the half-step residual, attention, convolution, second FFN, or block
output.

Before executing the next native comparison, the block-00 raw Q/K/V affine
seams freeze cross-kernel ceilings of `2^-7` maximum absolute drift and
`2^-13` relative L2 drift. These seams include FFN1's half-step residual and
the self-attention LayerNorm. They stop before head reshaping, relative
position scoring, softmax, or the attention output projection.

The local Apple Silicon Q/K/V run passed all 24 probes in 185.35 seconds
(`1 passed; 0 failed`). The worst maximum absolute drift was
`6.843358278e-5`, and the worst relative L2 was `9.905910225e-7`, both well
inside the frozen limits.

Before executing the complete block-00 attention comparison, its output seam
freezes cross-kernel ceilings of `2^-6` maximum absolute drift and `2^-11`
relative L2 drift. This adds sinusoidal relative positions, the biasless
position projection, Transformer-XL relative shift, score scaling, stable
softmax, value reduction, and the output affine. The accepted public path has
no padding or finite-context mask at these exact-length synchronous seams.

The first cache-free complete attention probe passed locally on Apple Silicon
(`1 passed; 0 failed`, test runtime 192.52 seconds) with
`1.907348633e-4` maximum absolute drift and `9.384883845e-7` relative L2.
This validates the relative-attention equations and indexing at one public
block-00 seam. The remaining seven reconstructable attention seams and later
blocks remain open until the reference contractions are moved off the slow
scalar path.

Before running the remainder of block 00, the captured raw depthwise-
convolution seam freezes `2^-5` maximum absolute and `2^-12` relative-L2
ceilings. The captured FFN2 and final block-output seams freeze `2^-4`
maximum absolute and `2^-10` relative-L2 ceilings to account for cumulative
upstream drift. These gates are fixed before observing native results.

The first complete block-00 tail probe then passed locally on Apple Silicon
(`1 passed; 0 failed`, test runtime 155.49 seconds). Against the authenticated
public activations, the raw depthwise-convolution seam had
`3.004074097e-5` maximum absolute drift and `9.965494622e-7` relative L2;
FFN2 had `6.637573242e-4` and `1.041926232e-6`; and the final normalized block
output had `4.339218140e-5` and `9.981426293e-7`. The attention output in the
same invocation reproduced the prior result (`1.907348633e-4` and
`9.384883845e-7`). All four probes were inside their predeclared gates. This is
one cache-free public seam, not yet all eight reconstructable seams or all 17
FastConformer blocks; those remain explicit promotion blockers.

A subsequent same-process, same-input A/B retained the scalar attention as the
incumbent and compared the FrankenTorch matrix-kernel candidate on the exact
189-frame public tensor. The scalar call took 3.309788 seconds and the candidate
took 1.514581 seconds, a 2.18x attention-only speedup. Candidate-versus-incumbent
drift was `1.106262207e-4` maximum absolute and `3.386240615e-7` relative L2;
the candidate independently passed the authenticated NVIDIA output seam with
`1.869201660e-4` and `9.443736629e-7`. The surrounding full-test wall time was
not accepted as a speed result because unrelated host contention varied sharply.

The complete block was then checked at all eight independently reconstructable
public seams (the first two transitions of each fixture): 32 attention,
depthwise-convolution, FFN2, and final-output comparisons all passed. Worst
maximum absolute drift was `2.536773682e-4` for attention,
`3.600120544e-5` for depthwise convolution, `1.037597656e-3` for FFN2, and
`4.458427429e-5` for the final output. The worst relative L2 among these probes
was `1.330486704e-6`. In this more favorable same-process timing sample the
scalar and matrix-kernel attention calls took 0.762980 and 0.174491 seconds,
respectively. Therefore the observed exact-input speedup range is 2.18x to
4.37x; host contention prevents a stronger throughput claim. Block 00 is now
public-seam complete for all independently reconstructable states. Layers 01-16
and the later compressed-cache transitions remain open.

Before executing the chained 17-layer comparison, layers 01-16 freeze the
existing cumulative block ceilings of `2^-4` maximum absolute and `2^-10`
relative-L2 drift for block input, FFN1, Q/K/V, attention, FFN2, and normalized
output. The raw depthwise-convolution seam retains its tighter `2^-5` and
`2^-12` ceilings. Block 00 retains the already-proven specialized gates above.
These thresholds are fixed before observing any native later-layer result.

The generalized safe-Rust implementation then passed the complete L3 public
gate locally on Apple Silicon: all 1,224 comparisons (`17 layers * 9 seams * 8
independently reconstructable states`) passed in a 261.52-second test run. The
largest absolute drift anywhere was the already-recorded block-00 input handoff,
`7.324218750e-3`. The worst relative L2 in the chained encoder was
`4.926325194e-6` at fixture `syiwe_complete_three_speakers`, step 000,
block-16 FFN2. All eight block-16 outputs passed; their largest maximum absolute
drift was `3.659725189e-5`, and their largest relative L2 was
`3.840209473e-6`. This closes the independently reconstructable L3 seam gate.
It does not claim later compressed-cache states or any L4-L8 result.

Before the first native L4 comparison, the encoder projection and every
captured seam in the 18 post-LayerNorm Transformer blocks freeze ceilings of
`2^-4` maximum absolute and `2^-10` relative-L2 drift. The exact source graph
uses projection `512 -> 192`, eight 24-wide attention heads, separate
query/key division by `sqrt(sqrt(24))`, no future mask at these all-valid
synchronous states, post-attention residual then LayerNorm, ReLU FFN
`192 -> 768 -> 192`, a second residual, and a second LayerNorm. Evaluation
dropout is inactive. These gates and equations are fixed before native results.

The first native L4 run failed closed at the block-00 stage named
`attention_output` (`2.177211221` maximum absolute drift and
`0.9889359979` relative L2); neither gate was changed. Direct reconstruction
inside the pinned PyTorch oracle exposed an important capture detail: the CPU
forward hook stores a detached but non-cloned alias of `out_projection`.
NeMo immediately executes `self_attn_output += encoder_query`, so the frozen
stage actually contains the post-residual, pre-LayerNorm state. The analogous
`dense_out` hook is likewise mutated by its following residual addition. The
native comparator must therefore compare those two frozen stage names against
the post-residual states while the model computation itself remains the exact
source equation. Raw Q/K/V and final block outputs are unaffected. This is a
truth-pack seam-semantics correction, not permission to relax a numeric gate;
complete L4 parity remains unclaimed until a repaired run passes.

After that interpretation repair, the same local Apple Silicon gate first
passed one complete streaming state through the projection and all 18 chained
Transformer blocks: 145 authenticated comparisons, with no tolerance change.
The expanded run then passed all eight independently reconstructable public
states: 1,160 authenticated L4 comparisons in a 399.92-second invocation. The
worst maximum absolute drift was `4.321336746e-5` at fixture
`hiyis_exact_two_chunks`, step 001, block-00 feed-forward output. The worst
relative L2 was `4.516767665e-6` at the same fixture and step's block-00
attention value. All eight block-17 outputs passed; their maximum absolute
drift ranged from `2.190470695e-6` to `6.370246410e-6`. This closes L4 for the
eight states that do not depend on native cache compression. Later states
remain dependent on L6 chaining.

Before any native L5 result, the speaker-head seams (`hidden`, `logits`,
`probabilities`, and the context-trimmed `stream_output`) freeze the same cumulative ceilings of `2^-4` maximum
absolute and `2^-10` relative-L2 drift. The exact evaluation graph is ReLU on
the block-17 output, identity dropout, affine `192 -> 192`, ReLU, identity
dropout, affine `192 -> 4`, and elementwise sigmoid. The raw hidden affine
output is captured before its following ReLU. These are four model capacity
lanes, not evidence that every input contains four active speakers.

The same 399.92-second local invocation also executed all four L5 seams for all
eight independently reconstructable states and passed. Because that command's
evidence filter retained the L4 aggregate and terminal test result rather than
an L5 aggregate, this establishes gate success but does not yet provide a
publishable L5 worst-drift row. The full chained L1-L6 run must retain an L5
aggregate before L5 is treated as fully documented.

Before the first native L6 comparison, every floating cache/FIFO/state boundary
freezes ceilings of `2^-4` maximum absolute and `2^-10` relative-L2 drift;
integer silence counts and Option presence are exact. The accepted recommended
state has a 188-frame speaker cache, 40-frame FIFO, 300-frame update period,
340-frame chunk, four speaker lanes, and three silence sentinels per lane. The
native implementation preserves initially absent cache/FIFO prediction
Options, populates both after the first transition, updates the cumulative
silence profile at probability-sum `< 0.2`, and uses the exact source
score/disable/strong-boost/weak-boost/latest-boost/global-top-k compression
order. Speaker permutation and random score noise remain absent in evaluation.
These gates and equations were fixed before the accepted native L6 result.

An exploratory, separately run probe reported two f32 CPU runs with one PyTorch
intra-op thread and two with eight intra-op threads on one public
53.603313-second input whose PCM-file SHA-256 is
`c8bc396e0d6a257b45c5d200f91eeff73ee3f5b21ee0bb0bfc3417356b86fe22`.
The probability tensor shape was `[1, 670, 4]`.

- repeated runs within each thread regime were bit-identical;
- changing from one to eight threads changed 763 probability elements;
- cross-thread maximum absolute difference was
  `7.152557373046875e-07`;
- cross-thread mean absolute difference was
  `2.7044704253853524e-09`; and
- all four derived segment outputs had SHA-256
  `17bd517a696ff2301713b47fcacceb49a653f4dca3885ea4f63eeaeaa45ad5a2`.

Those older figures remain exploratory and non-authoritative; the identity-bound
four-fixture pack above supersedes them as source-floor evidence.

## 8. Ordered equivalence ladder

Each level is prerequisite evidence for the next. A later level cannot repair
or excuse a failed earlier level.

| Level | Required equivalence |
|---|---|
| L0 Artifact | Exact model/config/source identity and complete conversion receipt |
| L1 Frontend | Frame count, valid lengths, window, spectrum, mel, and log-mel tensors |
| L2 Subsampling | Every convolution stage and final factor-8 encoder input |
| L3 FastConformer | Input/output and selected internals for all 17 blocks |
| L4 Projection and transformer | Projection plus all 18 transformer blocks |
| L5 Head | Four sigmoid activity probabilities before streaming mutation |
| L6 Streaming state | Every cache/FIFO tensor, length, selection, and proof that permutation remains absent in eval |
| L7 Discrete activity | Thresholded speaker activity under the frozen output geometry |
| L8 Final document | Arrival labels, turns, speech, overlap, and changes pass the strict existing validator |
| L9 Public task behavior | DER/JER/count/overlap/calibration on frozen development and sealed test rows |
| L10 Resources | Same-invocation CPU, RTF, peak RSS, allocation, and cancellation gates |

Numeric reports must include maximum absolute error, mean absolute error, a
scale-aware relative statistic, mismatch counts over the declared tolerance,
and the source-oracle floor. Discrete reports must be permutation-aware where
speaker labels are arbitrary. Whole-model equality without seam evidence is
insufficient.

## 9. Open-question register

| ID | Question | State | Gate |
|---|---|---|---|
| OQ-01 | Does evaluation inject configured dither? | Resolved: no | L1 |
| OQ-02 | Exact STFT padding, mel floor, and length rounding | Resolved for the synthetic pack and native valid-log-mel comparator; public real-voice extension open | L1 |
| OQ-03 | Depthwise subsampling padding and layout at tails | Resolved by authenticated public tail parity | L2 |
| OQ-04 | Exact relative-position attention equations, masks, and scaling | Resolved across all 17 native blocks and public states | L3 |
| OQ-05 | Exact inference head branch | Resolved across all 18 Transformer blocks and the four-lane head | L5 |
| OQ-06 | Top-k tie behavior and chronological reordering | Resolved for the accepted host/profile by the pinned safe libc++ 15.0.7 `nth_element` translation and exact chained probe | L6 |
| OQ-07 | First, steady, and partial-tail cache mutation | Resolved for the recommended 188-cache/40-FIFO/300-update profile, including short final chunks | L6 |
| OQ-08 | Speaker permutation during accepted inference | Resolved: disabled and absent in eval | L6 |
| OQ-09 | Converted package tensor map and transforms | Resolved: exact 992-record manifest, receipt, and 974-tensor package admitted | L0 |
| OQ-10 | Cross-input and cross-thread oracle variability | Synthetic and licensed-public source floors complete; other native platform/runtime tiers remain open | L1-L8 |
| OQ-11 | Model bytes in repository or releases | Resolved: forbidden for initial route | L0 |
| OQ-12 | Known requirement above four speakers | Model ineligibility resolved; product fallback pending `bd-y4ip.14` | L8-L10 |
| OQ-13 | Known timestamp intervals during parity | Resolved: post-forward mapping only | L9 |
| OQ-14 | Other operating systems and CPU feature tiers | Open; require separate runtime rows | L10 |
| OQ-15 | Frontend/postprocessing transitive runtime identity | Open: pin it or remove it from seam authority | L0-L1 |
| OQ-16 | Unknown recording actually contains more than four speakers | Open: capacity sentinel or capped status | L9 |
| OQ-17 | Activity still open at the final 10 ms sample versus strict 80 ms output validation | Resolved: raw L8 matches NeMo exactly; production turns clamp to physical duration | L7-L8 |

The fit screen has no remaining upstream semantic question that invalidates its
conditional-GO result. OQ-03 through OQ-08 and OQ-17 are resolved for the
accepted same-host recommended-profile route. The transitive runtime boundary
in OQ-15, cross-platform support in OQ-14, and capacity/product policy in
OQ-12/OQ-16 remain open. OQ-14 does not block this same-host evaluation route,
but it still blocks a broad platform-support claim.

## 10. Implementation slices

### Slice A: truth and conversion (`bd-y4ip.10`)

1. Extend the public-input oracle floor. **Complete for the frozen four-fixture
   licensed-public L1-L8 pack.**
2. Define the canonical conversion-receipt schema. **Complete for L0 v1.**
3. Convert the pinned checkpoint outside Git into safetensors. **Complete for
   the identity-bound, verifier-admitted package now distributed through the
   dedicated model release; no weights entered Git.**
4. Audit the exact tensor census and hashes. **Complete for all 992 source
   records, 974 exports, and 18 typed drops.**
5. Capture all L1-L8 real-voice oracle activations outside Git and retain only
   their identities and aggregate drift; commit exact activation values only
   for deterministic synthetic non-human fixtures. **Complete for the frozen
   source pack; native parity remains Slice B work.**
6. Add tamper and identity-drift tests. **Complete for compiled receipt/package
   roots, strict schemas, stage/full-shape/probe/hash checks, source-floor
   margins, and cache Option transitions; broader f32 seam mismatch controls
   remain Slice B work.**

### Slice B: safe f32 engine (`bd-y4ip.11`)

Add a genuinely separate `sortformer_inference` module. It should load only
the identity-bound non-executable package, use one explicit CPU-only
FrankenTorch facade, checkpoint cancellation between bounded chunks and every
layer, and initially return `DifferentialStageDocument`. The strict Sortformer
validator in `src/differential_oracle.rs` is the first output boundary.

This slice must not change `DiarizationEngine`, automatic routing, or the
production transcribe report contract. Comparison protocol v5 adds the native
Sortformer lane without changing product routing. The explicit
`sortformer-diarize` command is a path-redacted evaluation surface over the
same library session; it reports `evaluation_only` and cannot be selected by
`Auto`.

### Slice C: evaluation worker (`bd-y4ip.13`)

After whole-forward f32 parity, wire a native Sortformer worker only into the
existing evaluation/comparison lane. It must emit typed provenance and
capacity status and must not alter resolver defaults, production routing, or
the comparison protocol's authority. Public accuracy and resource rows remain
required before product integration.

### Slice D: optimization (`bd-y4ip.12`)

Only after f32 parity:

1. profile the whole forward pass;
2. apply a stagewise quantization ladder and retain drift reports;
3. establish reference-kernel equivalence for each candidate hot path;
4. fuse one measured bottleneck at a time; and
5. compare the live production baseline and candidate in the same invocation with
   matched audio, normalization, threads, diarization parameters, scoring,
   decoding, and transcript equivalence.

A self-speedup, unmatched thread setting, changed scoring rule, or external
oracle timing row is not a native performance win.

### Slice E: product integration (`bd-y4ip.14`)

After accuracy and resource certification, add a model-neutral native report
kind, typed Sortformer provenance, resolver/cache behavior, count and capacity
semantics, waveform preservation, CLI and robot parsing, persistence, and a
versioned comparison protocol, including an explicit capacity sentinel or a
candid four-lane-capped status. This slice also owns typed hard/soft known-
interval mapping, contradiction tests, immutable per-call weights, and explicit
anonymous-versus-identified output semantics. Identified output and any `Auto`
promotion require a mapping-specific frozen gate with predeclared public and
adversarial inputs, mapping accuracy, calibration, coverage, and false-
identification evidence; failure preserves anonymous output. `Auto` remains
unchanged until that work passes its own frozen gate.

## 11. Privacy, safety, and failure behavior

- No confidential audio or transcript is part of conversion, parity, or
  committed test evidence.
- Model packages remain outside the repository. The native f32 weights are
  distributed only as a dedicated GitHub release asset beside the NVIDIA
  license, required notice, and conversion receipt; `fw pull sortformer`
  verifies every embedded size and SHA-256 before cache admission. All
  real-voice activation or embedding values are ephemeral; Git may retain their
  hashes and aggregate drift only. Exact committed values use synthetic
  non-human inputs.
- Retained Git, robot, and oracle evidence uses hashes and typed failure codes,
  not local paths, audio samples, transcripts, speaker names, or embeddings.
  Operator-local conversion commands and terminal diagnostics may contain local
  model/output paths and must not be copied into committed evidence.
- The Rust loader is safe code and rejects executable checkpoint formats.
- Every future model allocation must derive from checked tensor dimensions and
  declared bounds.
- The completed loader/frontend uses fallible bounded allocation and checks
  cancellation before and after large reservations, during bounded hashing and
  loading, during chunked zero-fill, and throughout frontend work. The future
  engine must also check every layer, every chunk, cache mutation,
  post-processing, and validation.
- Timeout or cancellation must not leave an inference worker mutating shared
  state after the caller returns.
- Unsupported capacity, missing model, identity drift, malformed output, or a
  resource breach must produce a typed skip; a fallback may be invoked only
  after `bd-y4ip.14` separately certifies its applicable capacity, accuracy,
  privacy, and resource envelope. Neither path may silently degrade into a
  success row.

## 12. Current proof state

Completed here:

- full-model fit-screen and architectural selection;
- primary upstream, artifact, config, source, and runtime pins, with the
  transitive-closure gap explicitly open;
- license and model-byte distribution policy;
- aggregate parameter/export/drop/operator/state census plus the pinned
  externally-derived insertion-order inventory digest;
- a reviewed frozen offline converter, canonical 992-record topology projection,
  653,208-byte receipt, and 491,570,584-byte metadata-free package, all with
  compiled digests while licensed bytes remain outside Git and independently
  release-distributed with legal sidecars;
- explicit streaming and capacity contract;
- one-host exploratory nondeterminism probe, explicitly non-authoritative;
- a cycle-free, dependency-wired implementation and proof ladder;
- a safe conversion-receipt/package verifier with compiled trust roots, exact
  topology recomputation, strict safetensors parsing, fallible allocation,
  regular-file and path-swap defenses, synthetic tamper tests, and
  operator-local real-package admission proof; and
- separate synthetic and licensed-public activation receipt/package verifiers
  with compiled trust roots, ten-run source floors, all 4,540 L1-L8 public seam
  contracts, and exact operator-local activation admission boundaries; and
- a pinned-buffer, source-derived bounded whole-file Rust log-mel frontend with
  fallible allocation, mathematical/unit tests, and compiled valid-log-mel
  parity inside the frozen synthetic cross-kernel envelope;
- authenticated native L2 depthwise subsampling across the admitted public
  states;
- all 17 native FastConformer blocks across eight independently reconstructable
  public states, including exact prior-cache handoff and 1,224 L3 comparisons;
- the encoder projection and all 18 native Transformer blocks across those same
  states, with 1,160 L4 comparisons; and
- the complete four-lane sigmoid speaker head across those same eight states;
- exact-score and complete chained L6 cache/FIFO compression inside the frozen
  gates, including the pinned safe libc++ tie behavior and short final chunks;
- byte-exact L7 activity/speech/overlap/change and L8 anonymous-turn parity on
  every complete discrete public truth-pack tensor, including the full
  102-second three-speaker recording and complete four-speaker fixture;
- an authenticated `SortformerSession` with checked whole-recording streaming,
  neural-block cancellation checkpoints, bounded resource validation, and
  physical-duration tail clamping;
- an explicit, cancellation-aware `fw pull sortformer` path plus a
  path-redacted cached `sortformer-diarize` evaluation-only CLI whose real
  public three-speaker run inferred three active lanes; and
- one overlap-heavy two-speaker public DER/JER row plus local debug RTF,
  model-load, and child-only peak-RSS observations recorded above.

Not completed here:

- native intermediate frontend-stage parity before `log_mel_f32`;
- quantized whole-model parity or additional profile-justified fused kernels;
- release-build and 90-minute resource certification;
- frozen public multi-record accuracy certification; or
- transcribe/robot report integration or automatic production routing.

Those omissions are downstream gates, not evidence that they passed.
