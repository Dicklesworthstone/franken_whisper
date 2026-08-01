# Acoustic Diarization Contract v2

Status: implementation contract for `bd-odj7`
Contract identifier: `acoustic-diarization-v2`

## 1. Purpose and authority

The native acoustic diarizer answers “who spoke when?” from the normalized
waveform. It does not infer speaker identity, gender, or legal identity. Its
speaker references are opaque within-run cluster identifiers or references
provided explicitly by the caller.

This contract governs classical acoustic, later neural, and external
implementations after output normalization. An implementation cannot call
itself acoustic if it only examines timestamps, text, word counts, or segment
position.

The native engine contract remains authoritative for ASR text and timestamps.
This document adds the permutation-invariant speaker, confidence, supervision,
privacy, and bounded-resource contract that the ASR contract lacks.

## 2. Canonical data flow

```text
normalized 16 kHz mono PCM with finite samples in the closed [-1.0, 1.0] range
    -> VAD and speech-quality mask
    -> 25 ms frames / 10 ms hop
    -> separate voice and channel features
    -> multiscale acoustic change scores
    -> microturns and tracklets
    -> robust sufficient-statistic speaker profiles
    -> constrained deterministic clustering
    -> temporal smoothing and unknown rejection
    -> independent diarized-turn timeline
    -> DTW word-boundary transcript projection
```

`normalized_input_sha256` binds every result and hint range to the exact PCM
used by ASR. Existing RMS/VAD summaries may seed analysis but cannot substitute
for waveform samples.

## 3. State, actions, and loss

The decision state consists of:

- microturn voice and channel sufficient statistics plus quality masks;
- hard and soft known-speaker intervals;
- current robust profiles and channel subprofiles;
- one typed speaker-count request: inference, calibrated prior, range, or hard
  search constraint;
- prior temporal assignment;
- algorithm, feature-schema, weights, and calibration identities;
- remaining time, memory, and prototype budgets.

Available actions are:

- add or suppress a change boundary;
- merge compatible adjacent microturns into a tracklet;
- merge two compatible prototypes;
- assign a turn to a profile;
- create a new unanchored profile;
- return `UNKNOWN`;
- mark `overlap_suspected`;
- invoke a named deterministic fallback.

The initial loss hypothesis is a ledgered, calibratable policy rather than an
accuracy claim:

| Outcome | Initial loss |
|---|---:|
| Contradict a valid hard hint | forbidden |
| Wrong-speaker assignment or false speaker merge | 4.0 |
| False split of one speaker | 2.0 |
| Unknown assignment | 1.0 |
| Unsupported short speaker switch | 0.5 |

Changing these weights changes the policy identity and requires retained
evidence.

Speaker-change selection has a separate v2 loss contract. Its states are
`NoBoundary`, `Defer`, `EmitBoundary`, and `ConservativeFallback`. False splits
cost `1.0`; missed changes cost `9.0` because a later clustering stage can
reversibly merge an over-segmented tracklet, while an omitted boundary mixes
two acoustic regimes before clustering. The corresponding Bayes action
threshold is `1 / (1 + 9) = 0.10`. Timing, hint contradiction, latency, and
fallback costs are also frozen in `AcousticChangeCalibration`; changing any
value changes `acoustic_change_calibration_sha256`.

## 4. Acoustic evidence

The default `acoustic-feature-v2` schema maintains two views:

- **Voice:** twelve energy-centered filterbank cepstra, selected first and
  second differences, nullable fundamental frequency with a separate validity
  mask and uncertainty, periodicity, harmonic-to-noise ratio, three
  vocal-tract/formant proxies, voiced fraction, and temporal modulation.
- **Channel:** RMS and dynamics, spectral centroid/bandwidth/rolloff/flatness,
  effective band limit, high-frequency attenuation, spectral tilt, band
  ratios, crest/clipping, noise floor, stationary coloration, and conservative
  temporal-smearing, muffling, or distortion summaries.

Voice evidence is primary. Channel evidence is capped and may create multiple
channel subprofiles beneath one voice profile. Pitch is never sufficient by
itself and must never be translated into a gender label. Missing or uncertain
pitch is an invalid coordinate, never a physical zero.

### 4.1 Feature schema and aggregation

The library exposes a declarative `AcousticFeatureSchema` for v1 and v2. Every
family declares its coordinate range, units, voice/channel ownership, validity
rule, and normalization rule. The default is v2. The old eight-voice/eight-
channel v1 representation is reachable only through the explicit
`AcousticFeatureSchemaVersion::V1` segmentation entry point; it is not selected
by compatibility inference.

V2 owns 28 voice coordinates and 14 channel coordinates. Every tracklet,
enrollment observation, profile, prototype, and cluster carries the active
channel-coordinate prefix explicitly. V1 therefore compares exactly its first
eight channel coordinates, while a no-channel evaluation owns zero; neither
choice may be inferred from whichever voice coordinates happen to be valid.

V2 identity aggregation admits only voiced frames that are neither low-energy,
clipped, nor transient. It retains at most 64 deterministic highest-quality
subwindows per tracklet, then uses per-coordinate medians and MAD-derived
variance. A coordinate with no supporting subwindow remains invalid through
enrollment, prototype construction, clustering, and assignment distance.
Channel statistics separately admit usable non-clipped speech.

Per-recording normalization gives every provisional tracklet equal weight. It
uses the median and the larger of scaled MAD, scaled IQR, and a conservative
floor for each supported coordinate. This prevents a long or dominant speaker
from defining the center by frame count and erasing a short minority speaker.
The normalization never imputes a missing coordinate.

The frozen representation ablation surface is `full_v2`, `no_pitch`,
`no_channel`, `no_deltas`, `no_modulation`, and `v1`. These are experiment
configurations, not adaptive runtime selections. Each retained result records
the ablation ID, feature schema and hash, complete diarization request and
hash, and a configuration hash binding all of them to the runner version. The
speaker-count complexity penalty uses supported voice coordinates instead of
charging reduced representations for absent v2 dimensions.

`fw diarization-corpus ablate` executes all six representations and four
speaker-change detectors with the same frozen scorer: 250 ms speaker-boundary
collar, 250 ms primary change collar, excluded overlap, and ten calibration
bins. Change diagnostics additionally retain aggregate precision, recall, F1,
and p50/p90/p95 absolute timing error at 100, 250, and 500 ms collars, plus a
predeclared 19-point threshold sweep from 0.05 through 0.95. It uses reference
speech regions only as oracle VAD, subtracts ignored scoring regions, and never
supplies the reference speaker count. Source WAV/RTTM
files remain beneath an external input root. The validated reference bundle
and aggregate-only ablation evidence must be new JSON files in a separate
external directory; no source media or per-recording hypotheses are copied.
Evidence parsing recomputes schema, scorer, request, configuration,
development and held-out gates, and self hashes before accepting an artifact.
The artifact also carries a deterministic accuracy hash computed after
normalizing wall time, RTF, and process-RSS observations away, so identical
accuracy results can be distinguished from expected host-performance drift.

The representation decision is predeclared before observing corpus output.
On the frozen development split, full v2 must reduce micro-DER by at least 5%
relative to v1, must not increase macro-JER, and must not reduce change-F1.
On the frozen test split, full v2 must not increase either micro-DER or
macro-JER. Both decisions and their component deltas are part of the
self-hashed aggregate artifact; a run that merely completes is not a
successful representation result.

The v7 runner has an authority-bearing two-stage interface. `--stage
development` evaluates only the development split and cannot issue a held-out
verdict. `--stage certification` evaluates only the test split and requires
`--locked-development-evidence`; it verifies the exact development result
hash, accuracy hash, bundle, descriptor, duration protocol, calibration
identity, calibration-fit identity, detector-selection policy, and selected
detector and clustering mode before reading test audio. The v7 aggregate adds
selective coverage/risk, duration-weighted assignment calibration, typed
fallback counts, signed count error, and overlap TP/FP/FN. Overlap DER
exclusion does not remove intervals from the independent overlap detector
score. Test recordings observed during an earlier experiment are no longer
unseen and cannot be reused to mint a new promotion claim.

### 4.2 Frozen AMI representation result

The first non-synthetic run was frozen before its metrics were inspected. It
used the public AMI scenario split, oracle VAD but not oracle speaker count,
two recordings per development/test split, and deterministic 300-second
prefixes (600 seconds of scored audio per split). The path-free bundle hash is
`0d5219be241a3560cb55e3d8d9f63cd8d78ded4e46774d490583f34265eacd59`;
the performance-independent accuracy hash is
`9a0adb04b474d12a58f54cba63040b88b6f8139b73944c599603be95593c03b7`;
and the self-hashed aggregate result is
`de956df2fd96c690e866947fa44051ece31ea4e670933f4bf6058d7164b88696`.
The evidence JSON remains outside the checkout.

Full v2 reduced development micro-DER from `0.70067` to `0.53905` (23.07%
relative), reduced macro-JER from `0.90291` to `0.85863`, and raised
change-F1 from `0.09924` to `0.11429`, so the predeclared development gate
passed. On the frozen test slice, micro-DER fell from `0.69968` to `0.48714`
and macro-JER from `0.90965` to `0.85830`, so the held-out non-regression gate
also passed. Full-v2 RTF was `0.1069` on development and `0.0935` on test with
a sampled peak process RSS of 137,363,456 bytes on the measurement host.

The ablation is deliberately treated as diagnostic rather than uniformly
positive. Removing deltas made development micro-DER substantially worse
(`0.62330`), and removing modulation slightly worsened DER/JER and materially
worsened change-F1 (`0.09249`). Removing pitch or channel coordinates improved
DER/JER but reduced development change-F1. Most importantly, no variant found
the exact speaker count on this small slice; full v2 had mean absolute count
error `2.5`. The representation is therefore promoted over v1, while
pitch/channel fusion, boundary calibration, and count selection remain open
accuracy work. Because this test slice has now been observed, it must not be
reused as unseen promotion evidence after tuning; a new frozen test subset is
required.

### 4.3 Change posterior v2 development status

`acoustic-change-posterior-v2` replaces the raw Euclidean threshold with
bounded diagonal sufficient statistics and explicit detector ablations:
variance-aware GLR posterior, terminal Page-Hinkley/CUSUM approximation,
two-regime diagonal Bayesian/BIC approximation, and `FixedSafeV1`. All five
temporal scales remain inside a 401-frame ring. Voice and channel evidence are
fused separately; silence, legal word geometry, and optional TinyDiarize
support are typed bonuses rather than timestamp overrides. A coarse boundary
is refined inside a hash-bound ±300 ms neighborhood using spectral flux,
voicing and pitch discontinuity, energy valleys, and legal timestamp support.

Weak posterior excursions use bounded hysteresis: at most one candidate is
emitted during a 100-frame active interval and the detector re-arms only after
20 frames below half the action threshold. Evidence at or above 0.50 uses the
short 20-frame peak lane so rapid, unambiguous alternation is not swallowed.
Ill-conditioned covariance or insufficient voiced support invokes the frozen
fixed-safe score and records the fallback reason. The fixed-safe detector keeps
its original 20-frame suppression and remains the production default.

Development comparisons use the aggregate-only
`public-diarization-acoustic-ablation-v3` artifact. A 120-second-per-recording
AMI development diagnostic found that the Bayesian candidate reached change
F1 `0.13333`, Brier `0.21668`, and mean absolute boundary error `0.1975` s,
while the fixed-safe operating point matched no reference changes on that
slice. The candidate nevertheless failed promotion: ECE was `0.19496` against
the `0.10` gate, micro-DER regressed from `0.33854` to `0.51338`, and macro-JER
regressed from `0.52042` to `0.69470`. A two-second weak-peak suppression
experiment reduced recall to `0.08333` and was rejected. These are development
diagnostics, not certification.

Accordingly, normal segmentation and diarization entry points remain on
`FixedSafeV1`; posterior candidates are available only through explicit
ablation APIs. No held-out certification has been run for v2, and no
development result may be described as a promotion. The next accuracy work
must improve reversible clustering and boundary fusion before rerunning the
hash-locked development gate.

Privacy-safe diagnostics expose only schema and schema hash, aggregate
frame/missingness counts, supported dimension counts, bounded retained state,
and fallback/calibration state. They never expose feature values, audio,
transcript text, paths, or recording identifiers.

Wavelet evidence is limited to inexpensive Haar-like multiscale contrasts over
feature trajectories. A full raw-waveform CWT is not the default algorithm.

### 4.4 Probabilistic clustering v2 development status

The current clustering evidence uses
`public-diarization-acoustic-ablation-v7`,
`public-diarization-acoustic-ablation-runner-v7`, and
`diarization-scorer-v3`. It evaluates two public AMI development recordings,
each clipped to a deterministic 120-second prefix, with oracle VAD and without
oracle speaker count. The public bundle hash is
`34f405b6220d479f4d0d86937de77d51375ed39120abfd3a2f38e775a24e874e`;
the candidate calibration hash is
`fc286e7aec51d4b2362e3162bcd8a77451ef610a9e31f1295ba469856f4025ca`;
the performance-independent accuracy hash is
`4a0e62a073067c2d9c5f45378600844e240d9a0447954219ec6f28dd8d203f34`;
and the self-hashed result is
`8aef28a314c500feb33ff96afe233067fb03a6b92d093171967136e2ca8aac55`.
The aggregate artifacts remain outside the checkout.

Against fixed-safe clustering, the probabilistic candidate reduced micro-DER
from `0.27723` to `0.24649` (11.09% relative), reduced speaker confusion by
3.99 seconds, reduced mean absolute count error from `2.0` to `1.5`, increased
selective coverage from `0.87323` to `0.90952`, and slightly reduced selective
risk from `0.16741` to `0.16683`. All requested probabilistic runs completed
without fallback and mean five-view count stability was `1.0`. Wall time for
240 seconds of audio was `33.702` seconds (RTF `0.14043`) versus `33.757`
seconds (RTF `0.14065`) for fixed-safe; sampled peak RSS was 136,609,792 versus
136,265,728 bytes.

The candidate did not pass. Macro-JER regressed from `0.51001` to `0.58986`,
assignment ECE was `0.21716` against the `0.10` limit, and both modes had
overlap F1 `0.0`. An intermediate monotone square-root confidence experiment
reduced ECE to `0.13968` but still missed the limit and was not selected.
Accordingly, `selected_clustering_mode` remains `fixed_safe_v1`, no candidate
lock was minted, and no held-out recording was read. These results establish
real public development evidence and performance observations, not production
promotion or held-out certification.

The subsequent `acoustic-clustering-probabilistic-v3-development` count
candidate adds a separately versioned `speaker-count-estimate-v2` report. It
has not inherited the v2 evaluation authority: until a new frozen public
development bundle passes the count, DER/JER, calibration, determinism, memory,
and latency gates, the normal assignment path remains `fixed_safe_v1`. Native
fixed-safe runs still emit the count-estimate object, but with
`fixed_safe_uncalibrated`, no concrete bins or selected count, and all
probability mass assigned to `unresolved`.

## 5. Known intervals

`speaker-hints-v1` carries:

- `speaker_ref`: non-empty opaque reference;
- finite `start_ms < end_ms` within the normalized audio;
- confidence in `[0, 1]`;
- `hard_must_link` or `soft_enrollment`;
- optional provenance metadata;
- canonical document hash.

Hard intervals with different references may not overlap after sample
quantization. Hard intervals are immutable assignments, but enrollment still
removes boundary guards, non-speech, and low-quality frames. An interval with
no usable speech fails rather than creating an empty trusted profile.

Soft hints contribute capped pseudo-counts and priors. They can be rejected
when acoustically contradictory. Provenance is audit metadata and cannot
increase confidence by itself.

Profile training is independently audited from attribution. A hard interval
remains an immutable assignment even when its acoustic observation is
quarantined from profile training. Training uses robust global and
leave-one-out distance checks, a nearest-peer contamination check, and a
low-voiced-coverage downweight. One speaker may retain at most four voice
subprofiles and multiple channel subprofiles, which preserves repeated vocal
or recording modes without allowing unbounded profile growth.

Within-call metric adaptation is conservative and reversible. It requires at
least two enrolled speakers, two observations per speaker, and six total
observations. Per-coordinate weights are bounded to `[0.9375, 1.0625]`; unmet
support returns exact unit weights with a typed fallback. Durable profile
summaries expose only counts and policy outcomes, never reusable voice or
channel vectors.

The native acoustic contract accepts at most 1,024 known intervals per
request, 256 bytes per speaker reference, and 4,096 bytes per provenance
value. These limits are validated before the hard-hint overlap check.

## 6. Output and confidence

The diarized-turn timeline is the acoustic source of truth. Each turn contains:

- finite monotonic start and end;
- optional speaker ID, where absence means unknown;
- independent speaker-assignment and change confidence;
- source implementation and feature-schema identities;
- anchored/inferred source;
- overlap suspicion and fallback status.

ASR segment `confidence` remains ASR confidence. Speaker confidence is a
separate field. Transcript projection may split only at legal DTW word
boundaries and cannot invent, drop, duplicate, or reorder text.

### 6.1 Speaker-count and evidence result

`SpeakerCountRequest` has exactly one mode:

- `Infer` selects only speakers that pass acoustic evidence gates;
- `Prior` carries sorted, unique probability mass over positive counts as soft
  evidence pooled into at most 15% of concrete posterior mass; counts outside
  its support remain eligible, and five-view acoustic agreement attenuates the
  prior to 7.5% at unanimity;
- `Range` is a soft uniform preference inside the interval, not a hard search
  bound;
- `HardConstraint` searches for exactly the requested count but does not assert
  that the count exists in the recording.

Every mode keeps `UNKNOWN` legal. Only a `hard_must_link` interval can prohibit
`UNKNOWN` for its own tracklet. Count cardinality never suppresses rejection,
and no assignment may be changed merely to make a missing label appear.

An inferred speaker must have retained occupancy and quality evidence:
independent voiced support, recurrence or repeated tracklets, assignment
confidence, profile reliability, and separation from already-supported
speakers. A hard-hinted speaker is supported by the hard attribution even when
its sample is quarantined from profile training. Soft-enrollment observations
cannot validate themselves; an uncorroborated soft name remains an audited
hint and does not become a fabricated profile or output label.

`speaker_count` in the stable report contains the original typed request,
`resolved`, `satisfied`, `unsatisfied`, or `unresolved` status, the
evidence-supported count, active references, dominant and unknown voiced
shares, structured reasons, and a per-speaker evidence summary. `hint_evidence`
separately records a privacy-safe disposition and count audit for every known
interval. The report does not expose acoustic feature values.

Adjacent assignments merge into a turn only when their hard-hint and overlap
provenance agree. This prevents a long merged turn from laundering a weak or
unsupported assignment into the confidence of a neighboring tracklet.

### 6.2 Canonical DTW projection timeline (`bd-2noc.1`, `bd-2noc.2`)

`group_tokens_into_words` quantizes token boundaries to the alignment grid and
may legitimately emit a terminal `[t, t]` observation. That decoder
observation is not itself a projection interval. The native backend now owns
one conversion from raw observations into `dtw-projection-v2` canonical units
before diarization or persistence can consume them.

A canonical unit has:

- finite `f64` seconds;
- half-open `[start_sec, end_sec)` semantics;
- non-negative, strictly positive duration;
- monotonic, non-overlapping order;
- a minimum word-aligned duration of 1 ms, because acoustic boundary hints are
  integer milliseconds;
- source segment and word indices;
- boundary provenance and explicit clamp/expansion status.

The adjacency epsilon is 1 microsecond. It exists only to absorb
floating-point noise around a boundary that is intended to be identical. It
is not the 50 ms cross-engine comparison budget. A sub-epsilon overlap is
canonicalized to exact adjacency. Material overlap, reversed time,
non-finite time, negative time, unpaired parent timestamps, materially
overlapping parents, and extra timing vectors fail with stable
`FW-DTW-PROJECTION-*` reasons.

For valid DTW observations, the adapter reserves 1 ms for each remaining word,
clamps observations to the enclosing decoder segment, and expands quantized
zero-width units within that reservation. This repairs a terminal `[t, t]`
without inventing, dropping, duplicating, or reordering text. Punctuation-only
content remains legal.

Missing DTW words for a segment use deterministic linear interpolation within
that parent segment. If a parent is too short to represent every word as a
distinct millisecond interval, the adapter emits one parent-segment unit. Both
cases retain fallback provenance and set `word_aligned_safe=false`; acoustic
projection then uses conservative duration dominance rather than claiming
legal DTW word boundaries. `no_timestamps` also disables that claim.

The privacy-safe `projection_timeline` raw-output object records the schema,
units, interval semantics, tolerances, input/output counts, provenance counts,
adjustment counts, and `word_aligned_safe`. That object contains no transcript
text, recording path, or speaker identity. The orchestrator trusts the typed
`projection_timeline.word_aligned_safe` field rather than inferring legality
from the older descriptive `word_timestamps` string.

`fallback_reasons` is an ordered list of stable reason strings. It distinguishes
missing decoder word timestamps, insufficient parent duration for distinct
millisecond word units, and timestamp suppression requested by the caller.
This required field and the durable summary contract are the reason the schema
identity is v2; consumers must reject v1 rather than silently assuming the new
fallback semantics.

The simulated proportional CTC alignment stage does not rewrite a timeline
carrying that proof. Real decoder attention-DTW offsets are authoritative, and
mixing proportional corrections with per-segment fallbacks can otherwise
reintroduce overlap at the end of the recording. The align stage records a
deterministic preservation note and leaves canonical segment bytes unchanged.

SQLite `result_json` is the authoritative durable copy of the typed
diarization report and the projection timeline. `RunStore` exposes both after
restart, and JSONL export/import carries the same canonical payload into a
fresh database before rebuilding its normalized diarization indexes. Robot
`run_complete` deliberately emits speaker turns and projected segments but not
the backend raw-output object, which can contain internal model paths; durable
projection provenance is retrieved through the stored-run surface instead.
That surface exposes only the privacy-safe `projection_timeline` sub-object,
not the rest of backend raw output or its internal model paths.

The fixed-safe speaker assignment combines best-versus-second margin with
profile reliability and reports `heuristic_uncalibrated`. The probabilistic
development candidate reports a separately versioned likelihood calibration;
the raw likelihood, not the reported mapping, controls unknown rejection so a
confidence transform cannot silently expand coverage. Speaker-change
candidates separately carry the versioned v2 posterior, component evidence,
supporting-scale mask, refinement offset, detector identity, fallback reason,
and calibration hash. The retained scoring surface reports Brier score,
expected calibration error, reliability bins, threshold sweeps, and coverage
when ground truth is available, so returning unknown or emitting no boundaries
cannot appear successful.

The probabilistic temporal path uses duration-aware continuity: switching cost
depends on current run length, next-tracklet duration, inter-tracklet gap,
boundary confidence, and whether `UNKNOWN` is involved. Short unsupported
fragments receive a penalty, while a strong acoustic boundary or real gap earns
bounded credit. The fixed-safe path retains its original constants, and every
candidate parameter is part of the speaker-pair calibration hash.

Overlap is an independent acoustic claim. The tracklet aggregates a bounded
dual-periodicity probability only from non-clipped, non-transient frames. A
probabilistic assignment emits a second speaker only when both independent
speaker likelihoods exceed their floor and their ratio is sufficiently close;
otherwise `overlap_suspected` remains diagnostic. A supported secondary
assignment projects to two simultaneous turns. A secondary label without
overlap evidence fails closed.

`speaker_queries` is a bounded active-agent surface for unknown, low-confidence,
or overlap-ambiguous spans. Adjacent requests are merged and capped at 32.
Each query ID hashes the input hash, interval, reason, candidates, and policy.
Queries contain no audio, transcript text, feature values, or path. When an
agent supplies a known interval in response, its provenance changes the
canonical hint hash but never its acoustic weight.

## 7. Determinism and resource limits

- Frame cadence is 16 kHz, 400 samples, 160-sample hop for feature schemas v1
  and v2.
- Cancellation is checked at least every 32 frames and within clustering and
  smoothing. Projection is bounded linear work inside the independently
  budgeted Diarize stage; persistence and cleanup retain their existing
  pipeline cancellation boundaries.
- Whole-call raw frame matrices and full raw-audio wavelet transforms are
  forbidden. `AcousticFeatureStream` accepts arbitrary sample chunks, retains
  only one fixed frame buffer plus DSP state, and is byte-for-byte equivalent
  to batch extraction across chunk boundaries. A late invalid chunk may return
  an error after earlier frames have already been emitted; callers that require
  atomic publication must buffer or transact their own output sink.
- `AcousticSegmentationStream` adds the fixed 401-frame change ring and emits
  compact tracklets. Its retained working state is duration-independent;
  returned tracklets are deliberately output-proportional. Normal runtime
  retains only that ring and compact tracklets. The
  public evaluator may opt into a duration-proportional score stream, but its
  existing 256 MiB source-audio cap gives that diagnostic allocation a fixed
  upper bound; it is immediately reduced to aggregate metrics and never enters
  the stable report.
- The native acoustic implementation defaults to at most 512 global prototypes. Anchored prototypes
  are never discarded. Cap pressure is reported.
- Exact constrained agglomeration is bounded by the prototype cap, followed by
  linear-in-turns temporal refinement.
- Stable labels place anchored references first, then unanchored speakers by
  earliest reliable occurrence and a total-order comparison of the compact
  feature vector. Generated `SPEAKER_NN` labels never collide with an opaque
  reference supplied by the caller.
- Identical input, request, implementation, and feature schema produce
  byte-identical typed output in the deterministic implementation.

## 8. Conservative fallback

Insufficient or out-of-calibration evidence remains unknown. Hard-hinted
intervals may remain labeled while all other speech stays unknown. The engine
does not invent speakers merely to satisfy a range minimum or exact count; it
reports `unsatisfied_constraints`, reports `speaker_count_unresolved`, or
returns a hard error according to typed fallback policy.

The v3 probabilistic speaker-count candidate uses five deterministic semantic
views: full evidence, no pitch, no dynamics, no formants, and no channel. It
retains their complete bounded merge-risk curves, combines them with a
symmetrized degree-bounded normalized-affinity eigengap proposal, applies hard
constraint-graph lower bounds, and linearly pools at most 15% caller-prior mass
into the acoustic count distribution before checking the selected count
against effective post-assignment occupancy. Five-view acoustic agreement
linearly attenuates that mix to 7.5% at unanimity. The bounded pool can move
probability but cannot erase acoustically supported counts, acquire the
unbounded leverage of a near-zero log prior, or veto unanimous evidence through
the unresolved-mass threshold. The
public estimate carries ordered concrete count bins plus separate unresolved
mass, entropy, stability, six typed lane summaries, and content-free
calibration/evidence hashes. It also reports content-free resource accounting:
prototype and retained-edge counts, affinity comparisons, estimated peak
algorithm buffers, stability replicates, eigensolver iterations and sparse
matrix-vector terms, and the final residual when available. These values and
the solver's diagonal shift are bound into the evidence/calibration
fingerprints. Lane agreement is only a development score input; the estimate
is explicitly `development_uncertified`, not described as a calibrated
posterior.

Selection requires the concrete MAP action to dominate unresolved mass, at
least three of five feature views to support it, co-association consensus, and
matching supported occupancy. Any failure retains an unresolved estimate and
invokes the typed fixed-safe assignment fallback. No-voice, insufficient
prototype, invalid-affinity, non-convergence, contradictory-constraint, and
resource-limited states remain non-authoritative. Public evidence aggregates
stability for every requested probabilistic run, including runs that fell
back, so fallback cannot inflate the mean by disappearing from its
denominator.

The six-dimensional temporal/lexical heuristic is not an acoustic fallback.
An explicit acoustic request cannot silently invoke external or lexical
behavior. Soft count priors and ranges are rejected by external/neural engines
instead of being silently hardened into backend min/max controls. Every
fallback names its source and reason.

## 9. Scoring

The retained evaluation authority is `diarization-scorer-v4`. Low-level
metric helpers are not sufficient evidence by themselves: a retained verdict
must be produced by `score_diarization_documents` from these exact versioned
documents:

| Document | Schema identity |
|---|---|
| Reference | `diarization-reference-v2` |
| Hypothesis | `diarization-hypothesis-v2` |
| Configuration | `diarization-scorer-config-v2` |
| Result | `diarization-score-result-v2` |
| Corpus manifest | `diarization-corpus-manifest-v1` |
| Leakage audit | `diarization-leakage-audit-v1` |

All time geometry in those documents uses integer milliseconds. Documents
must use canonical interval and identifier order; corpus recording/split keys
are strictly increasing so duplicate entries within one split fail closed.
Unknown fields, unsupported versions, zero-duration calls, out-of-bounds
intervals, missing reference labels, confidence outside `[0, 1]`, confidence
attached to an unknown label, ambiguous overlapping hints, and non-canonical
ordering fail closed with `diarization.scorer.*` reason codes.

The reference and hypothesis identify the same opaque recording and exact
duration. They contain timed speaker turns but deliberately contain no
filesystem path, media URI, transcript text, or free-form provenance field.
Reference v2 may additionally contain aligned word intervals whose `word_id`
is an opaque non-lexical annotation identity. The interval and reference
speaker are sufficient to score speaker attribution; lexical tokens are
forbidden from retained evaluation inputs. Reference turns are labeled;
hypothesis turns may use an absent label for an honest unknown. Hypothesis v2
may carry the complete bounded `speaker-count-estimate-v2`. Assignment
confidence, count uncertainty, and overlap suspicion remain separate
observations.

### 9.1 Frozen metric policy

DER is computed after maximum-overlap one-to-one speaker permutation:

```text
DER = (missed speaker-time + false-alarm speaker-time
       + speaker-confusion time) / reference speaker-time
```

The report keeps all three components separate. Overlapping reference speakers
contribute speaker-time independently. JER is the mean per-reference-speaker
Jaccard error after the same mapping. Overlap F1 is computed directly as
`2 TP / (2 TP + FP + FN)`, so a corpus with reference overlap and no predicted
overlap has defined F1 zero even though precision alone is undefined.

`speaker_boundary_collar_ms` is the half-width removed around each reference
speaker change for DER/JER and associated attribution metrics. It does not
erase the change from the change-point task. `overlap_policy` explicitly
includes or excludes reference-overlap regions; no default hidden in a corpus
adapter may change that policy. Corpus-defined ignored regions are also
removed and contribute to the reported ignored duration.

The same result separately reports:

- union speech miss, false alarm, and speech-activity error rate;
- one-to-one change precision, recall, F1, and mean absolute timing error under
  the named `change_boundary_collar_ms`;
- conditional speaker-attribution accuracy and raw speaker-count error;
- proper multiclass count-posterior Brier score, finite negative log
  likelihood plus an explicit zero-reference-probability flag, concrete
  top-k coverage, a deterministic credible set that may retain unresolved
  mass, entropy, and calibration authority; a reference count outside the
  concrete posterior support still contributes its missing target-class term
  to the Brier score rather than disappearing from the outcome space;
- scored occupancy per anonymized hypothesis label, effective speaker count,
  phantom-label count, dominant-label share, UNKNOWN share, recurrence, and
  per-reference recall-collapse diagnostics;
- transcript-free aligned-word speaker counts and word diarization error rate
  (WDER), with the same ignored-region, collar, and overlap policy as the
  duration score;
- duration-weighted overlap precision, recall, and F1;
- hard/soft hint adherence, contradiction, unknown duration, and hard
  violation duration;
- known-speaker coverage and selective risk, so abstaining everywhere cannot
  look accurate;
- duration-weighted Brier score and fixed-bin expected calibration error,
  always accompanied by confidence coverage;
- wall time, audio duration, real-time factor, and peak RSS, kept separate from
  accuracy.

Reference, hypothesis, configuration, and result hashes use canonical compact
JSON and SHA-256. `result_sha256` hashes the complete result with that one field
temporarily empty, making tampering independently detectable. Repeated scoring
of identical canonical inputs must serialize byte-for-byte identically.

Speaker-count exactness is not a collapse test. A hypothesis can emit exactly
the reference number of label names while assigning nearly all speech to one
label. Scorer v4 therefore independently declares dominant collapse when a
multi-speaker reference crosses the configured labeled-share threshold, and
reference collapse when any mapped reference speaker falls below its
configured attribution recall. Labels below
`minimum_effective_occupancy_ms` do not count as effective or phantom
speakers. UNKNOWN share is measured over hypothesis speaker-time
(`UNKNOWN / (labeled + UNKNOWN)`), so false alarms cannot make it exceed one.
All thresholds are integer millionths or milliseconds in the self-hashed
configuration.

Reference labels that occur only inside excluded scoring regions do not enter
the reference-collapse count or minority-recall diagnostic. Hypothesis labels
seen only there remain visible with zero scored occupancy, but are neither
effective nor phantom speakers.

### 9.2 Corpus manifests and leakage

`diarization-corpus-manifest-v1` is path-free and contains only opaque corpus,
license, recording, source-call, speaker, derivation, augmentation, and
enrollment identities plus `train|development|test`. Identifier validation
rejects path separators, traversal, control characters, media extensions,
transcript markers, and common private-download markers.

The deterministic leakage audit checks every pair of different splits for:

- duplicate recordings;
- clips sharing an origin call;
- known speakers appearing across splits;
- shared derived or mixture ancestors;
- augmentations of the same source;
- shared or cross-linked enrollment recordings.

Findings expose only validated opaque IDs and machine-stable categories. A
passing audit and its self-hash are mandatory inputs to any corpus tuning or
held-out comparison.

Synthetic audio proves arithmetic and invariants only. Corpus accuracy requires
retained real multi-speaker ground truth, license provenance, exact scorer
configuration, and output hashes.

## 10. Evidence and rollout

Internal decision evidence can contain:

- normalized input, hint document, algorithm, feature, weights, calibration,
  parameter, and loss-matrix hashes;
- change-score components and selected boundaries;
- profile quality summaries and anchor use;
- selected-solution clustering merge trace, speaker-count candidates, cap
  pressure, and temporal refinement settings;
- assignment margins, confidence, unknown/overlap decisions, and fallback;
- stage duration, RTF, allocations or peak memory, cancellation, and errors.

Retained public corpus artifacts are stricter: they contain only path-free
aggregate counts, metrics, reliability bins, threshold/collar summaries,
configuration hashes, stage locks, and performance totals. They never retain a
per-recording boundary, score, feature, identifier, filename, transcript, or
audio excerpt.

Focused unit/e2e proof, corpus DER/JER proof, broad Cargo proof, and performance
certification are separate authority states. Rollout follows
Shadow -> Validated -> Fallback -> Primary -> Sole. A hard-hint violation,
privacy regression, non-deterministic replay, contract error, or supported
quality/performance regression triggers rollback.

The runtime meaning of those stages is fail closed:

| Stage | `auto` behavior |
|---|---|
| Shadow | Acoustic output is not exposed |
| Validated | Contract proof may accumulate, but acoustic output is not exposed |
| Fallback | Use verified external speaker evidence when present; otherwise acoustic |
| Primary | Prefer acoustic |
| Sole | Admit only acoustic |

An invalid rollout environment value resolves to Shadow and is reported as an
invalid configuration. An explicit acoustic request is a caller decision and
bypasses only the `auto` admission gate; it does not bypass validation,
resource, privacy, or fallback rules.

## 11. Privacy and corpus handling

Schema v5 may persist turns, hint audit rows, typed speaker-count outcomes,
per-hint dispositions, and privacy-safe profile summaries inside SQLite. It
does not persist raw PCM, frame features, Fourier spectra, cepstra, reusable
speaker vectors, the CLI hint-document source path, or corpus metadata.
`persist_profiles` records explicit consent but does not expand the v5 storage
surface; reusable vectors require a separately reviewed schema and retention
policy.

Private evaluation material must be read in place and must never be copied into
the repository, fixtures, remote build inputs, JSONL snapshots, Beads, logs, or
documentation. Its filenames, transcript text, hashes, durations, and derived
metrics are not admissible public evidence. Only hermetic synthetic fixtures
may be retained here until a redistributable, provenance-cleared corpus is
approved.

### 11.1 Local confidential evaluator

`confidential-diarization-evaluation-manifest-v2` is deliberately different
from the path-free public corpus manifest. It is a local input document that
contains absolute audio, reference, and hypothesis paths and therefore must
remain outside the checkout. Its Rust representation supports deserialization
only: it has no `Debug` or serialization surface.

The `diarization-eval` command:

- discovers the canonical project root from the current checkout;
- requires a canonical absolute input root disjoint from that project;
- resolves every manifest source through symlinks and requires it to remain
  beneath the input root;
- requires a new absolute `.json` output whose canonical parent is outside the
  project;
- caps manifest and scoring-document reads and streams audio hashing with
  bounded memory;
- maps every path, parse, and I/O failure to a stable
  `confidential_evaluation.*` error that never contains a source basename or
  source value;
- checkpoints cancellation before the manifest, every recording, every
  streaming audio-hash block, and the final write, leaving no aggregate when
  cancelled;
- writes only
  `confidential-diarization-evaluation-aggregate-v2`.

The aggregate contains micro/macro accuracy, change, raw count error,
count-posterior proper scores and coverage, occupancy-collapse totals,
transcript-free word-attribution totals, overlap, calibration, and optional
performance summaries plus opaque content/config fingerprints. Posterior
unavailability, unresolved selections, and zero reference probability remain
separate counts. It contains no per-recording row, path, filename, transcript,
timestamp, speaker/recording identity, feature vector, or excerpt. Repeated
evaluation of identical content is byte-stable apart from the caller-chosen
external filename, which is never serialized.

### 11.2 Public and user-licensed corpus adapter

`diarization-corpus registry` emits
`public-diarization-corpus-registry-v1`, the frozen acquisition and conversion
contract. The initial registry spans:

| Key | Conditions | License |
|---|---|---|
| `ami-scenario-v1` | English meetings, close/far microphones, overlap | `CC-BY-4.0` |
| `aishell-4-openslr111-v1` | Mandarin meetings, arrays, noise, short turns, overlap | `CC-BY-SA-4.0` |
| `voxconverse-v1` | in-the-wild backgrounds, same-gender and overlapping speech | `CC-BY-4.0` with original-video copyright notice |
| `callhome-american-english-2e-v1` | dyadic 8 kHz telephone/channel mismatch | operator-held LDC user agreement |

The registry records an authoritative source URL, license URL and exact
acknowledgement ID, expected external layout, conversion contract, integrity
policy, condition tags, and split policy. A registry entry documents what the
operator must acquire; it does not download data, accept a license, or confer
rights. The LDC entry is usable only by an operator who already has lawful
access.

`diarization-corpus build` consumes
`public-diarization-corpus-input-v2` from an absolute root outside the checkout.
Every selected input is a relative path under that canonical root. Symlink
escapes, traversal, absolute descriptor paths, wrong SHA-256 values, unexpected
WAV sample rate/channel count, invalid selected channels, malformed RTTM,
unmapped speakers, and out-of-bounds turns fail closed. RTTM is the deliberately
small interchange surface: exactly ten `SPEAKER` fields, plain decimal seconds,
one selected recording/channel, and an explicit source-label to path-free
speaker-ID map. Concurrent different-speaker turns are preserved and marked as
overlap. Ignored regions remain explicit scorer inputs.

Each recording may bind an optional external
`public-diarization-word-annotation-v1` document by relative path and exact
SHA-256. It contains only the recording identity and canonically ordered
opaque word IDs, integer-millisecond intervals, and reference speaker IDs.
The adapter validates every word against active reference speech, caps
per-recording and corpus totals, and never imports lexical text.

The generated `public-diarization-corpus-bundle-v2` contains the path-free
manifest, canonical reference documents, media/annotation/reference SHA-256
values, optional word-annotation SHA-256 values and counts, checked WAV
geometry, and a passing self-hashed leakage audit. It never contains local
paths, URIs, transcripts, or media bytes. The output is created once in a
directory outside both the checkout and input root; source media is never
copied. The path-bearing descriptor type is deserialization-only and has no
`Debug` or serialization implementation.

The current ablation evidence is
`public-diarization-acoustic-ablation-v8` with runner v8. Every split reports
the full count confusion matrix, exact/error quantiles, reference-count and
duration strata, posterior calibration summaries, collapse/occupancy
diagnostics, and optional micro/macro WDER. These additions do not promote a
candidate: the historical v7 development result in section 4.4 remains the
last retained verdict until a hash-locked v8 development run passes.

The AMI adapter enforces the corpus site's scenario-only training,
development, and unseen-test meeting-family split. Other corpora use an
external descriptor whose exact bytes are frozen by SHA-256, followed by the
same cross-split speaker/origin/derivation/augmentation/enrollment audit. Any
tuning or comparison must name the bundle hash before looking at held-out
results. Changing a source file, mapping, ignored region, or split produces a
different bundle identity.

CI uses only generated WAV/RTTM fixtures to prove adapter arithmetic, malformed
input handling, overlap, ignored regions, channel/sample-rate checking,
license acknowledgement, cancellation, and byte-stable replay. Full accuracy
certification points the same command at externally acquired data and retains
the resulting bundle and scorer outputs outside the repository.

### 11.3 Adversarial and metamorphic acoustic recipes

`src/adversarial_corpus.rs` is the public-safe failure-reproduction substrate.
It generates finite in-memory PCM from `adversarial-synthetic-call-v1` recipes;
it does not read a path or serialize the resulting samples. Synthetic profiles
contain only oscillator, amplitude, stationary coloration, and stereo-position
parameters. Turns contain only a numeric profile index, integer time range,
gain, pitch movement, and a playback condition. They do not contain words,
names, demographic labels, recordings, or biometric templates.

The v1 challenge registry contains one deterministic seed for each required
regime:

| Source or transform family | Metamorphic contract |
|---|---|
| Gain/distance imbalance | Speaker labels remain permutation-equivalent |
| Stationary EQ/muffling and band limitation | Labels remain stable; quality may degrade |
| Resampling and quantization | Timing is unchanged; consistency error is measured |
| Clipping, noise, reverb, and interruptions | Labels remain stable or become explicitly uncertain; no invented identity |
| Leading/trailing silence | Every reference boundary shifts by the exact leading duration |
| Rapid turns and long turns | Source geometry is authoritative |
| Similar pitch and within-speaker voice-state shifts | Pitch alone may neither merge nor split an identity |
| Within-speaker channel shifts and loudspeaker playback | Channel evidence may create a subprofile, not a new voice |
| Stereo channel swap | Speaker output is channel-permutation invariant |
| Controlled overlap | Overlap evidence increases without fabricating a third identity |

Every `adversarial-transform-plan-v1` binds the exact input PCM hash and
contains at most 64 bounded integer-parameter steps. Source authority is either
synthetic or public-licensed; the latter requires a lowercase SHA-256 of an
external acknowledgement or license record, never its path or text. Execution
checkpoints cancellation between steps, rejects non-finite or malformed PCM,
caps allocations, and emits `adversarial-transform-evidence-v1`. Its graph
records only the plan hash, per-step recipe hash, input/output audio hashes,
and expected relationship. It contains no audio, path, filename, transcript,
embedding, speaker name, or per-frame feature.

Pipeline harnesses provide aggregate fingerprints for input, normalization,
speech mask, feature extraction, change detection, clustering, projection,
and scoring. Comparison returns the first differing or missing stage. A stable
regression classification is an uppercase error code plus that stage.
Deterministic delta minimization removes transform subsequences only when the
caller-supplied evaluator reproduces the exact same classification twice;
disagreement fails as a non-deterministic classifier. The result retains
original step indices and an evaluation count. The minimized artifact is a
recipe, not an accuracy certificate.

An identity-preserving recipe does not promise that an imperfect candidate
will pass. Its purpose is to turn violations into small, reproducible public
regressions. Promotion still requires the frozen scorer, public corpus gates,
unseen held-out evidence, and the rollout authority described above.

### 11.4 Stage-aware external differential oracles

`src/differential_oracle.rs` is an explicit developer-only bridge to
operator-installed diarization systems. It does not add a Cargo dependency,
does not run during transcription, and does not make any external system part
of the shipped native decision path. Its stable registry spans three
architecturally different families:

- cascaded pipelines: pyannote and NeMo spectral clustering;
- Bayesian HMM refinement: VBx;
- end-to-end/attractor systems: EEND, DiaPer, and Sortformer.

Each registry entry names a dedicated
`FRANKEN_WHISPER_*_ORACLE_BIN` override and a default adapter executable. The
operator supplies that adapter. A version probe receives
`--franken-whisper-diarization-oracle-version --protocol
franken-whisper-diarization-oracle-protocol-v1` and must emit one strict
`franken-whisper-diarization-oracle-version-v1` JSON object on stdout. A run
receives `--franken-whisper-diarization-oracle-run`, the same protocol flag,
an external `--audio` path, and a lowercase SHA-256 `--recording-key`. It must
emit one strict `franken-whisper-diarization-stage-document-v1` object on
stdout. Arguments are never retained, and neither stdout nor stderr content is
copied into the report.

The canonical stage document has a duration and optional outputs for:

1. speech activity intervals;
2. opaque, non-lexical word timing IDs;
3. speaker-change boundaries;
4. opaque segment-to-cluster assignments;
5. overlap intervals;
6. final diarization turns.

Word identities must use bounded `w-` hexadecimal tokens and cluster segment
identities bounded `seg-` hexadecimal tokens. There is no field for transcript
text, a media path, a model path, a raw embedding vector, or a speaker name.
Intervals and counts are bounded; activity/overlap intervals must be ordered
and non-overlapping; confidences must be finite and in `[0, 1]`.

The comparator reports all six stages in that order. Activity and overlap use
exact integer-millisecond intersection-over-union. Word timing joins opaque
IDs and measures two-boundary collar recall. Change points use the frozen
one-to-one matcher. Cluster comparison uses contingency counts and pairwise
co-assignment, making it label-permutation invariant in linear-logarithmic
time rather than materializing all segment pairs. It also measures shared
segment coverage and requires matching geometry for each shared opaque segment
identity, so an adapter cannot appear equivalent by omitting or moving
anchors. Confidence availability and mean absolute confidence delta are
reported separately, but do not determine equivalence because independently
implemented tools need not calibrate confidence to the same scale. Final turns
use the frozen Hungarian speaker mapping and retain only label-free DER/JER
components. Missing stages remain explicitly missing; they are not converted
into errors or fabricated values. `earliest_divergence` is the first present
stage whose frozen diagnostic threshold is exceeded.

An optional third stage document can help interpret a disagreement. Its only
categories are `reference_favors_native`, `reference_favors_oracle`,
`reference_tied`, and inconclusive/unavailable states. These are diagnostics,
not correctness certificates. Every report hard-codes:

```json
{"authority":"diagnostic_only","native_incorrectness_claim_permitted":false}
```

Missing binaries, nonzero exits, timeouts, incompatible versions, invalid
JSON, invalid geometry, and tool/recording identity mismatches create a clean
`skipped` report with a stable reason and failure stage. Cancellation remains
cancellation and kills the child rather than being misreported as a tool
result. Safe partial provenance is retained when available: tool family,
validated tool/adapter versions, executable hash, version/run stdout hashes,
audio hash, and input-document hashes. Paths, stderr, output content, labels,
word IDs, and local recording identities are not retained. Every report
self-verifies its authority, state invariants, stage ordering, configuration
hash, provenance hashes, and result hash before being written with
create-new semantics outside the checkout.

The CLI exposes only explicit development commands:

```bash
franken_whisper diarization-oracle registry
franken_whisper diarization-oracle run \
  --tool pyannote \
  --audio /absolute/external/audio \
  --native /absolute/external/native-stage.json \
  --reference /absolute/external/reference-stage.json \
  --output /absolute/external/differential-report.json
```

All input documents, media, and output reports must be absolute external
files. A missing `--reference` is valid and produces
`inconclusive_no_reference` for genuine disagreements.

### 11.5 Optional ECAPA model and numerical conformance boundary

`src/ecapa_conformance.rs` freezes the model/evidence contract and
`src/ecapa_inference.rs` implements its bounded safe-Rust ECAPA-TDNN forward
path. Neither module admits neural inference into `auto`, changes the acoustic
default, downloads a model, or parses a framework checkpoint at runtime.
Profile/clustering integration and routing remain separate work.

The source is
[`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/eac27266f68caa806381260bd44ace38b136c76a)
at immutable revision `eac27266f68caa806381260bd44ace38b136c76a`,
under Apache-2.0. The `embedding_model.ckpt` identity is:

- 83,316,686 source bytes;
- SHA-256
  `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2`;
- 231 source entries: 200 inference `f32` tensors and 31 scalar `i64`
  BatchNorm `num_batches_tracked` counters;
- hyperparameter SHA-256
  `ecd11c44202b32edb72709dd1013a16f2f060ebee3438ae8a9f9fecb0666ecd2`;
- training-code revision
  `aa0185408025e80f6c748d2c7af7fa96958c2231`.

The model card says the model was trained on VoxCeleb1 and VoxCeleb2. That is
speaker-verification training over English web video, not calibration for
meeting diarization. Telephone, far-field, playback, muffling, accent,
language, overlap, and domain shifts require held-out validation. An embedding
is acoustic similarity evidence, never a gender, name, or person-identity
claim.

The versioned export protocol is
`franken-whisper-ecapa-export-v1`, implemented by the
`ecapa-tdnn-voxceleb-v1` profile in
`scripts/convert_to_safetensors.py`. Source checkpoint loading is a
development-only, out-of-process trust boundary. The profile accepts only the
checkpoint identity above and requires Python 3.12.12, NumPy 2.2.6, Torch 2.7.1,
and safetensors 0.5.3. It then does the following:

1. requires the exact 83,316,686-byte source hash and 231-entry census;
2. rejects every non-tensor or unexpected dtype, dropping only the 31 named
   `num_batches_tracked` counters;
3. preserves unfused BatchNorm parameters and PyTorch logical row-major layout;
4. materializes contiguous IEEE-754 `f32` values in little-endian order;
5. emits canonical, lexicographically ordered safetensors header and payload
   data with path- and time-free provenance metadata; and
6. parses the complete in-memory byte stream with the official safetensors
   reader, then writes, fsyncs, and rehashes a same-directory temporary file
   before atomically hard-linking that verified inode into an exclusively
   created final path.

Two isolated executions produce the same 83,246,544-byte package, SHA-256
`9276a840c52cdd2e9afb73cd87a38e15749e12bf494d3ca47b5bc162f237cbcc`.
The contained tensor payload is 83,223,808 bytes. Neither source nor exported
weights belong in Git, and the converted artifact is not yet published.
`scripts/fetch_aux_models.sh` therefore pins the immutable source URL, source
hash, conversion command, and output hash without pretending that a download
is available.

The shipped Rust verifier first streams the complete package through a bounded,
cancel-aware exact-size and SHA-256 check. It then passes that same
authenticated owned byte buffer—without reopening the path—to
`native_engine::weights::SafetensorsFile` and `WeightsManifest` to require the
exact 200 names and shapes, require every dtype to be `F32`, and compare the
complete deterministic metadata object. Structural, mapping, dtype, metadata,
truncation, corruption, and cancellation failures report stable `ecapa.*`
reasons without printing paths, tensor contents, or source bytes. There is no
second model-package format or sidecar manifest.

The Rust PCM frontend in this bead is a bounded scalar conformance reference,
not the later production kernel; it accepts at most 16,000 samples (one second).
For exact 16 kHz mono finite PCM in `[-1, 1]`, it uses a 400-sample periodic
Hamming window, 160-sample hop, centered zero padding, 400-point one-sided
squared-magnitude spectrum, 80 SpeechBrain symmetric triangular HTK mel filters
over 0–8 kHz, `amin=1e-10`, 80 dB clipping, and per-utterance feature-mean
subtraction without standard-deviation normalization. The neural boundary
separately rejects normalized feature windows below 51 frames. Resampling and
downmixing must already have occurred at the normalized-audio boundary. Callers
must apply `validate_ecapa_input_format` while sample-rate and channel metadata
are still available; the raw-slice conformance frontend cannot detect
mislabeled 8 kHz or interleaved PCM and never guesses their format. A
production PCM-to-feature kernel and common-pipeline hookup remain integration
gates.

The raw 192-value model output is the golden embedding stage. `EcapaModel`
returns an L2-unit-normalized vector and rejects non-finite, wrong-shaped, or
norm-below-`1e-6` output. Future common-diarizer integration will consume that
normalized representation; it is not routed into production clustering yet.
The public analytic fixture combines 173 Hz and 347 Hz harmonics, a chirp, and
one impulse; it contains no speech or identity evidence.
`franken-whisper-ecapa-golden-v1` binds full-array hashes and selected values
for the two frontend stages, initial TDNN, first SE-Res2 block, multi-feature
aggregation, attentive pooling, and raw embedding.

`franken-whisper-ecapa-full-oracle-v1` is the corresponding transcript-free
seven-tensor safetensors capture. Its exact identity is 2,160,320 bytes and
SHA-256
`2c80806fbf68262ab1e0a1b52af18139f08272b7802fc3b0fd96011192dcf485`.
The payload contains 539,616 `f32` values: 16,160 frontend values and 523,456
neural-stage values. Its deterministic metadata binds the golden-evidence and
contract identities, analytic fixture, model and training-code revisions,
export schema, Python 3.12.12, NumPy 2.2.6, Torch and Torchaudio 2.7.1,
Safetensors 0.5.3, and SpeechBrain 0.5.16. Oracle generation passes explicit
all-valid lengths through SpeechBrain normalization and ECAPA inference, clones
the raw filterbank before sentence normalization, and snapshots every hooked
stage so later in-place operations cannot mutate evidence.

The offline exporter constructs and independently parses both safetensors byte
streams before exclusively creating either output. The Rust oracle verifier
then applies the same bounded, cancel-aware exact-size and SHA-256 check as the
weight verifier, parses that authenticated owned buffer without reopening the
path, and requires the exact names, shapes, `F32` dtypes, metadata, and
per-tensor payload hashes. Neither artifact is vendored or tracked in Git.

The network convolution boundary is distinct from the frontend boundary.
Every ECAPA convolution uses SpeechBrain's same-length reflection padding over
the dilation-expanded effective kernel; it does not use the frontend's centered
zero padding. TDNN order is convolution, ReLU, then evaluation-mode BatchNorm.
Res2Net chunk zero is the identity, chunk one is convolved directly, and each
later chunk is added to the preceding block output before convolution.
Attention is normalized over time independently for every channel, and both
global-context and attentive standard deviations clamp variance at `1e-12`.
Inference accepts 51 through 301 frames (one half-second through three seconds
at the frozen hop) and features with absolute value at most 160. Longer
tracklets must, once this representation is integrated, be deterministically
windowed by the common diarization pipeline; the neural kernel never allocates
or runs in proportion to a complete recording.

The forward path preplans a checked conservative numeric-buffer ceiling before
copying input. At 301 frames, the ECAPA-owned `f32` activation and kernel-band
payload ceiling is 7,944,704 bytes. The plan adds an 8,388,608-byte reserve for
the reviewed FrankenTorch/matrixmultiply packing buffers, yielding a combined
16,333,312-byte ceiling and a 20 MiB default caller limit. Allocator metadata,
stack use, resident weights, and test-only golden captures are explicitly
outside that number. Every in-scope production-forward ECAPA-owned heap `f32`
scratch allocation first acquires a safe RAII logical-byte lease. The lease
fails closed if live buffers would exceed the admitted owned bound and
decrements on every success/error/cancellation drop path. Successful inference
requires the live count to return to zero and then records the observed logical
peak in the versioned trace. The external model test exercises this meter at
both the oracle's 101 frames and the admitted 301-frame maximum; test-only stage
captures remain deliberately unmetered.

Folded evaluation BatchNorm makes the resident model payload 83,070,208 bytes.
The separately named 204,065,488-byte load accounting is exactly the retained
83,246,544-byte package plus that resident payload plus the largest
37,748,736-byte decoded source tensor. It bounds those logical payloads, not
allocator capacity/metadata, JSON maps, names, shapes, reader buffers, stack,
or process RSS. Compute loops and finite-value scans checkpoint in bounded row,
channel, or value chunks. Public library callers can supply the same callback
while loading and inferring; the no-callback convenience still honors the
process Ctrl-C token. The kernel entry point is explicitly FrankenTorch CPU f32
and cannot auto-dispatch to Metal. Timing and maximum-attention-sum diagnostics
are observational, content-redacted, and nondeterministic; the latter is a
low-bandwidth signal-derived aggregate rather than source content. Exact
repeatability is asserted only within the same process/build/host kernel path;
portable cross-backend and cross-host conformance is tolerance-based.

The packing proof was reviewed against FrankenTorch revision
`523aaf827faf538aa541126ee222fcd7af348410`. Diagnostics expose that evidence
identity as `scratch_proof_reviewed_frankentorch_revision`; the field records
the source revision against which the proof was reviewed, not an attestation of
the mutable sibling checkout compiled into the running binary. The repository
intentionally consumes FrankenTorch as a sibling path dependency rather than
pretending Cargo pins that checkout. Changing that checkout or source topology
requires renewing the proof and updating the field. Builds also reject
matrixmultiply's `MATMUL_SGEMM_NC`, `MATMUL_SGEMM_KC`, and
`MATMUL_SGEMM_MC` compile-time overrides; they would otherwise invalidate the
published packing reserve.

Each golden `reference_sha256` hashes the CPU-contiguous C-order tensor in its
declared shape after encoding every value as little-endian IEEE-754 `f32`.
The network-stage shapes are channel-first `[1, channels, time]`; a native
time-major matrix must therefore be logically transposed before comparison.
The hash authenticates the exact SpeechBrain/PyTorch oracle capture and its
layout; it is not an exact-byte requirement for output from a distinct numeric
backend. A supplied full oracle capture must match this hash before its values
can be used for tolerance-based native comparison.

Declared maximum absolute/relative tolerances are respectively `0.05/0.005`
for pre-normalization filterbanks, `0.08/0.005` for normalized filterbanks,
`0.002/0.002` for initial TDNN, SE-Res2, and aggregation, `0.001/0.002` for
pooling, and `0.02/0.002` for the raw embedding. The scalar Rust frontend is
held to a tighter `0.001` absolute error on the frozen selected points.
The authenticated external conformance test compares the complete Rust frontend
arrays with the two oracle frontend tensors. It then feeds the oracle-normalized
filterbank into the native forward path, isolating backend arithmetic while it
compares every value in all five neural-stage tensors. A second full neural pass
feeds the Rust frontend output into the same network and compares all 523,456
neural-stage values again, detecting error amplification across the composed
boundary. Thus all 539,616 reference elements are checked and every neural
value is also checked through composition, together with attention
normalization, output unit norm, fixed-build repeatability, and observed scratch
accounting. The weight package and full oracle remain external test inputs
supplied through
`FRANKEN_WHISPER_ECAPA_TEST_WEIGHTS` and
`FRANKEN_WHISPER_ECAPA_TEST_ORACLE`; ordinary `cargo test` deliberately skips
this large public-artifact proof rather than silently substituting a fixture.
Non-finite values, native shape drift, oracle evidence hash/version/shape
drift, or a tolerance failure are hard conformance failures. They may not be
waived by downstream DER, silently widened, or converted into an
acoustic-engine success. A tolerance change requires a new schema/version,
regenerated public evidence, and an explicit discrepancy record.

### 11.6 Repository and release guard

Audio/video extensions and transcript sidecars are ignored broadly, including
case variants and text/JSON/subtitle forms. Raw decoder spans and transcript-
shaped performance text are also ignored under `tests/artifacts/perf`.

`scripts/check_repository_privacy.rs` is a standalone standard-library-only
gate. It scans tracked or staged path names first. If any prohibited path is
present, it emits only path/reason NDJSON and exits before reading file
contents. Only after the path phase is clean does it inspect magic bytes and
transcript-shaped content in risky artifact roots, so a misleading filename
cannot bypass the policy. It never prints matched content.

Both the automatic tag workflow and the distribution workflow compile this
gate directly with `rustc`. Distribution builds remain allowed to proceed
after advisory test failures, but never after a privacy failure. A known
legacy raw-performance artifact set intentionally keeps this release gate red
until owner-authorized working-tree removal and a separately authorized public
history rewrite are complete.
