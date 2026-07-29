# Acoustic Diarization Contract v1

Status: implementation contract for `bd-odj7`
Contract identifier: `acoustic-diarization-v1`

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
- exact/minimum/maximum speaker constraints;
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

## 4. Acoustic evidence

The feature schema maintains two views:

- **Voice:** non-globally-normalized cepstral envelope, deltas, fundamental
  frequency with voicing confidence, harmonicity, voiced fraction, and robust
  vocal-tract summaries.
- **Channel:** RMS and dynamics, spectral centroid/bandwidth/rolloff/flatness,
  spectral tilt, band ratios, crest/clipping, noise floor, and conservative
  temporal-smearing or distortion summaries.

Voice evidence is primary. Channel evidence is capped and may create multiple
channel subprofiles beneath one voice profile. Pitch is never sufficient by
itself and must never be translated into a gender label.

Wavelet evidence is limited to inexpensive Haar-like multiscale contrasts over
feature trajectories. A full raw-waveform CWT is not the default algorithm.

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

Acoustic v1 accepts at most 1,024 known intervals per request, 256 bytes per
speaker reference, and 4,096 bytes per provenance value. These limits are
validated before the hard-hint overlap check.

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

### 6.1 Canonical DTW projection timeline (`bd-2noc.1`, `bd-2noc.2`)

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

Acoustic v1 confidence currently combines best-versus-second assignment margin
with profile reliability and reports `heuristic_uncalibrated`. Resampling
stability and a named corpus calibration artifact remain promotion gates, not
inputs the current implementation pretends to possess. The retained scoring
surface reports Brier score, expected calibration error, and coverage when
ground-truth observations are available, so returning unknown everywhere
cannot appear successful.

## 7. Determinism and resource limits

- Frame cadence is 16 kHz, 400 samples, 160-sample hop for feature schema v1.
- Cancellation is checked at least every 32 frames and within clustering and
  smoothing. Projection is bounded linear work inside the independently
  budgeted Diarize stage; persistence and cleanup retain their existing
  pipeline cancellation boundaries.
- Whole-call raw frame matrices and full raw-audio wavelet transforms are
  forbidden.
- Acoustic v1 defaults to at most 512 global prototypes. Anchored prototypes
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
does not invent speakers merely to satisfy `min_speakers`; it reports the
unsatisfied constraint or returns a hard error according to typed policy.

The six-dimensional temporal/lexical heuristic is not an acoustic fallback.
An explicit acoustic request cannot silently invoke external or lexical
behavior. Every fallback names its source and reason.

## 9. Scoring

The retained evaluation authority is `diarization-scorer-v1`. Low-level
metric helpers are not sufficient evidence by themselves: a retained verdict
must be produced by `score_diarization_documents` from these exact versioned
documents:

| Document | Schema identity |
|---|---|
| Reference | `diarization-reference-v1` |
| Hypothesis | `diarization-hypothesis-v1` |
| Configuration | `diarization-scorer-config-v1` |
| Result | `diarization-score-result-v1` |
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
duration. They contain timed speaker turns only: there is deliberately no
filesystem path, media URI, transcript text, or free-form provenance field.
Reference turns are labeled; hypothesis turns may use an absent label for an
honest unknown. Hypothesis assignment confidence and overlap suspicion remain
separate observations.

### 9.1 Frozen metric policy

DER is computed after maximum-overlap one-to-one speaker permutation:

```text
DER = (missed speaker-time + false-alarm speaker-time
       + speaker-confusion time) / reference speaker-time
```

The report keeps all three components separate. Overlapping reference speakers
contribute speaker-time independently. JER is the mean per-reference-speaker
Jaccard error after the same mapping.

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
- conditional speaker-attribution accuracy and speaker-count error;
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

Every retained decision artifact records:

- normalized input, hint document, algorithm, feature, weights, calibration,
  parameter, and loss-matrix hashes;
- change-score components and selected boundaries;
- profile quality summaries and anchor use;
- selected-solution clustering merge trace, speaker-count candidates, cap
  pressure, and temporal refinement settings;
- assignment margins, confidence, unknown/overlap decisions, and fallback;
- stage duration, RTF, allocations or peak memory, cancellation, and errors.

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

Schema v4 may persist turns, hint audit rows, and privacy-safe profile summaries
inside SQLite. It does not persist raw PCM, frame features, Fourier spectra,
cepstra, reusable speaker vectors, the CLI hint-document source path, or corpus
metadata.
`persist_profiles` records explicit consent but does not expand the v4 storage
surface; reusable vectors require a separately reviewed schema and retention
policy.

Private evaluation material must be read in place and must never be copied into
the repository, fixtures, remote build inputs, JSONL snapshots, Beads, logs, or
documentation. Its filenames, transcript text, hashes, durations, and derived
metrics are not admissible public evidence. Only hermetic synthetic fixtures
may be retained here until a redistributable, provenance-cleared corpus is
approved.
