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
normalized 16 kHz mono PCM
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

Confidence combines best-versus-second assignment margin, profile reliability,
resampling stability, and a named calibration artifact. Calibration reports
Brier score, expected calibration error, and coverage so returning unknown
everywhere cannot appear successful.

## 7. Determinism and resource limits

- Frame cadence is 16 kHz, 400 samples, 160-sample hop for feature schema v1.
- Cancellation is checked at least every 32 frames and during clustering,
  smoothing, projection, persistence, and cleanup.
- Whole-call raw frame matrices and full raw-audio wavelet transforms are
  forbidden.
- Acoustic v1 defaults to at most 512 global prototypes. Anchored prototypes
  are never discarded. Cap pressure is reported.
- Exact constrained agglomeration is bounded by the prototype cap, followed by
  linear-in-turns temporal refinement.
- Stable labels place anchored references first, then unanchored speakers by
  earliest reliable occurrence and stable feature-hash tie-break.
- Identical input, request, implementation, feature schema, and calibration
  artifacts produce identical turn and evidence hashes across thread counts.

## 8. Conservative fallback

Insufficient or out-of-calibration evidence remains unknown. Hard-hinted
intervals may remain labeled while all other speech stays unknown. The engine
does not invent speakers merely to satisfy `min_speakers`; it reports the
unsatisfied constraint or returns a hard error according to typed policy.

The six-dimensional temporal/lexical heuristic is not an acoustic fallback.
An explicit acoustic request cannot silently invoke external or lexical
behavior. Every fallback names its source and reason.

## 9. Scoring

DER is computed after maximum-overlap speaker permutation:

```text
DER = (missed speaker-time + false-alarm speaker-time
       + speaker-confusion time) / reference speaker-time
```

The report keeps all three components separate. Overlapping reference speakers
contribute speaker-time independently. JER is the mean per-reference-speaker
Jaccard error after the same mapping.

Change-point scoring is one-to-one and always names its forgiveness collar.
Confidence calibration is evaluated only where correctness is known and always
reports coverage.

Synthetic audio proves arithmetic and invariants only. Corpus accuracy requires
retained real multi-speaker ground truth, license provenance, exact scorer
configuration, and output hashes.

## 10. Evidence and rollout

Every retained decision artifact records:

- normalized input, hint document, algorithm, feature, weights, calibration,
  parameter, and loss-matrix hashes;
- change-score components and selected boundaries;
- profile quality summaries and anchor use;
- clustering merge trace, speaker-count candidates, cap pressure, and temporal
  refinement settings;
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
