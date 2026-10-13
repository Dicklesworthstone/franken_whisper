# Isolated registry release candidates

Issue #17 cannot be fixed by substituting the older published kernels for the
reviewed source pins. It also cannot be fixed by dropping `fj-oracle`. The
registry candidate workflow preserves the numerical code and feature graph,
changes only dependency locations/package identities/version requirements,
and refuses to qualify an unavailable or different registry implementation.

## Run the gate

From a clean committed checkout, use a **new directory outside the checkout**:

```sh
python3 scripts/prepare_registry_release.py \
  --output /tmp/franken-whisper-registry-candidate
```

The default Cargo command is `rch exec -- cargo`. The runner must return its
metadata output and package artifact to the candidate output directory. A
missing runner, missing artifact, failed build, failed test, or unavailable
registry dependency is a failure, not evidence of readiness. An explicitly
selected qualification environment can set `--cargo-command`; the argument is
split into argv without shell evaluation. Preserve the actual runner/toolchain
logs with the receipt.

The gate reads immutable, tracked Git blobs, not mutable worktree files. It
excludes the development `Cargo.lock`, `.git`, and `target`; it never copies
siblings. It rejects dirty tracked files, tracked symlinks/submodules, a
destination inside the checkout, existing output directories, and inherited
ancestor Cargo configs. Hidden worktree changes and checkout filters cannot
change the reported committed inputs.

The reviewed `.cargo/config.toml` is preserved byte-for-byte: its architecture-
scoped CPU baseline and debug test-stack setting affect actual build/test
behavior. Unknown Cargo configuration, source overrides, and aliases are
rejected, not silently stripped. Changes to those build semantics require
reviewing the allowlist in the gate. Cargo receives a fresh Cargo home/target
with inherited Cargo source overrides, Rust wrappers, and test-stack overrides
removed. The original checkout remains untouched.

The generated candidate `source/Cargo.toml` has these exact registry requirements:

| Rust dependency alias | Published package required | Exact version |
| --- | --- | --- |
| `ft-core` | `frankentorch-core` | `=0.1.1` |
| `ft-kernel-cpu` | `frankentorch-kernel-cpu` | `=0.1.2` |
| `ft-kernel-metal` | `ft-kernel-metal` | `=0.1.1` |
| `fj-core` | `frankenjax-core` | `=0.1.0` |
| `fj-lax` | `frankenjax-lax` | `=0.1.0` |

All sibling paths, including the other three repository families and dev/target
sections, become registry dependencies. Other version requirements are retained
and then fixed by the candidate's registry lockfile. The manifest rewrite checks
full parsed-TOML equality against the expected dependency-only transformation:
features, optionality, external feature flags, profiles, targets, and package
metadata cannot silently change. An unsupported manifest layout fails closed.

The source contract covers the three Torch packages plus the two JAX runtime
packages and their independently published test helper. Torch source identities
remain those at `39b90cf7a90d7ecb8ae6f1637a6e287855182aeb`, including the Metal
command-buffer fixes. Their version-preparation commit is
`2c13bcd18a31f1e917a19d478adc182ef8c0ba99`.
The JAX source identities remain those at
`4f04dd82ace83f609eedd232839244ce872bb4ce`; namespace preparation is
`1816ae56b4ff251c8644e9250176a30585650196`. This JAX source is newer than
Whisper's development sibling pin. Equivalence to the old oracle is **not**
assumed: the all-feature tests must pass against the candidate before it qualifies.
The contract's `publication_revision` is provenance for prepared source, **not**
a statement that publication has happened or that the registry archive was built
at precisely that revision. Source tree identity is checked independently.

## What must pass

The existing archive audit runs first, including Metal on Linux and the JAX
helper even though dependency dev-dependencies do not flow into Whisper's runtime
lockfile. Missing/yanked versions, source drift, checksum failures, and source
identity collisions stop the pipeline before Cargo runs. The candidate and
packaged lockfiles must use the same checksums as those audited archives; a
correctly shaped but unrelated checksum does not pass.

Cargo resolves **all features and all platforms**. Metadata and every lock entry
must identify crates.io dependencies, with exactly one local package: the release
root. The five critical runtime packages must have precisely the reviewed
versions, not merely a compatible range. Default and all-feature library/integration
tests run with `--locked`. The explicit `fj-oracle` target also runs. Normal
`cargo package --locked --all-features` verification stays enabled.

The actual `.crate` must contain the tested code, model inputs, committed
build/toolchain configuration, and the existing model-free oracle test. Other
packaged files must also match the candidate, except Cargo's generated manifest,
lock and VCS information. The original manifest must be present and unchanged;
normalization must preserve features, effective dependency options and profiles.

The gate then extracts that actual archive into `packaged-source`, with a
**second empty Cargo home and build directory**, separate from staging and
Cargo's internal verification directory. It fetches under the packaged lock and
requires frozen release builds of both default and all-feature library/binary
targets. The packaged `conformance_oracle_tests` target runs in the release
profile. Its libtest JSON must show a nonempty suite, all tests passed, and no
ignored, filtered, failed or measured tests. Exit zero with no executed tests is
not conformance evidence. Source, lock and archive changes during validation
are rejected.

A successful receipt requires both `status: registry_candidate_verified` and
`packaged_artifact_validation: pass`. It records the source revision, generated
manifest/build-config hashes, candidate and packaged lock hashes, archive hash,
executed oracle count, command working directories, and phase outcomes. Stdout
and stderr logs are retained per command. **No command publishes a crate.**

## Replay, rather than resolve again

The first candidate deliberately resolves a new registry lockfile; it does not
pretend that a development path lock is already a registry lock. Retain that
candidate and replay its exact lock in a fresh directory:

```sh
python3 scripts/prepare_registry_release.py \
  --output /tmp/franken-whisper-registry-replay \
  --registry-lock /tmp/franken-whisper-registry-candidate/source/Cargo.lock
```

Replay validates the supplied lock and uses `--locked` for metadata as well as
staged builds/tests/packaging. Extracted-artifact validation uses `--frozen` after
fetching. Preserve the **packaged** lockfile with the released crate; Cargo may
normalize a package lock separately from the working candidate lock.
Byte-for-byte binary reproducibility additionally needs fixed toolchain, platform,
SDK/system-library, and build-environment inputs; this gate does not certify that.

For release automation, `--expected-revision <full-commit-sha>` rejects a different
source checkout. `--release-tag v<version>` checks the committed package version
without modifying it. A tag naming 0.11.0 cannot qualify a 0.10.0 manifest.
Both checks happen before Cargo runs, including in preparation-only mode.

## Automatic Linux and Apple Silicon qualification

`.github/workflows/registry-release.yml` runs on relevant `main` pushes, `v*`
tag pushes, and manual dispatch. Its preflight runs offline regression tests,
checks source/tag identity, and audits the actual published dependency archives.
Unavailable or mismatched packages **fail** the workflow before expensive Rust
jobs; they are never converted to skipped-success or older-version fallbacks.

Linux then runs the full gate on `ubuntu-24.04`. Only a successful receipt with
matching source, manifest, lock and crate hashes can export the all-platform
registry lock. The `macos-15` Apple Silicon job downloads that exact lock,
verifies its digest, and replays the full gate with the same committed Rust
toolchain. It must not resolve a different dependency closure. These dedicated
qualification jobs explicitly select standalone Cargo, rather than relying on
an RCH dispatcher or sibling checkouts on a developer machine.

The final `Registry release qualification result` check fails if any prerequisite
job failed, was cancelled, or was skipped. Each job retains available receipts,
logs, host information, lockfiles and package archives; Cargo homes, credentials
and build caches are not uploaded. Evidence has a 30-day Actions retention window;
retain successful release evidence durably with the actual release.

This workflow has read-only repository permissions, no registry publication
credentials, and no publish step. It does not itself change branch protections,
block a separate manual publisher, or certify native Metal fault/recovery tests.
A publisher must require its successful result and the remaining native-platform
release evidence before uploading. After dependency publication, manual dispatch
can retry qualification without changing the reviewed dependency contract.

## Preparation is not readiness

An offline inspection is available explicitly:

```sh
python3 scripts/prepare_registry_release.py \
  --output /tmp/franken-whisper-registry-inspection --prepare-only
```

This returns `status: prepared_only`, with registry, tests, and packaging marked
`not_run`. It must not authorize a release. The ordinary development dependency
manifest, lockfile, and `prepare_release_siblings.sh` remain sibling-based and
are **not** advertised as registry-only. The package inclusion list now retains
the existing oracle test. Do not replace development dependencies or promote a
registry lock until actual publication and Cargo qualification have succeeded.

Even a successful registry qualification on an ARM Mac does **not** certify
native Metal fault/recovery behavior. The receipt always makes this explicit.
Run the real Metal tests on Apple Silicon against the published package, without
relying on the Torch workspace's Git-only `block` patch. A Linux stub or a host
architecture label cannot establish that. Existing broader release gates, smoke
tests, and native-library qualification still apply.

JAX publication order is `frankenjax-test-utils`, then `frankenjax-core`, then
`frankenjax-lax`, with package verification and existing conformance assertions
preserved. Torch core must be published before the CPU package. No older package
is treated as an equivalent fallback. Source preparations in GitHub are not
crates.io releases; leave #17 open until the registry and platform evidence exists.

Offline regression tests (the pipeline command tests use a fake Cargo process,
not a Rust compiler):

```sh
python3 -m unittest discover -s scripts -p 'test_*registry*.py' -v
```
