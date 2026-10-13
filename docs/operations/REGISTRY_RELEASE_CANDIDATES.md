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

The gate first copies only tracked, committed inputs. It excludes the development
`Cargo.lock`, `.cargo` overrides, `.git`, and `target`; it never copies siblings.
It rejects dirty tracked files, tracked symlinks/submodules, a destination inside
the checkout, existing output directories, and inherited ancestor Cargo configs.
Cargo receives a fresh Cargo home/target directory with inherited Cargo source
overrides and Rust wrappers removed. The original checkout remains untouched.

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
identity collisions stop the pipeline before Cargo runs.

Cargo then resolves **all features and all platforms**. Metadata and every lock
entry must identify crates.io dependencies, with exactly one local package: the
release root. The five critical runtime packages must have precisely the reviewed
versions, not merely a compatible range. Default and all-feature library/integration
tests run with `--locked`. A separate `--features fj-oracle --test
conformance_oracle_tests` invocation requires that specific conformance target to
exist and pass; merely defining an unused feature is not sufficient. Finally,
`cargo package --locked --all-features` runs with verification enabled. The actual
`.crate` is inspected for registry-only dependencies, the preserved oracle feature,
and a registry-only packaged lockfile. The gate checks that validation did not
change the candidate lockfile.

The receipt records the source revision, generated-manifest hash, candidate and
packaged lock hashes, archive hash, command list, and phase outcomes. Stdout and
stderr logs are retained per command. **No command publishes a crate.**

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
builds/tests/packaging. Preserve the **packaged** lockfile with the released crate;
Cargo may normalize a package lock separately from the working candidate lock.
Byte-for-byte binary reproducibility additionally needs fixed toolchain, platform,
SDK/system-library, and build-environment inputs; this gate does not certify that.

## Preparation is not readiness

An offline inspection is available explicitly:

```sh
python3 scripts/prepare_registry_release.py \
  --output /tmp/franken-whisper-registry-inspection --prepare-only
```

This returns `status: prepared_only`, with registry, tests, and packaging marked
`not_run`. It must not authorize a release. The ordinary development manifest,
lockfile, and `prepare_release_siblings.sh` are intentionally unchanged in this
transition: they remain sibling-based and are **not** advertised as registry-only.
Use the isolated candidate workflow for registry qualification. Do not replace the
development inputs or promote a registry lock until actual publication and Cargo
qualification have succeeded. The isolated rewrite is an additional release path,
not a claim that every existing release workflow now uses it automatically.

Even a successful `registry_candidate_verified` receipt does **not** certify native
Metal fault/recovery behavior. The receipt always makes this explicit. Run the
real Metal tests on Apple Silicon against the published package, without relying
on the Torch workspace's Git-only `block` patch. A Linux stub cannot establish
that. Existing broader release gates, smoke tests, native-library qualification,
and release-version/tag consistency checks still apply.

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
