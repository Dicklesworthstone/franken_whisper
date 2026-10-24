# Package dependency families before publication

The downstream registry release gate intentionally fails until every reviewed
version is available. That must not prevent testing those versions *before*
publishing them. `scripts/package_registry_dependencies.py` uses Cargo's
multi-package workspace packaging (stable since Rust 1.90) to solve that bootstrap
problem. It does not contact the index to require the unpublished family first,
invoke `cargo publish`, write credentials, or modify an upstream checkout.

## Run

From the Whisper checkout containing this tool, with a clean Torch checkout:

```sh
python3 scripts/package_registry_dependencies.py \
  --family torch --repo /path/to/frankentorch \
  --expected-revision "$(git -C /path/to/frankentorch rev-parse HEAD)" \
  --output /tmp/torch-publication-bundle \
  --without-workspace-overrides
```

For JAX, retain its helper and numerical tests:

```sh
python3 scripts/package_registry_dependencies.py \
  --family jax --repo /path/to/frankenjax \
  --expected-revision "$(git -C /path/to/frankenjax rev-parse HEAD)" \
  --output /tmp/jax-publication-bundle
```

The default command runner is `rch exec -- cargo`. Dedicated qualification hosts
and CI explicitly choose `--cargo-command cargo`; other command strings and
arbitrary Cargo source overrides are rejected. The committed Whisper dated
nightly is copied into each bundle. `--jobs` limits compilation concurrency and
`--timeout` bounds each Cargo command. A missing compiler, failed command,
missing fixture or missing artifact is a failed qualification, not a skip.

The output must be a new directory outside the checkout, without ancestor Cargo
configuration. `--prepare-only` writes inspectable inputs and reports
`prepared_only`, with tests/packaging `not_run`; it does not qualify anything.
Use `--lock /path/to/previous-bundle/source/Cargo.lock` to replay that family's
resolution. Never pass Whisper's release lock or the original full-workspace lock
as a substitute for the reduced dependency family's lock.

## What is built

Torch comprises `frankentorch-core 0.1.1`, `frankentorch-kernel-cpu 0.1.2`, and
`ft-kernel-metal 0.1.1`. JAX comprises `frankenjax-test-utils 0.1.0`,
`frankenjax-core 0.1.0`, and `frankenjax-lax 0.1.0`. The source contract is the same
reviewed contract used by the downstream archive audit. Unknown identities,
versions, source trees, internal paths or incomplete families fail preparation.

The tool reads immutable committed blobs, not mutable worktree copies. Every
tracked file beneath each selected crate is retained, including its original
manifest, tests, fixtures, benchmarks and licenses. Root LICENSE/README files are
retained for relative package metadata. The generated virtual workspace keeps
inherited dependency options, package metadata, lints and profiles, but selects
only the requested family. This avoids pulling unrelated workspace members such
as Python bindings into the publication bootstrap. References to unselected local
crates are errors, not silently deleted dependencies. External fixture references
are not fabricated: Cargo must find all required inputs or fail.

Fresh Cargo home and target directories prevent user configuration, credentials,
workspace caches or the development lockfile from supplying hidden dependencies.
Only the selected family may resolve locally; every external input must resolve
to checksummed crates.io source. All-feature workspace tests run first, including
doctests. Then **one** `cargo package --workspace --all-features --locked` invocation
build-verifies the family. Cargo stages the new internal packages for each other;
running separate per-package commands would reinstate the publication deadlock.
Verification stays enabled throughout.

Each resulting archive is checked against the reviewed source tree, Rust library
name, feature declarations and effective inherited dependency options. Its lock
must refer to the exact newly generated internal archives by checksum, rather
than older same-name registry implementations. The bundle retains the generated
lock, archives, command logs and `dependency-package-receipt.json`, including a
publication order that accounts for dev-dependencies: JAX helper before core
before lax, and Torch core before CPU.

## The Metal patch boundary

Torch's root workspace patches `block` from Git. Cargo does not publish that
workspace patch with `ft-kernel-metal`. Copying it into this gate would test a
different dependency closure from the one consumers receive. Consequently the
tool requires the explicit `--without-workspace-overrides` acknowledgement,
records exactly what was omitted, and removes it **only from the generated
workspace**. No upstream patch or Metal source is altered.

A successful Linux package build exercises the non-Mac stub, not Metal. The
Torch workflow also attempts the unpatched build on Apple Silicon, but even its
ordinary workspace tests can skip unavailable hardware. Neither lane is native
fault/recovery certification. The existing downstream mandatory native Metal
probes still apply. If the unpatched closure fails, repair its publication path;
do not restore the hidden Git patch to make this gate green.

## Publication is still a separate action

`dependency_packages_verified` means local source packages passed this host's
workspace tests and Cargo package verification. `published` is always false and
`registry_availability` remains `not_checked`. The intermediate family lock has
local workspace members; it is not a registry-only Whisper release lock. This
receipt cannot satisfy `registry_candidate_verified` or native Metal release
qualification.

Once the real host evidence is reviewed, an authorized publisher can use Cargo's
multi-package publication from the generated workspace, in the documented order,
with normal verification enabled. Cargo still requires registry authentication;
this tool supplies none. Re-audit the actual uploaded archives and run Whisper's
existing registry-only, packaged-oracle, locked-replay and native-Metal gates
before promoting a release. Do not infer publication from a workflow or bundle.

## Offline regression tests

```sh
python3 -m unittest discover -s scripts -p 'test_registry_dependencies.py' -v
```

These controls use real Git fixtures and simulated Cargo/package output. They
are not Rust compilation, numerical conformance or native GPU execution evidence.
