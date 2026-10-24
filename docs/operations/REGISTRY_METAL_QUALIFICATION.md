# Native Metal registry qualification

The registry release workflow now requires native Metal probes after its Apple
Silicon packaged-crate replay. An ARM host label or successful CPU fallback no
longer suffices for the workflow's final qualification result. Existing registry,
package and fj-oracle gates remain prerequisites and are not weakened.

From an Apple Silicon qualification host with a working Metal device, first
complete `prepare_registry_release.py`, then run:

```sh
python3 scripts/qualify_registry_metal.py \
  --candidate /tmp/franken-whisper-registry-candidate \
  --output /tmp/franken-whisper-native-metal \
  --expected-revision "$(git rev-parse HEAD)"
```

The output directory must be new and outside the checkout and candidate. This
command uses local Cargo, never an RCH worker that might be a different platform.
It reads the candidate's successful receipt and original `.crate`, binds their
source revision, manifest, archive and packaged-lock hashes, and checks the two
packaged probe sources against immutable Git blobs. The original candidate is
not modified. It freshly audits the registry archives, requires exact critical
lock checksums, rejects path/Git substitutions in metadata, and confirms the
root's actual Metal dependency edge is registry-backed.

The archive is reconstructed in a fresh directory, retaining the committed Rust
toolchain and reviewed build configuration. A fresh Cargo home is used. After a
locked fetch, metadata and all tests run frozen with the packaged lock. Source
and archive changes during any command fail qualification.

## Distinct production and diagnostic builds

`tests/registry_metal_production.rs` requires a usable GPU and an inactive fault
seam. It compares native GEMM to an independent f64 reference on exact dyadic
inputs, including edge tiles and a Whisper-like shape. This runs with ordinary
release dependency settings.

`tests/registry_metal_faults.rs` runs in a **separate target directory** with only
`profile.release.package.ft-kernel-metal.debug-assertions=true` added. It tests
transient retries, exact output preservation on exhaustion, outage-circuit
recovery, queue-slot release after repeated failures, thread-local injection,
and fused-encoder retry bit identity. This diagnostic build is not a shipping
artifact and is never substituted for the production build.

Both targets are ignored by ordinary test runs because their hardware/profile
requirements differ. The native gate explicitly runs them with `--ignored`,
serializes tests, and parses libtest JSON. Every required named test must start
and pass exactly once; zero tests, skipped/filtered tests, duplicate events,
nonzero commands, and missing terminal results all fail. Missing hardware or an
inactive diagnostic seam fails inside the Rust tests rather than returning early.

## Evidence and limits

A successful `native-metal-receipt.json` records `native_metal_verified`, the
source revision, crate and packaged-lock hashes, audited dependencies, command
arguments and the 1 production / 5 fault test results. The workflow retains the
receipt and command logs even when validation fails. The earlier general-purpose
registry receipt still does not claim native Metal certification; the separate
native receipt supplies that evidence and the macOS job requires its command.

Fault injection intentionally leaves a command buffer **unsubmitted**. It tests
the existing command-buffer error/retry boundary and recovery behavior; it does
not reproduce an OS GPU reset, partial execution, device loss, or every hardware
fault. Broader physical-device qualification remains separate. Neither this gate
nor its workflow publishes anything, rewrites source contracts, or silently uses
the Torch workspace's Git-only `block` patch. Registry dependencies must actually
be published and pass the unpatched native build before qualification succeeds.

Offline Python regression tests simulate Cargo/Metal output and exercise real
Git snapshots; they are not Rust compilation or GPU execution evidence:

```sh
python3 -m unittest discover -s scripts -p 'test_registry_metal.py' -v
```
