# Registry archive identity gate

Issue #17 exposed a case that version resolution cannot detect: two different
implementations using the same package name and version. Run this additional
preflight before admitting a registry-only release:

```sh
python3 scripts/verify_registry_archives.py > registry-archive-audit.json
```

Python 3.11+ is required. A missing package/version, yanked version, network
failure, checksum mismatch, source mismatch, or malformed archive exits nonzero.
There is no fallback to an older version, sibling checkout, or cached source.
Nothing is published and no source baseline is rewritten.

The reviewed input is `scripts/registry-source-contract.json`. The contract
covers all three FrankenTorch packages involved in #17 plus the JAX core, lax,
and test-helper packages, including Metal even when this command runs on Linux. It requests the *new* versions,
not the already-published August implementations. These version declarations
are publication targets, not evidence of publication.

For each entry, the audit checks the exact crates.io index record, downloads
the immutable `.crate` archive, checks the archive's SHA-256 against the index,
and reconstructs the Git tree ID of every file under `src/`. Paths, executable
bits, and file bytes participate in that tree ID. This catches both missing
modules and silent same-version edits. SHA-1 is used only to compare with Git
object IDs; archive transport integrity uses the registry's SHA-256 checksum.

The Torch baseline tree IDs were read from commit
`39b90cf7a90d7ecb8ae6f1637a6e287855182aeb`. Reproduce them in a source checkout:

```sh
git rev-parse 39b90cf7a90d7ecb8ae6f1637a6e287855182aeb:crates/ft-core/src
git rev-parse 39b90cf7a90d7ecb8ae6f1637a6e287855182aeb:crates/ft-kernel-cpu/src
git rev-parse 39b90cf7a90d7ecb8ae6f1637a6e287855182aeb:crates/ft-kernel-metal/src
```

A future intentional implementation change needs an explicitly reviewed new
version and contract update, not an automatic baseline refresh to make a gate
pass. `extra_files` maps any additional build input (for example `build.rs`)
to its reviewed Git blob ID. Archives are inspected in memory, never extracted;
path traversal, symlinks, duplicate members and resource-limit violations fail.

## What this does not certify

The JSON receipt deliberately reports `build_and_conformance: not_run`, even
when artifact integrity passes. This is a source-tree identity check for the
listed packages, not a complete dependency-closure or behavioral proof. It
does not replace Cargo's all-feature/all-target resolution, a registry-only
lockfile, package verification, the existing release gate, conformance fixtures,
or an Apple Silicon Metal failure/recovery run. Build-script inputs outside
`src/` require explicit review; arbitrary compile-time external resources and
all manifest semantics are not automatically inferred by this audit.

The JAX source baselines and publication-preparation revisions are documented in
`REGISTRY_RELEASE_CANDIDATES.md`. The integrated candidate gate runs this archive
audit before isolated Cargo resolution, tests, and package verification.

The workspace-only `block` Git patch must not leak into a claimed registry-only
result. Qualify an extracted package outside the FrankenTorch workspace. A
Linux Metal stub run cannot validate the real GPU path. Keep `fj-oracle`
functional and run its conformance tests. Contract entries describe required
implementations; they do not claim the JAX publications are already available.

Run the offline regression tests with:

```sh
python3 -m unittest discover -s scripts -p test_verify_registry_archives.py -v
```

The tree-hash test uses Git itself as an independent reference, rather than
having the implementation generate its own expected answer.
