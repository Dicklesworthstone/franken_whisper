# Preserve oracle evidence without permitting input drift

JAX pre-publication run `38494376957` reached and passed its enabled tests:
161 core tests, 1,765 lax tests (360 existing ignored tests), and four helper
tests. Its next step failed with `tests changed the locked source inputs`.
This was not a successful package build: Cargo package had not run.

Three existing lax tests write forensic and case-log JSON beneath the repository's
`artifacts` directory. They use `CARGO_MANIFEST_DIR`, so changing the working
directory cannot redirect those files. The former inventory comparison treated
these new outputs exactly like changes to source, Cargo.lock or golden fixtures.

## Exact input/output separation

`scripts/registry_test_outputs.py` recognizes the six existing JSON output paths
only for `frankenjax-lax 0.1.0` with reviewed source tree
`7aeb83da4ffa461be93aff082643eb659a144867`. The emitters are
`e2e_bitwise_reductions_oracle`, `e2e_special_functions_oracle`, and
`e2e_special_function_error_paths_graceful` in `fj-lax/src/lib.rs`.
The source contract, source-tree identities, tests and expected values are not
changed by this repair. A future source tree does not inherit the exemption.

Before tests, none of the six output paths may exist or overlap an input. After
the unchanged all-feature workspace test command succeeds, the guard requires:

- Every original input, including the lock and repository-level goldens, unchanged.
- Exactly those six new files, with no extra additions anywhere in the workspace.
- Regular, non-executable, unlinked, bounded files containing structured JSON.

This is not an exemption for `artifacts/`, nor permission to update a golden.
Missing outputs, source changes, lock changes, renamed files, added Rust code,
extra JSON files beside the approved files, and symlinks remain failures. The
receipt identifies added, modified, removed and unexpected paths on failure.
All six outputs are copied byte-for-byte to `test-evidence/` outside the source
workspace and recorded with their hashes and sizes. The originals remain in
place. The whole post-test tree, including this evidence, is then frozen for
Cargo package and archive validation. Nothing is deleted to make the guard pass.

The receipt now distinguishes failed test commands from commands that never ran.
When tests pass but input integrity fails, `unit_tests` is `pass`,
`test_input_integrity` is `failed`, and the overall result remains `failed`.
Successful source-package qualification requires both phases and the existing
package/archive checks. Neither JSON evidence alone nor a zero exit from tests
constitutes registry availability, native Metal qualification or publication.

## Regression coverage

```sh
python3 -m unittest discover -s scripts -p 'test_registry_dependencies_outputs.py' -v
```

The controls exercise real filesystem inventories and copies. The pipeline
controls simulate Cargo while running the actual archive/lock validation on
small generated crate fixtures. They prove orchestration and rejection behavior,
not Rust compilation or numerical equivalence to JAX. Existing fixture and
packaging tests remain part of `test_registry_dependencies*.py` discovery.

## Separate Torch failures remain blocking

Torch run `38494323322` failed the existing
`conv2d_backward_masked_skips_the_gemm_the_mask_declines` assertion on both
platforms. Its macOS lane also failed `bidiag_blocked_output_is_bit_stable`.
This repair does not relax or skip either assertion, change kernel routing,
set an artificial CPU count, replace golden values, or claim those failures
are fixed. The workflow must still pass its original Rust command and the
normal package verification before a dependency bundle is qualified.
