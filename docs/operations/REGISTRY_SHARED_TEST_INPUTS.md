# Shared golden inputs in dependency qualification

The first real pre-publication builds reached Rust compilation, then failed
because the reduced workspace omitted golden data referenced by existing tests.
Torch run `38485433914` failed on both Linux and Apple Silicon at four
`include_str!` references in `ft-kernel-cpu`. JAX run `38485493208` failed at
one in `fj-core`. The Torch core source also references its repository-level
`ft_core_numel_pass22.txt` fixture, which must survive the same reduction.
These were missing-input failures, not numerical test failures or publication
failures. Their original artifacts contain the compiler diagnostics.

## Preserve the existing assertions and expected outputs

The source contract now optionally gives a package `workspace_test_files`:

```json
{
  "workspace_test_files": {
    "artifacts/optimization/golden_outputs/ft_core_numel_pass22.txt":
      "0d1aea81f51de3880dd0d4d15a4cab871aee7505"
  }
}
```

Each key is the exact repository-relative data path; each value is its reviewed
Git blob ID. These are the original files, not regenerated expectations. The
packager stages them from the requested commit at their original paths, where
Rust's relative `include_str!` references find them. It verifies each blob and
requires every listed file to exist. Conflicting pins, unsafe paths, glob
pathspecs, symlinks, executable inputs, directories and changed goldens fail
preparation. Untracked files cannot repair a missing committed input.

Only selected family declarations are used. An entire artifacts directory is
never copied. The receipt records `workspace_test_inputs` with Git and SHA-256
identities, and the existing source-inventory checks also cover these bytes
through tests and packaging. The original checkout, numerical source, library
identities, versions, source-tree contract, assertions, features and tolerances
are unchanged. No test is removed, ignored or weakened by this repair.

## Scope

This data closes the **source workspace test** inputs. It is not a path or Git
dependency, does not enter Cargo's dependency resolver, and does not assert that
repository-level fixtures are embedded in each published crate. Packaged source
and registry-only consumer validation remain separate gates. In particular,
reproducing upstream unit tests from an individual `.crate` still needs the
external fixtures until their upstream layout is made package-local. Do not
claim those packaged upstream tests ran from the family workspace test result.

The existing one-invocation Cargo package verification stays enabled, and the
strict downstream Whisper oracle and native Metal probes are still required.
A successful fixture preparation is not a successful Rust test run and is not
evidence of crates.io publication. Retain new CI results before qualifying the
dependency family.

```sh
python3 -m unittest discover -s scripts -p 'test_registry_dependencies*.py' -v
```

The added controls exercise real Git repositories and filesystem staging, not a
Rust compiler. They include the previously missing external-fixture layout.
