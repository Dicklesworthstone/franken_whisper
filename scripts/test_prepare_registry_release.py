#!/usr/bin/env python3
"""Offline controls for registry release staging and fail-closed validation."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import prepare_registry_release as release
import verify_registry_archives as archives

VERSIONS = {"frankentorch-core": "0.1.1", "frankentorch-kernel-cpu": "0.1.2",
            "ft-kernel-metal": "0.1.1", "frankenjax-core": "0.1.0", "frankenjax-lax": "0.1.0",
            "frankenjax-test-utils": "0.1.0"}
SPECS = [dict(name=n, version=v, repository="https://github.com/example/source",
              source_revision="a" * 40, src_tree="b" * 40) for n, v in VERSIONS.items()]
MANIFEST = '''[package]
name = "franken_whisper"
version = "0.10.0"
[features]
default = []
fj-oracle = ["dep:fj-lax", "dep:fj-core"]
tui = ["dep:ftui"]
[dependencies]
ft-core = { version = "0.1.0", path = "../torch/core", package = "frankentorch-core" }
ft-kernel-cpu = { version = "0.1.0", path = "../torch/cpu", package = "frankentorch-kernel-cpu" }
fj-core = { version = "0.1", path = "../jax/core", optional = true }
fj-lax = { version = "0.1", path = "../jax/lax", optional = true }
ftui = { version = "0.9.0", path = "../tui", optional = true, default-features = false, features = ["crossterm"] }
serde = "1"
[dev-dependencies]
ft-core = { version = "0.1.0", path = "../torch/core", package = "frankentorch-core" }
[target.'cfg(target_os = "macos")'.dependencies]
ft-kernel-metal = { version = "0.1.0", path = "../torch/metal" }
[profile.release]
lto = true
'''
ROOT = dict(name="franken_whisper", version="0.10.0")
CONFIG = '''# Reviewed release build settings.
[target.'cfg(target_arch = "x86_64")']
rustflags = ["-Z", "threads=4", "-C", "target-cpu=x86-64-v3"]
[env]
RUST_MIN_STACK = "67108864"
[build]
rustflags = ["-Z", "threads=4"]
'''


def lock_bytes(*, missing=None, changed=None, bad_source=None, bad_checksum=None):
    result = ['version = 4', '[[package]]', 'name = "franken_whisper"', 'version = "0.10.0"']
    for name, version in VERSIONS.items():
        if name in (missing, "frankenjax-test-utils"):
            continue
        source = "git+https://github.com/example/fork" if name == bad_source else release.REGISTRY
        result.extend(['[[package]]', f'name = "{name}"', f'version = "{changed if changed and name == "frankentorch-core" else version}"',
                       f'source = "{source}"', f'checksum = "{"invalid" if name == bad_checksum else "c" * 64}"'])
    return ("\n".join(result) + "\n").encode()


def oracle_events(passed=1, ignored=0):
    events = [dict(type="suite", event="started", test_count=passed + ignored)]
    events.extend(dict(type="test", event="ok", name=f"oracle_{i}") for i in range(passed))
    events.append(dict(type="suite", event="ok", passed=passed, failed=0, ignored=ignored,
                       measured=0, filtered_out=0))
    return b"\n".join(json.dumps(e).encode() for e in events)


class ManifestTests(unittest.TestCase):
    def test_all_aliases_receive_exact_reviewed_versions(self):
        actual = tomllib.loads(release.registry_manifest(MANIFEST, SPECS))
        for section in release.dependencies(actual):
            for alias, dep in section.items():
                if alias in release.ALIASES:
                    self.assertEqual(dep["package"], release.ALIASES[alias])
                    self.assertEqual(dep["version"], "=" + VERSIONS[dep["package"]])
        archives.registry_dependencies(actual)

    def test_features_profiles_optional_flags_and_external_requirements_survive(self):
        original = tomllib.loads(MANIFEST)
        actual = tomllib.loads(release.registry_manifest(MANIFEST, SPECS))
        self.assertEqual(actual["features"], original["features"])
        self.assertEqual(actual["profile"], original["profile"])
        self.assertEqual(actual["dependencies"]["serde"], "1")
        self.assertTrue(actual["dependencies"]["fj-core"]["optional"])
        tui = dict(original["dependencies"]["ftui"])
        del tui["path"]
        self.assertEqual(actual["dependencies"]["ftui"], tui)

    def test_metal_not_skipped_on_linux(self):
        actual = tomllib.loads(release.registry_manifest(MANIFEST, SPECS))
        metal = actual["target"]['cfg(target_os = "macos")']["dependencies"]["ft-kernel-metal"]
        self.assertEqual(metal["version"], "=0.1.1")
        self.assertNotIn("path", metal)

    def test_disabling_oracle_is_rejected(self):
        with self.assertRaisesRegex(archives.AuditError, "retain both"):
            release.registry_manifest(MANIFEST.replace('"dep:fj-core"', '"dep:ftui"'), SPECS)

    def test_source_overrides_and_path_only_versions_are_rejected(self):
        for entry in ('new = { path = "../missing" }\n',
                      'new = { version = "1", git = "https://github.com/other/source" }\n',
                      'new = { version = "1", registry = "private" }\n'):
            with self.subTest(entry=entry), self.assertRaises(archives.AuditError):
                release.registry_manifest(MANIFEST.replace('[dependencies]\n', '[dependencies]\n' + entry), SPECS)

    def test_workspace_patch_cannot_leak(self):
        with self.assertRaisesRegex(archives.AuditError, "overrides"):
            release.registry_manifest(MANIFEST + '\n[patch.crates-io]\nblock = { git = "https://github.com/example/block" }\n', SPECS)

    def test_missing_contract_package_is_rejected(self):
        with self.assertRaisesRegex(archives.AuditError, "contract omits"):
            release.registry_manifest(MANIFEST, SPECS[1:])

    def test_unrecognized_multiline_layout_fails_instead_of_partial_rewrite(self):
        text = MANIFEST.replace('ftui = { version = "0.9.0", path = "../tui", optional = true, default-features = false, features = ["crossterm"] }',
                                '[dependencies.ftui]\nversion = "0.9.0"\npath = "../tui"\noptional = true')
        with self.assertRaisesRegex(archives.AuditError, "layout"):
            release.registry_manifest(text, SPECS)

    def test_double_pass_is_idempotent(self):
        once = release.registry_manifest(MANIFEST, SPECS)
        self.assertEqual(release.registry_manifest(once, SPECS), once)


class ClosureTests(unittest.TestCase):
    def test_valid_registry_closure_does_not_require_dependency_dev_helper(self):
        release.check_lock(lock_bytes(), ROOT, SPECS)

    def test_missing_optional_or_metal_lock_entries_are_rejected(self):
        for name in ("frankenjax-core", "frankenjax-lax", "ft-kernel-metal"):
            with self.subTest(name=name), self.assertRaisesRegex(archives.AuditError, "locked implementation"):
                release.check_lock(lock_bytes(missing=name), ROOT, SPECS)

    def test_old_same_family_kernel_does_not_pass(self):
        with self.assertRaisesRegex(archives.AuditError, "locked implementation"):
            release.check_lock(lock_bytes(changed="0.1.0"), ROOT, SPECS)

    def test_git_dependency_and_invalid_checksum_fail(self):
        for kwargs in (dict(bad_source="ft-kernel-metal"), dict(bad_checksum="frankenjax-core")):
            with self.subTest(kwargs=kwargs), self.assertRaises(archives.AuditError):
                release.check_lock(lock_bytes(**kwargs), ROOT, SPECS)

    def test_duplicate_root_is_not_accepted(self):
        data = lock_bytes() + b'\n[[package]]\nname = "franken_whisper"\nversion = "0.10.0"\n'
        with self.assertRaisesRegex(archives.AuditError, "exactly one"):
            release.check_lock(data, ROOT, SPECS)

    def test_metadata_rejects_path_and_git_substitutions(self):
        root = dict(id="root", **ROOT, source=None)
        for source in (None, "git+https://github.com/example/fork"):
            data = dict(resolve=dict(root="root"), packages=[root, dict(id="other", name="hidden", source=source)])
            with self.subTest(source=source), self.assertRaisesRegex(archives.AuditError, "non-registry"):
                release.check_metadata(data, ROOT)

    def test_metadata_must_describe_the_actual_root(self):
        with self.assertRaisesRegex(archives.AuditError, "staged release root"):
            release.check_metadata(dict(resolve=dict(root="other"), packages=[]), ROOT)


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.repo = self.parent / "repo"
        self.repo.mkdir()
        (self.repo / "src").mkdir()
        (self.repo / "src/lib.rs").write_text("// unchanged numerical source\n")
        (self.repo / "tests").mkdir()
        (self.repo / "tests/conformance_oracle_tests.rs").write_text("// independent oracle fixture\n")
        (self.repo / "Cargo.toml").write_text(MANIFEST)
        (self.repo / "Cargo.lock").write_bytes(b"development sibling lock\n")
        (self.repo / "scripts").mkdir()
        (self.repo / "scripts/registry-source-contract.json").write_text(json.dumps(dict(schema=1, packages=SPECS)))
        (self.repo / ".cargo").mkdir()
        (self.repo / ".cargo/config.toml").write_text(CONFIG)
        self.git("init", "-q")
        self.git("add", ".")
        self.commit()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.DEVNULL)

    def commit(self):
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")

    def test_snapshot_preserves_build_config_but_not_development_lock_or_untracked_files(self):
        (self.repo / "untracked-secret").write_text("not a release input")
        stage, revision = release.snapshot(self.repo, self.parent / "candidate")
        self.assertFalse((stage / "Cargo.lock").exists())
        self.assertEqual((stage / ".cargo/config.toml").read_text(), CONFIG)
        self.assertFalse((stage / "untracked-secret").exists())
        self.assertEqual((stage / "src/lib.rs").read_bytes(), (self.repo / "src/lib.rs").read_bytes())
        self.assertEqual(len(revision), 40)
        self.assertEqual((self.repo / "Cargo.lock").read_bytes(), b"development sibling lock\n")

    def test_unreviewed_config_is_rejected_instead_of_silently_removed(self):
        for extra in ('[source.crates-io]\nreplace-with = "private"\n',
                      '[alias]\ntest = "check"\n',
                      '[patch.crates-io]\nblock = { path = "../block" }\n'):
            with self.subTest(extra=extra):
                (self.repo / ".cargo/config.toml").write_text(CONFIG + extra)
                self.git("add", ".cargo/config.toml")
                self.commit()
                with self.assertRaisesRegex(archives.AuditError, "unreviewed Cargo"):
                    release.snapshot(self.repo, self.parent / str(len(extra)))

    def test_hidden_worktree_changes_cannot_become_release_source(self):
        committed = (self.repo / "src/lib.rs").read_bytes()
        self.git("update-index", "--assume-unchanged", "src/lib.rs")
        (self.repo / "src/lib.rs").write_text("// not in the reported commit\n")
        self.assertFalse(self.git("status", "--porcelain").strip())
        stage, _ = release.snapshot(self.repo, self.parent / "candidate")
        self.assertEqual((stage / "src/lib.rs").read_bytes(), committed)

    def test_executable_mode_is_preserved_from_git(self):
        script = self.repo / "scripts/run.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        self.git("add", "scripts/run.sh")
        self.commit()
        stage, _ = release.snapshot(self.repo, self.parent / "candidate")
        self.assertEqual((stage / "scripts/run.sh").stat().st_mode & 0o777, 0o755)

    def test_prepare_only_receipt_never_claims_registry_or_build_success(self):
        output = self.parent / "candidate"
        with contextlib.redirect_stdout(io.StringIO()):
            code = release.main(["--repo", str(self.repo), "--output", str(output), "--prepare-only"])
        self.assertEqual(code, 0)
        receipt = json.loads((output / "registry-release-receipt.json").read_text())
        self.assertEqual(receipt["status"], "prepared_only")
        self.assertEqual(receipt["registry_artifacts"], "not_run")
        self.assertEqual(receipt["whisper_tests"], "not_run")
        self.assertFalse(receipt["published"])
        self.assertFalse(receipt["commands"])
        self.assertEqual(len(receipt["build_config_sha256"]), 64)
        self.assertFalse((output / "source/Cargo.lock").exists())
        self.assertEqual(self.git("status", "--porcelain").strip(), b"")

    def test_missing_publication_stops_before_any_cargo_command(self):
        output = self.parent / "candidate"
        failed = dict(schema=1, artifact_integrity="fail", errors=["not published"], verified=[])
        real_run = subprocess.run
        def git_only(command, **kwargs):
            self.assertEqual(command[0], "git", "no Cargo command may run before publication checks pass")
            return real_run(command, **kwargs)
        with patch.object(archives, "audit", return_value=failed), patch.object(release.subprocess, "run", side_effect=git_only) as run:
            with contextlib.redirect_stderr(io.StringIO()):
                code = release.main(["--repo", str(self.repo), "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertTrue(all(call.args[0][0] == "git" for call in run.call_args_list))
        receipt = json.loads((output / "registry-release-receipt.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["whisper_tests"], "not_run")

    def test_ci_source_and_release_tag_are_bound_to_the_committed_manifest(self):
        output = self.parent / "candidate"
        revision = self.git("rev-parse", "HEAD").decode().strip()
        with contextlib.redirect_stdout(io.StringIO()):
            code = release.main(["--repo", str(self.repo), "--output", str(output),
                                 "--expected-revision", revision, "--release-tag", "v0.10.0", "--prepare-only"])
        self.assertEqual(code, 0)
        receipt = json.loads((output / "registry-release-receipt.json").read_text())
        self.assertEqual(receipt["source_revision"], revision)
        self.assertEqual(receipt["release_tag"], "v0.10.0")
        self.assertEqual(receipt["status"], "prepared_only")

    def test_mismatched_tag_cannot_relabel_a_release(self):
        output = self.parent / "candidate"
        with contextlib.redirect_stderr(io.StringIO()):
            code = release.main(["--repo", str(self.repo), "--output", str(output),
                                 "--release-tag", "v0.11.0"])
        self.assertEqual(code, 1)
        receipt = json.loads((output / "registry-release-receipt.json").read_text())
        self.assertIn("does not match package version", receipt["error"])
        self.assertFalse(receipt["commands"])
        self.assertEqual(tomllib.loads((self.repo / "Cargo.toml").read_text())["package"]["version"], "0.10.0")

    def test_stale_ci_revision_cannot_qualify_a_different_commit(self):
        output = self.parent / "candidate"
        with contextlib.redirect_stderr(io.StringIO()):
            code = release.main(["--repo", str(self.repo), "--output", str(output),
                                 "--expected-revision", "0" * 40])
        self.assertEqual(code, 1)
        receipt = json.loads((output / "registry-release-receipt.json").read_text())
        self.assertIn("source revision differs", receipt["error"])
        self.assertFalse(receipt["commands"])

    def test_existing_evidence_directory_is_not_overwritten(self):
        output = self.parent / "candidate"
        output.mkdir()
        sentinel = output / "registry-release-receipt.json"
        sentinel.write_text("prior evidence")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(release.main(["--repo", str(self.repo), "--output", str(output)]), 1)
        self.assertEqual(sentinel.read_text(), "prior evidence")

    def test_dirty_checkout_and_nested_destination_fail(self):
        with self.assertRaisesRegex(archives.AuditError, "outside"):
            release.snapshot(self.repo, self.repo / "candidate")
        (self.repo / "src/lib.rs").write_text("dirty")
        with self.assertRaisesRegex(archives.AuditError, "committed"):
            release.snapshot(self.repo, self.parent / "candidate")

    def test_ancestor_cargo_config_is_rejected(self):
        (self.parent / ".cargo").mkdir()
        (self.parent / ".cargo/config").write_text("# inherited override")
        with self.assertRaisesRegex(archives.AuditError, "inherit Cargo configuration"):
            release.snapshot(self.repo, self.parent / "candidate")

    def test_tracked_symlink_cannot_pull_in_external_sources(self):
        (self.repo / "external").symlink_to(self.parent / "outside")
        self.git("add", "external")
        self.commit()
        with self.assertRaisesRegex(archives.AuditError, "symlink"):
            release.snapshot(self.repo, self.parent / "candidate")

    def test_cargo_environment_drops_source_overrides_and_wrappers(self):
        with patch.dict(os.environ, {"CARGO_HOME": "/unsafe", "CARGO_REGISTRIES_CRATES_IO_INDEX": "private",
                                     "RUSTC_WRAPPER": "custom", "RUSTFLAGS": "custom", "RUSTUP_TOOLCHAIN": "override",
                                     "RUST_MIN_STACK": "1", "RUST_TEST_THREADS": "1"}):
            env = release.cargo_environment(self.parent / "candidate")
        self.assertNotIn("CARGO_REGISTRIES_CRATES_IO_INDEX", env)
        self.assertNotIn("RUSTC_WRAPPER", env)
        self.assertNotIn("RUSTFLAGS", env)
        self.assertNotIn("RUSTUP_TOOLCHAIN", env)
        self.assertNotIn("RUST_MIN_STACK", env)
        self.assertNotIn("RUST_TEST_THREADS", env)
        self.assertEqual(env["CARGO_HOME"], str(self.parent / "candidate/cargo-home"))



class PipelineTests(unittest.TestCase):
    def run_pipeline(self, replay=False, mutate_lock=False, package_fault=None, oracle=None,
                     fail_packaged_build=False, mutate_packaged=False, audit_checksum="c" * 64):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stage = output / "source"
            stage.mkdir()
            text = release.registry_manifest(MANIFEST, SPECS)
            (stage / "Cargo.toml").write_text(text)
            contents = {"src/lib.rs": b"// numerical source\n",
                        "tests/conformance_oracle_tests.rs": b"// independent oracle\n",
                        ".cargo/config.toml": CONFIG.encode()}
            for name, data in contents.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            if replay:
                (stage / "Cargo.lock").write_bytes(lock_bytes())
            report = {"commands": []}
            observed = []

            def fake_cargo(command, **kwargs):
                observed.append(command)
                self.assertEqual(command[0], "cargo-fixture")
                cwd = kwargs["cwd"]
                packaged = cwd == output / "packaged-source"
                self.assertIn(cwd, (stage, output / "packaged-source"))
                env_root = output / "packaged-validation" if packaged else output
                self.assertEqual(kwargs["env"]["CARGO_HOME"], str(env_root / "cargo-home"))
                self.assertEqual(kwargs["env"]["CARGO_TARGET_DIR"], str(env_root / "target"))
                self.assertEqual((cwd / ".cargo/config.toml").read_text(), CONFIG)
                if packaged and "fetch" not in command:
                    self.assertIn("--frozen", command)
                if "metadata" in command:
                    if not packaged:
                        self.assertEqual("--locked" in command, replay)
                        (stage / "Cargo.lock").write_bytes(lock_bytes())
                    root = dict(id="root", **ROOT, source=None)
                    metadata = dict(resolve=dict(root="root"), packages=[root])
                    kwargs["stdout"].write(json.dumps(metadata).encode())
                elif "package" in command:
                    self.assertIn("--locked", command)
                    self.assertIn("--all-features", command)
                    self.assertNotIn("--no-verify", command)
                    path = output / "target/package/franken_whisper-0.10.0.crate"
                    path.parent.mkdir(parents=True)
                    package_files = {"Cargo.toml": text.encode(), "Cargo.toml.orig": text.encode(),
                                     "Cargo.lock": lock_bytes(), **contents}
                    if package_fault == "missing-oracle":
                        package_files.pop("tests/conformance_oracle_tests.rs")
                    elif package_fault == "missing-config":
                        package_files.pop(".cargo/config.toml")
                    elif package_fault == "changed-source":
                        package_files["src/lib.rs"] = b"// a different implementation\n"
                    elif package_fault == "extra-source":
                        package_files["build.rs"] = b"fn main() {}\n"
                    elif package_fault == "dependency-change":
                        package_files["Cargo.toml"] = text.replace('serde = "1"', 'serde = "2"').encode()
                    with tarfile.open(path, "w:gz") as crate:
                        for name, data in package_files.items():
                            member = tarfile.TarInfo("franken_whisper-0.10.0/" + name)
                            member.size = len(data)
                            crate.addfile(member, io.BytesIO(data))
                    if mutate_lock:
                        (stage / "Cargo.lock").write_bytes(lock_bytes() + b"# mutation\n")
                elif packaged and "build" in command:
                    if fail_packaged_build:
                        raise subprocess.CalledProcessError(101, command)
                    self.assertIn("--release", command)
                elif packaged and "test" in command:
                    self.assertIn("--release", command)
                    self.assertIn("--format=json", command)
                    kwargs["stdout"].write(oracle_events() if oracle is None else oracle)
                    if mutate_packaged:
                        (cwd / "Cargo.lock").write_bytes(lock_bytes() + b"# changed\n")
                return subprocess.CompletedProcess(command, 0)

            passed = dict(artifact_integrity="pass", errors=[], verified=[
                dict(name=s["name"], version=s["version"], archive_sha256=audit_checksum) for s in SPECS])
            with patch.object(archives, "audit", return_value=passed), patch.object(release.subprocess, "run", side_effect=fake_cargo):
                release.qualify(stage, output, SPECS, ["cargo-fixture"], report)
            self.assertEqual(report["status"], "registry_candidate_verified")
            self.assertEqual(report["packaged_artifact_validation"], "pass")
            self.assertEqual(report["packaged_oracle_tests_passed"], 1)
            self.assertTrue(all(i["status"] == "pass" for i in report["invocations"]))
            self.assertEqual(report["replaying_registry_lock"], replay)
            tests = [c for c in observed if "test" in c]
            self.assertEqual(len(tests), 4)
            self.assertNotIn("--all-features", tests[0])
            self.assertIn("--all-features", tests[2])
            self.assertIn("conformance_oracle_tests", tests[1])
            self.assertIn("fj-oracle", tests[1])
            self.assertTrue(all("--locked" in c for c in tests[:3]))
            self.assertIn("--frozen", tests[3])
            self.assertEqual(len([c for c in observed if "build" in c]), 2)
            self.assertTrue(all("--tests" in tests[i] for i in (0, 2)))
            self.assertEqual(len(report["crate_sha256"]), 64)

    def test_complete_pipeline_commands_and_receipts_with_fake_cargo(self):
        self.run_pipeline()

    def test_replay_uses_locked_metadata_and_builds(self):
        self.run_pipeline(replay=True)

    def test_even_a_runner_that_mutates_locked_inputs_is_rejected(self):
        with self.assertRaisesRegex(archives.AuditError, "changed the candidate lockfile"):
            self.run_pipeline(mutate_lock=True)

    def test_incomplete_or_different_packages_do_not_qualify(self):
        for fault in ("missing-oracle", "missing-config", "changed-source", "extra-source", "dependency-change"):
            with self.subTest(fault=fault), self.assertRaises(archives.AuditError):
                self.run_pipeline(package_fault=fault)

    def test_same_version_but_different_audited_archive_checksum_is_rejected(self):
        with self.assertRaisesRegex(archives.AuditError, "locked checksum differs"):
            self.run_pipeline(audit_checksum="d" * 64)

    def test_a_packaged_build_failure_cannot_produce_success(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline(fail_packaged_build=True)

    def test_zero_or_ignored_oracle_tests_do_not_qualify(self):
        for output in (b"", oracle_events(0), oracle_events(0, 1), oracle_events(1, 1)):
            with self.subTest(output=output), self.assertRaisesRegex(archives.AuditError, "nonempty"):
                self.run_pipeline(oracle=output)

    def test_packaged_lock_mutation_is_rejected(self):
        with self.assertRaisesRegex(archives.AuditError, "packaged validation changed an input"):
            self.run_pipeline(mutate_packaged=True)


class ArtifactTests(unittest.TestCase):
    def test_json_execution_receipt_requires_the_whole_suite(self):
        self.assertEqual(release.check_oracle_execution(oracle_events(3)), 3)
        for output in (oracle_events(3).replace(b'"passed": 3', b'"passed": 4'),
                       oracle_events(3).replace(b'"event": "ok"', b'"event": "failed"'),
                       oracle_events(3).replace(b'"filtered_out": 0', b'"filtered_out": 1'),
                       oracle_events(3).replace(b'oracle_1', b'oracle_0')):
            with self.subTest(output=output), self.assertRaises(archives.AuditError):
                release.check_oracle_execution(output)

    def test_normalized_dependency_defaults_do_not_count_as_changes(self):
        manifest = tomllib.loads(release.registry_manifest(MANIFEST, SPECS))
        normalized = copy.deepcopy(manifest)
        normalized["dependencies"]["serde"] = dict(version="1", optional=False, **{"default-features": True})
        normalized["dependencies"]["fj-core"]["features"] = []
        self.assertEqual(release.dependency_contract(manifest), release.dependency_contract(normalized))

    def test_incomplete_or_duplicate_audit_receipts_are_rejected(self):
        verified = [dict(name=s["name"], version=s["version"], archive_sha256="c" * 64) for s in SPECS]
        for records in (verified[:-1], [*verified, verified[0]]):
            with self.subTest(records=records), self.assertRaisesRegex(archives.AuditError, "complete source contract"):
                release.check_audited_checksums(lock_bytes(), records, SPECS)

    def test_extraction_rejects_file_directory_collision_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"
            with self.assertRaisesRegex(archives.AuditError, "collision"):
                release.extract_package({"src": (0o644, b"file"), "src/lib.rs": (0o644, b"code")}, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
