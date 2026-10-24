#!/usr/bin/env python3
"""Offline native-gate controls. Fake Cargo output is NOT GPU execution evidence."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import qualify_registry_metal as native
import verify_registry_archives as archives

ROOT = {"name": "franken_whisper", "version": "0.10.0"}
VERSIONS = {"frankentorch-core": "0.1.1", "frankentorch-kernel-cpu": "0.1.2",
            "ft-kernel-metal": "0.1.1", "frankenjax-core": "0.1.0", "frankenjax-lax": "0.1.0",
            "frankenjax-test-utils": "0.1.0"}
AUDITED = [dict(name=n, version=v, archive_sha256="c" * 64) for n, v in VERSIONS.items()]
MANIFEST = b'''[package]
name = "franken_whisper"
version = "0.10.0"
[dependencies]
ft-kernel-metal = "=0.1.1"
'''


def receipt(revision):
    return dict(status="registry_candidate_verified", published=False, source_revision=revision,
                packaged_oracle_tests_passed=4, **dict.fromkeys(native.PHASES, "pass"))


def events(names):
    result = [dict(type="suite", event="started", test_count=len(names))]
    for name in sorted(names):
        result += [dict(type="test", event="started", name=name), dict(type="test", event="ok", name=name)]
    result += [dict(type="suite", event="ok", passed=len(names), failed=0, ignored=0, filtered_out=0)]
    return result


def encode(result):
    return ("\n".join(json.dumps(r) for r in result) + "\n").encode()


def lock_bytes():
    text = 'version = 4\n\n[[package]]\nname = "franken_whisper"\nversion = "0.10.0"\n'
    for p in AUDITED:
        if p["name"] == "frankenjax-test-utils":
            continue
        text += f'\n[[package]]\nname = "{p["name"]}"\nversion = "{p["version"]}"\n'
        text += f'source = "{native.REGISTRY}"\nchecksum = "{p["archive_sha256"]}"\n'
    return text.encode()


def metadata():
    return dict(packages=[dict(id="root", source=None, **ROOT),
                          dict(id="metal", name="ft-kernel-metal", version="0.1.1", source=native.REGISTRY)],
                resolve=dict(root="root", nodes=[dict(id="root", deps=[dict(name="ft_kernel_metal", pkg="metal")])]))


class EvidenceTests(unittest.TestCase):
    def test_named_suites_pass(self):
        for names in native.TESTS.values():
            self.assertEqual(native.require_tests(encode(events(names)), names), len(names))

    def test_missing_failed_ignored_duplicate_or_unexpected_tests_fail(self):
        names = {"one", "two"}
        good = events(names)
        bad = [good[:-1], good[1:], events(set()), good + [good[-1]],
               good[:1] + good[2:], good[:2] + [good[1]] + good[2:],
               events({"one", "different"})]
        for state in ("failed", "ignored", "timeout"):
            changed = copy.deepcopy(good)
            changed[2]["event"] = state
            bad.append(changed)
        for field, value in (("passed", True), ("ignored", 1), ("filtered_out", 1), ("failed", 1)):
            changed = copy.deepcopy(good)
            changed[-1][field] = value
            bad.append(changed)
        for result in bad:
            with self.subTest(result=result), self.assertRaises(archives.AuditError):
                native.require_tests(encode(result), names)

    def test_receipt_requires_all_phases_and_real_oracle_count(self):
        good = receipt("a" * 40)
        native.require_receipt(good, "a" * 40)
        changes = [(p, "not_run") for p in native.PHASES]
        changes += [("status", "prepared_only"), ("published", True), ("source_revision", "b" * 40),
                    ("packaged_oracle_tests_passed", 0), ("packaged_oracle_tests_passed", True)]
        for key, value in changes:
            with self.subTest(key=key), self.assertRaises(archives.AuditError):
                native.require_receipt(dict(good, **{key: value}), "a" * 40)

    def test_lock_must_match_registry_archive_checksums(self):
        native.check_lock(lock_bytes(), ROOT, AUDITED)
        for data in (lock_bytes().replace(b"registry+", b"git+", 1),
                     lock_bytes().replace(b"c" * 64, b"d" * 64, 1),
                     lock_bytes().replace(b"ft-kernel-metal", b"unrelated-metal"),
                     lock_bytes().replace(b'"0.1.2"', b'"0.1.0"'),
                     lock_bytes() + b'\n[[package]]\nname="franken_whisper"\nversion="0.10.0"\n'):
            with self.subTest(data=data[:80]), self.assertRaises(archives.AuditError):
                native.check_lock(data, ROOT, AUDITED)

    def test_metadata_requires_registry_metal_on_actual_root_edge(self):
        native.check_metadata(json.dumps(metadata()).encode(), ROOT)
        for field in ("source", "edge", "root"):
            data = metadata()
            if field == "source":
                data["packages"][1]["source"] = "git+https://example.invalid/fork"
            elif field == "edge":
                data["resolve"]["nodes"][0]["deps"] = []
            else:
                data["resolve"]["root"] = "other"
            with self.subTest(field=field), self.assertRaises(archives.AuditError):
                native.check_metadata(json.dumps(data).encode(), ROOT)

    def test_linux_and_intel_mac_are_not_native_qualification(self):
        for system, machine in (("linux", "aarch64"), ("darwin", "x86_64")):
            with patch.object(native.sys, "platform", system), patch.object(native.platform, "machine", return_value=machine):
                with self.assertRaises(archives.AuditError):
                    native.require_host()
        with patch.object(native.sys, "platform", "darwin"), patch.object(native.platform, "machine", return_value="arm64"):
            native.require_host()

    def test_environment_does_not_inherit_dependency_or_kernel_overrides(self):
        keys = {k: "override" for k in ("CARGO_HOME", "CARGO_REGISTRIES_CRATES_IO_INDEX", "RUSTFLAGS",
                "RUSTC_WRAPPER", "RUSTUP_TOOLCHAIN", "RUST_TEST_THREADS", "FT_METAL_SG_GEMM", "DYLD_INSERT_LIBRARIES")}
        with patch.dict(os.environ, keys):
            env = native.clean_environment(Path("/qualification"))
        for key in keys.keys() - {"CARGO_HOME"}:
            self.assertNotIn(key, env)
        self.assertEqual(env["CARGO_HOME"], "/qualification/cargo-home")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.candidate = self.base / "candidate"
        (self.candidate / "source").mkdir(parents=True)
        (self.candidate / "source/Cargo.toml").write_bytes(MANIFEST)
        self.crate = self.candidate / "target/package/franken_whisper-0.10.0.crate"
        self.crate.parent.mkdir(parents=True)
        self.files = {"Cargo.toml": MANIFEST, "Cargo.lock": lock_bytes(),
                      ".cargo/config.toml": b'[env]\nRUST_MIN_STACK="67108864"\n',
                      "rust-toolchain.toml": b'[toolchain]\nchannel="nightly-2026-08-31"\n',
                      "src/lib.rs": b"// immutable library\n"}
        for target in native.TESTS:
            path = f"tests/{target}.rs"
            self.files[path] = (Path(__file__).parents[1] / path).read_bytes()
        for name, data in self.files.items():
            p = self.repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        specs = [dict(name=n, version=v, src_tree="a" * 40, source_revision="b" * 40,
                      repository="https://github.com/example/source") for n, v in VERSIONS.items()]
        (self.repo / "scripts").mkdir()
        (self.repo / "scripts/registry-source-contract.json").write_text(json.dumps(dict(schema=1, packages=specs)))
        def git(*args):
            return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE)
        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
        self.revision = git("rev-parse", "HEAD").decode().strip()
        self.output = self.base / "native"
        self.write_candidate()
        self.commands = []

    def write_candidate(self):
        with tarfile.open(self.crate, "w:gz") as crate:
            for name, data in self.files.items():
                member = tarfile.TarInfo("franken_whisper-0.10.0/" + name)
                member.size = len(data)
                member.mode = 0o644
                crate.addfile(member, io.BytesIO(data))
        self.receipt = dict(receipt(self.revision), manifest_sha256=native.sha256(MANIFEST),
                            crate_sha256=native.sha256(self.crate.read_bytes()),
                            packaged_lock_sha256=native.sha256(self.files["Cargo.lock"]))
        (self.candidate / "registry-release-receipt.json").write_text(json.dumps(self.receipt))

    def fake_cargo(self, command, **kwargs):
        self.commands.append(command)
        if command[0] != "cargo-fixture":
            return self.real_run(command, **kwargs)
        if "metadata" in command:
            kwargs["stdout"].write(json.dumps(metadata()).encode())
        if "test" in command:
            target = command[command.index("--test") + 1]
            fault = target == "registry_metal_faults"
            self.assertEqual(native.FAULT_OVERRIDE in command, fault)
            self.assertIn("--frozen", command)
            self.assertIn("--release", command)
            self.assertIn("--ignored", command)
            self.assertIn("--test-threads=1", command)
            self.assertEqual(kwargs["env"]["CARGO_TARGET_DIR"], str(self.output / ("fault-target" if fault else "production-target")))
            self.assertEqual(kwargs["env"]["CARGO_HOME"], str(self.output / "cargo-home"))
            kwargs["stdout"].write(encode(events(native.TESTS[target])))
        return subprocess.CompletedProcess(command, 0)

    def run_gate(self, *, side_effect=None, audit=None):
        self.real_run = subprocess.run
        passed = dict(artifact_integrity="pass", verified=AUDITED, errors=[])
        with patch.object(native.sys, "platform", "darwin"), patch.object(native.platform, "machine", return_value="arm64"), \
             patch.object(archives, "audit", return_value=passed if audit is None else audit), \
             patch.object(native.subprocess, "run", side_effect=side_effect or self.fake_cargo), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return native.main(["--candidate", str(self.candidate), "--output", str(self.output),
                                "--repo", str(self.repo), "--expected-revision", self.revision, "--cargo", "cargo-fixture"])

    def test_complete_simulated_pipeline_preserves_inputs_and_separates_builds(self):
        original = self.crate.read_bytes()
        self.assertEqual(self.run_gate(), 0)
        report = json.loads((self.output / "native-metal-receipt.json").read_text())
        self.assertEqual(report["status"], "native_metal_verified")
        self.assertEqual(report["production_tests_passed"], 1)
        self.assertEqual(report["fault_tests_passed"], 5)
        self.assertFalse(report["published"])
        self.assertEqual(self.crate.read_bytes(), original)
        self.assertEqual((self.output / "source/Cargo.lock").read_bytes(), lock_bytes())
        self.assertEqual(len([c for c in self.commands if c[0] == "cargo-fixture"]), 5)

    def test_unpublished_archives_stop_before_cargo(self):
        self.assertEqual(self.run_gate(audit=dict(artifact_integrity="fail", verified=[], errors=["missing"])), 1)
        self.assertFalse(any(c[0] == "cargo-fixture" for c in self.commands))

    def test_missing_or_altered_packaged_native_tests_fail(self):
        self.files["tests/registry_metal_faults.rs"] = b"// weakened test\n"
        self.write_candidate()
        self.assertEqual(self.run_gate(), 1)
        self.assertFalse(any(c[0] == "cargo-fixture" for c in self.commands))

    def test_missing_packaged_native_test_fails(self):
        self.files.pop("tests/registry_metal_faults.rs")
        self.write_candidate()
        self.assertEqual(self.run_gate(), 1)

    def test_source_mutation_is_a_failure_not_a_receipt(self):
        def change(command, **kwargs):
            result = self.fake_cargo(command, **kwargs)
            if command[0] == "cargo-fixture" and "test" in command:
                (self.output / "source/src/lib.rs").write_text("// changed\n")
            return result
        self.assertEqual(self.run_gate(side_effect=change), 1)
        self.assertEqual(json.loads((self.output / "native-metal-receipt.json").read_text())["status"], "failed")

    def test_cargo_timeout_preserves_failure_receipt(self):
        def fail(command, **kwargs):
            if command[0] == "cargo-fixture":
                raise subprocess.TimeoutExpired(command, 1800)
            return self.fake_cargo(command, **kwargs)
        self.assertEqual(self.run_gate(side_effect=fail), 1)
        report = json.loads((self.output / "native-metal-receipt.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["production_tests_passed"], 0)

    def test_nonzero_cargo_preserves_failure_receipt(self):
        def fail(command, **kwargs):
            if command[0] == "cargo-fixture" and "test" in command:
                raise subprocess.CalledProcessError(101, command)
            return self.fake_cargo(command, **kwargs)
        self.assertEqual(self.run_gate(side_effect=fail), 1)
        report = json.loads((self.output / "native-metal-receipt.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["production_tests_passed"], 0)

    def test_zero_native_tests_cannot_qualify(self):
        def empty_suite(command, **kwargs):
            if command[0] == "cargo-fixture" and "test" in command:
                kwargs["stdout"].write(encode(events(set())))
                return subprocess.CompletedProcess(command, 0)
            return self.fake_cargo(command, **kwargs)
        self.assertEqual(self.run_gate(side_effect=empty_suite), 1)
        report = json.loads((self.output / "native-metal-receipt.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["production_tests_passed"], 0)

    def test_changed_crate_is_rejected_before_cargo(self):
        with self.crate.open("ab") as stream:
            stream.write(b"changed archive")
        self.assertEqual(self.run_gate(), 1)
        self.assertFalse(any(c[0] == "cargo-fixture" for c in self.commands))

    def test_old_evidence_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "native-metal-receipt.json"
        sentinel.write_text("prior evidence")
        self.assertEqual(self.run_gate(), 1)
        self.assertEqual(sentinel.read_text(), "prior evidence")

    def test_ancestor_cargo_config_is_rejected(self):
        (self.base / ".cargo").mkdir()
        (self.base / ".cargo/config.toml").write_text("# ancestor override\n")
        self.assertEqual(self.run_gate(), 1)
        self.assertFalse(self.commands)


if __name__ == "__main__":
    unittest.main()
