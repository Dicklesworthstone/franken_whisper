"""Real-filesystem output guards and simulated Cargo/package regression tests."""
import copy
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

import package_registry_dependencies as packages
import registry_test_outputs as outputs
import verify_registry_archives as archives

EMITTER = next(iter(outputs.REVIEWED_EMITTERS))
PATHS = tuple(sorted(outputs.REVIEWED_EMITTERS[EMITTER]))
SPECS = {
    EMITTER[0]: dict(version=EMITTER[1], src_tree=EMITTER[2]),
    "frankenjax-core": dict(version="0.1.0", src_tree="de700ac07837961002f6bc1f256e0fada64a0ee6"),
}
ROOTS = outputs.output_roots(SPECS)


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        self.stage = self.output / "source"
        self.stage.mkdir()
        for name, value in {"Cargo.lock": "immutable lock", "src/lib.rs": "immutable Rust",
                            "artifacts/reference.golden": "immutable expected output"}.items():
            self.write(name, value.encode())
        self.before = packages.inventory(self.stage)
        self.report = {}

    def write(self, name, data=b'{"all_passed":true}\n'):
        target = self.stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def emit(self):
        for name in PATHS:
            self.write(name)

    def collect(self):
        outputs.collect_outputs(self.stage, self.output, self.before,
                                packages.inventory(self.stage), PATHS, self.report, ROOTS)

    def test_exact_six_outputs_are_copied_and_hashed_without_changing_inputs(self):
        outputs.require_fresh_outputs(self.stage, self.before, PATHS, ROOTS)
        self.emit()
        self.collect()
        self.assertEqual(self.report["test_input_integrity"], "pass")
        self.assertEqual(set(self.report["workspace_test_outputs"]), set(PATHS))
        for name, record in self.report["workspace_test_outputs"].items():
            data = (self.output / record["artifact"]).read_bytes()
            self.assertEqual(data, (self.stage / name).read_bytes())
            self.assertEqual(packages.sha256(data), record["sha256"])
        after = packages.inventory(self.stage)
        self.assertEqual({p: after[p] for p in self.before}, self.before)

    def test_unknown_emitter_tree_or_version_gets_no_exemption(self):
        self.assertEqual(outputs.output_paths(SPECS), PATHS)
        self.assertEqual(outputs.output_paths({EMITTER[0]: dict(version="0.1.1", src_tree=EMITTER[2])}), ())
        self.assertEqual(outputs.output_paths({EMITTER[0]: dict(version=EMITTER[1], src_tree="0" * 40)}), ())
        self.assertEqual(outputs.output_paths({"another-package": SPECS[EMITTER[0]]}), ())

    def test_existing_output_rejected_before_tests(self):
        self.write(PATHS[0])
        with self.assertRaisesRegex(archives.AuditError, "already exists"):
            outputs.require_fresh_outputs(self.stage, self.before, PATHS)

    def test_input_cannot_be_reclassified_as_output(self):
        self.write(PATHS[0])
        self.before = packages.inventory(self.stage)
        with self.assertRaisesRegex(archives.AuditError, "overlaps immutable"):
            self.collect()

    def test_output_parent_cannot_be_an_input_file(self):
        self.write("artifacts/e2e", b"input file")
        with self.assertRaisesRegex(archives.AuditError, "parent is an input"):
            outputs.require_fresh_outputs(self.stage, packages.inventory(self.stage), PATHS)

    def test_output_contract_rejects_noncanonical_paths(self):
        for path in ("/artifacts/x.json", "artifacts/../Cargo.lock", "artifacts/x.rs",
                     "artifacts//x.json", "artifacts/.cargo/x.json", "artifacts/*.json",
                     "src/x.json", "artifacts/x\\y.json", "artifacts/x\ny.json"):
            with self.subTest(path=path), self.assertRaises(archives.AuditError):
                outputs.require_fresh_outputs(self.stage, self.before, (path,))

    def test_missing_required_output_fails(self):
        for name in PATHS[1:]:
            self.write(name)
        with self.assertRaisesRegex(archives.AuditError, "missing_outputs"):
            self.collect()
        self.assertEqual(self.report["test_input_integrity"], "failed")

    def test_reviewed_roots_accept_additional_structured_evidence(self):
        self.emit()
        extras = (
            "artifacts/e2e/e2e_bfloat16.e2e.json",
            "artifacts/testing/logs/fj-core/extra_case.json",
            "artifacts/testing/logs/fj-lax/extra_case.json",
        )
        for name in extras:
            self.write(name)
        self.collect()
        self.assertTrue(set(extras) <= set(self.report["workspace_test_outputs"]))

    def test_unreviewed_artifact_path_still_fails(self):
        self.emit()
        self.write("artifacts/other/unreviewed.json")
        with self.assertRaisesRegex(archives.AuditError, "unreviewed.json"):
            self.collect()

    def test_e2e_root_requires_e2e_json_suffix(self):
        self.emit()
        self.write("artifacts/e2e/unreviewed.json")
        with self.assertRaisesRegex(archives.AuditError, "unreviewed.json"):
            self.collect()

    def test_source_lock_and_golden_mutations_still_fail(self):
        self.emit()
        for name in self.before:
            original = (self.stage / name).read_bytes()
            self.write(name, original + b" altered")
            with self.subTest(name=name), self.assertRaisesRegex(archives.AuditError, "modified"):
                self.collect()
            self.write(name, original)

    def test_removed_input_still_fails(self):
        self.emit()
        # Rename rather than delete; the guard must report both sides.
        (self.stage / "src/lib.rs").rename(self.stage / "src/moved.rs")
        with self.assertRaisesRegex(archives.AuditError, "removed"):
            self.collect()
        self.assertIn("src/lib.rs", self.report["test_workspace_delta"]["removed"])

    def test_invalid_json_is_not_accepted(self):
        self.emit()
        self.write(PATHS[0], b"not JSON")
        with self.assertRaisesRegex(archives.AuditError, "invalid JSON"):
            self.collect()

    def test_scalar_json_is_not_evidence(self):
        self.emit()
        self.write(PATHS[0], b"true")
        with self.assertRaisesRegex(archives.AuditError, "structured JSON"):
            self.collect()

    def test_executable_output_is_rejected(self):
        self.emit()
        (self.stage / PATHS[0]).chmod(0o755)
        with self.assertRaisesRegex(archives.AuditError, "bounded regular"):
            self.collect()

    def test_linked_output_is_rejected(self):
        self.emit()
        os.link(self.stage / PATHS[0], self.output / "external-alias")
        with self.assertRaisesRegex(archives.AuditError, "bounded regular"):
            self.collect()

    def test_symlink_is_not_followed(self):
        for name in PATHS[1:]:
            self.write(name)
        (self.stage / PATHS[0]).symlink_to(self.stage / PATHS[1])
        with self.assertRaisesRegex(archives.AuditError, "symlink"):
            self.collect()

    def test_size_limit_is_enforced(self):
        self.emit()
        with patch.object(outputs, "MAX_EVIDENCE_BYTES", 2):
            with self.assertRaisesRegex(archives.AuditError, "bounded regular"):
                self.collect()

    def test_changed_evidence_after_inventory_is_rejected(self):
        self.emit()
        after = packages.inventory(self.stage)
        self.write(PATHS[0], b'{"changed":true}')
        with self.assertRaisesRegex(archives.AuditError, "changed during collection"):
            outputs.collect_outputs(self.stage, self.output, self.before, after, PATHS, self.report, ROOTS)

    def test_prior_evidence_is_not_overwritten(self):
        self.emit()
        (self.output / "test-evidence").mkdir()
        sentinel = self.output / "test-evidence/prior"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_no_outputs_keeps_original_exact_inventory_rule(self):
        report = {}
        outputs.collect_outputs(self.stage, self.output, self.before, self.before, (), report)
        self.assertEqual(report["workspace_test_outputs"], {})
        self.write("artifacts/unexpected.json")
        with self.assertRaises(archives.AuditError):
            outputs.collect_outputs(self.stage, self.output, self.before,
                                    packages.inventory(self.stage), (), {})


class PipelineTests(unittest.TestCase):
    def pipeline(self, fault=None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stage = output / "source"
            stage.mkdir()
            paths = packages.FAMILIES["jax"][1]
            order = ["frankenjax-test-utils", "frankenjax-core", "frankenjax-lax"]
            specs, manifests = {}, {}
            source = b"pub fn identity() -> u8 { 1 }\n"
            for name in order:
                root = stage / paths[name]
                (root / "src").mkdir(parents=True)
                (root / "src/lib.rs").write_bytes(source)
                specs[name] = dict(name=name, version="0.1.0", source_revision="a" * 40,
                    repository="https://github.com/Dicklesworthstone/frankenjax",
                    src_tree=archives.source_tree({"src/lib.rs": (0o644, source)}))
                manifest = dict(package=dict(name=name, version="0.1.0", edition="2024",
                    repository=specs[name]["repository"]), features={"extra": []})
                for dependency in order[:order.index(name)]:
                    manifest.setdefault("dev-dependencies", {})[dependency] = {
                        "version": "=0.1.0", "path": "../" + Path(paths[dependency]).name}
                manifests[name] = manifest
                (root / "Cargo.toml").write_text(packages.render(manifest))
            workspace = dict(members=list(paths.values()), resolver="2")
            (stage / "Cargo.toml").write_text(packages.render(dict(workspace=workspace)))
            report = {"commands": [], "publication_order": order}
            self.last_report = report
            self.calls = []

            def lock(local, previous):
                return packages.render({"version": 4, "package": [
                    dict(name=n, version="0.1.0") for n in local] + [
                    dict(name=n, version="0.1.0", source=packages.REGISTRY, checksum=h)
                    for n, h in previous.items()]}).encode()

            def fake_cargo(command, **kwargs):
                self.calls.append(command)
                if "metadata" in command:
                    (stage / "Cargo.lock").write_bytes(lock(order, {}))
                    members = [dict(name=n, version="0.1.0", id=n, source=None,
                        manifest_path=str(stage / paths[n] / "Cargo.toml")) for n in order]
                    kwargs["stdout"].write(json.dumps(dict(packages=members, workspace_members=order)).encode())
                elif "test" in command:
                    if fault == "test_failure":
                        raise subprocess.CalledProcessError(101, command)
                    for name in PATHS:
                        p = stage / name
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text('{"all_passed":true}')
                    if fault == "source_mutation":
                        (stage / paths[order[0]] / "src/lib.rs").write_text("changed")
                elif "package" in command:
                    self.assertEqual(report["unit_tests"], "pass")
                    self.assertEqual(report["test_input_integrity"], "pass")
                    if fault == "package_failure":
                        raise subprocess.CalledProcessError(101, command)
                    previous = {}
                    (output / "target/package").mkdir(parents=True)
                    for name in order:
                        normalized = copy.deepcopy(manifests[name])
                        for key, dep in packages.effective_dependencies(manifests[name], workspace).items():
                            normalized[key[1]][key[2]] = dep
                        files = {"Cargo.toml": packages.render(normalized).encode(),
                                 "Cargo.lock": lock([name], previous), "src/lib.rs": source}
                        buf = io.BytesIO()
                        with tarfile.open(fileobj=buf, mode="w:gz") as archive:
                            for path, content in files.items():
                                entry = tarfile.TarInfo(f"{name}-0.1.0/{path}")
                                entry.size, entry.mode = len(content), 0o644
                                archive.addfile(entry, io.BytesIO(content))
                        data = buf.getvalue()
                        (output / "target/package" / f"{name}-0.1.0.crate").write_bytes(data)
                        previous[name] = packages.sha256(data)
                    if fault == "evidence_mutation":
                        (stage / PATHS[0]).write_text('{"altered":true}')
                return subprocess.CompletedProcess(command, 0)

            # A tiny reviewed fixture emitter, not a claim to reproduce JAX numerics.
            fixture_key = ("frankenjax-lax", "0.1.0", specs["frankenjax-lax"]["src_tree"])
            with patch.dict(outputs.REVIEWED_EMITTERS, {fixture_key: PATHS}), \
                 patch.object(packages.subprocess, "run", side_effect=fake_cargo):
                packages.qualify(stage, output, "jax", list(specs.values()), workspace,
                                 ["cargo"], 2, 60, report)
            self.assertEqual(report["status"], "dependency_packages_verified")
            self.assertEqual(set(report["artifacts"]), set(order))
            self.assertEqual(len(report["workspace_test_outputs"]), 6)
            tests = next(c for c in self.calls if "test" in c)
            self.assertEqual(tests, ["cargo", "test", "--workspace", "--all-features", "--locked", "--jobs", "2"])
            self.assertEqual(len([c for c in self.calls if "package" in c]), 1)
            self.assertTrue(all("--no-verify" not in c for c in self.calls))

    def test_passing_evidence_emitter_reaches_real_archive_validation(self):
        self.pipeline()

    def test_genuine_source_mutation_still_stops_before_packaging(self):
        with self.assertRaisesRegex(archives.AuditError, "changed the locked source inputs"):
            self.pipeline("source_mutation")
        self.assertEqual(self.last_report["unit_tests"], "pass")
        self.assertEqual(self.last_report["test_input_integrity"], "failed")
        self.assertFalse(any("package" in c for c in self.calls))

    def test_packaging_cannot_rewrite_the_accepted_evidence(self):
        with self.assertRaisesRegex(archives.AuditError, "changed the locked source inputs"):
            self.pipeline("evidence_mutation")

    def test_test_failures_are_recorded_and_never_bypassed(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.pipeline("test_failure")
        self.assertEqual(self.last_report["unit_tests"], "failed")
        self.assertFalse(any("package" in c for c in self.calls))

    def test_package_failures_are_not_unit_test_failures(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.pipeline("package_failure")
        self.assertEqual(self.last_report["unit_tests"], "pass")
        self.assertEqual(self.last_report["package_verification"], "failed")


if __name__ == "__main__":
    unittest.main()
