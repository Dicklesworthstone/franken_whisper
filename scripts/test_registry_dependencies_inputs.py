#!/usr/bin/env python3
"""Real-Git regression tests for shared golden inputs; no Rust execution claimed."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import package_registry_dependencies as packages
import verify_registry_archives as archives


class WorkspaceTestInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.repo = self.parent / "repo"
        self.repo.mkdir()
        self.toolchain = b'[toolchain]\nchannel = "nightly-2026-08-31"\n'

    def git(self, *args):
        return packages.git(self.repo, *args)

    def commit(self):
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "fixture inputs")
        self.revision = self.git("rev-parse", "HEAD").decode().strip()

    def fixture(self, family="torch"):
        self.family = family
        repository, self.paths = packages.FAMILIES[family]
        self.canonical = "https://github.com/Dicklesworthstone/" + repository
        self.git("init", "-q")
        (self.repo / "Cargo.toml").write_text(packages.render({
            "workspace": {"resolver": "2", "members": list(self.paths.values())},
        }))
        (self.repo / "LICENSE").write_text("fixture license\n")
        self.files = {}
        for name, path in self.paths.items():
            directory = self.repo / path
            (directory / "src").mkdir(parents=True)
            data_path = f"artifacts/golden/{name}.txt"
            data = (name + "\nexpected_bits=3ff0000000000000\n").encode()
            self.files[data_path] = data
            target = self.repo / data_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            (directory / "Cargo.toml").write_text(packages.render({
                "package": {"name": name, "version": "0.1.0", "repository": self.canonical},
            }))
            (directory / "src/lib.rs").write_text(
                '#[cfg(test)]\nconst GOLDEN: &str = include_str!("../../../' + data_path + '");\n')
        (self.repo / "artifacts/unrelated.txt").write_text("must not enter the selected closure")
        self.commit()
        self.specs = [dict(name=name, version="0.1.0", repository=self.canonical,
            source_revision=self.revision,
            src_tree=self.git("rev-parse", f"HEAD:{path}/src").decode().strip(),
            workspace_test_files={f"artifacts/golden/{name}.txt":
                archives.git_object("blob", self.files[f"artifacts/golden/{name}.txt"]).hex()})
            for name, path in self.paths.items()]

    def prepare(self, specs=None):
        output = self.parent / f"output-{len(list(self.parent.iterdir()))}"
        output.mkdir()
        report = {"published": False, "commands": [], "unit_tests": "not_run"}
        stage, workspace = packages.prepare(self.repo, output, self.family,
            self.specs if specs is None else specs, self.toolchain, self.revision, False, report)
        return stage, workspace, report

    def test_stages_original_golden_bytes_at_include_str_locations(self):
        self.fixture()
        stage, workspace, report = self.prepare()
        for name, path in self.paths.items():
            relative = f"artifacts/golden/{name}.txt"
            included = stage / path / "src" / ("../../../" + relative)
            self.assertEqual(included.read_bytes(), self.files[relative])
            self.assertEqual(report["workspace_test_inputs"][relative], {
                "git_blob": archives.git_object("blob", self.files[relative]).hex(),
                "sha256": packages.sha256(self.files[relative]),
            })
            for original in (self.repo / path).rglob("*"):
                if original.is_file():
                    self.assertEqual(original.read_bytes(), (stage / original.relative_to(self.repo)).read_bytes())
        self.assertEqual(set(workspace["members"]), set(self.paths.values()))
        self.assertFalse((stage / "artifacts/unrelated.txt").exists())
        self.assertFalse((stage / ".git").exists())
        self.assertFalse(report["published"])
        self.assertEqual(report["unit_tests"], "not_run")
        self.assertEqual(report["commands"], [])
        self.assertEqual(self.git("status", "--porcelain"), b"")

    def test_jax_helper_core_lax_keep_their_data_and_sources(self):
        self.fixture("jax")
        stage, _, report = self.prepare()
        self.assertEqual(len(report["workspace_test_inputs"]), 3)
        for path, data in self.files.items():
            self.assertEqual((stage / path).read_bytes(), data)

    def test_old_contract_without_shared_data_remains_supported(self):
        self.fixture()
        for spec in self.specs:
            spec.pop("workspace_test_files")
        stage, _, report = self.prepare()
        self.assertEqual(report["workspace_test_inputs"], {})
        self.assertFalse((stage / "artifacts").exists())

    def test_worktree_content_hidden_by_git_flags_is_not_used(self):
        self.fixture()
        path = next(iter(self.files))
        self.git("update-index", "--assume-unchanged", path)
        (self.repo / path).write_text("unreviewed golden replacement")
        stage, _, _ = self.prepare()
        self.assertEqual((stage / path).read_bytes(), self.files[path])

    def test_untracked_fixture_cannot_satisfy_a_missing_committed_input(self):
        self.fixture()
        path = "artifacts/missing.txt"
        (self.repo / path).write_bytes(b"untracked replacement")
        self.specs[0]["workspace_test_files"][path] = archives.git_object("blob", b"untracked replacement").hex()
        with self.assertRaisesRegex(archives.AuditError, "missing committed"):
            self.prepare()

    def test_changed_golden_does_not_get_a_new_baseline_automatically(self):
        self.fixture()
        path = next(iter(self.files))
        (self.repo / path).write_text("changed expected output")
        self.commit()
        with self.assertRaisesRegex(archives.AuditError, "changed reviewed"):
            self.prepare()

    def test_incorrect_fixture_blob_is_rejected(self):
        self.fixture()
        path = next(iter(self.specs[0]["workspace_test_files"]))
        self.specs[0]["workspace_test_files"][path] = "0" * 40
        with self.assertRaisesRegex(archives.AuditError, "changed reviewed"):
            self.prepare()

    def test_executable_fixture_is_not_accepted_as_data(self):
        self.fixture()
        (self.repo / next(iter(self.files))).chmod(0o755)
        self.commit()
        with self.assertRaisesRegex(archives.AuditError, "changed reviewed"):
            self.prepare()

    def test_symlink_fixture_is_rejected_without_following_it(self):
        self.fixture()
        path = "artifacts/link.txt"
        (self.repo / path).symlink_to(self.parent / "outside")
        self.commit()
        self.specs[0]["workspace_test_files"][path] = self.git("rev-parse", f"HEAD:{path}").decode().strip()
        with self.assertRaisesRegex(archives.AuditError, "unsupported publication input"):
            self.prepare()

    def test_directory_cannot_stand_in_for_a_declared_file(self):
        self.fixture()
        path = "artifacts/directory.txt"
        (self.repo / path).mkdir()
        (self.repo / path / "unexpected.txt").write_text("not a declared file")
        self.commit()
        self.specs[0]["workspace_test_files"][path] = self.git("rev-parse", f"HEAD:{path}").decode().strip()
        with self.assertRaisesRegex(archives.AuditError, "changed reviewed"):
            self.prepare()

    def test_shared_file_is_deduplicated_when_two_members_pin_it(self):
        self.fixture()
        self.specs[1]["workspace_test_files"].update(self.specs[0]["workspace_test_files"])
        _, _, report = self.prepare()
        self.assertEqual(len(report["workspace_test_inputs"]), 3)

    def test_conflicting_member_pins_fail_before_staging(self):
        self.fixture()
        path = next(iter(self.specs[0]["workspace_test_files"]))
        self.specs[1]["workspace_test_files"][path] = "0" * 40
        with self.assertRaisesRegex(archives.AuditError, "conflicting"):
            self.prepare()

    def test_other_family_inputs_are_not_imported(self):
        self.fixture()
        foreign = dict(name="frankenjax-core", repository="https://github.com/Dicklesworthstone/frankenjax",
                       workspace_test_files={"artifacts/foreign.txt": "0" * 40})
        _, _, report = self.prepare([*self.specs, foreign])
        self.assertNotIn("artifacts/foreign.txt", report["workspace_test_inputs"])

    def test_dangerous_paths_and_pathspecs_are_rejected(self):
        paths = ["../outside.txt", "/artifacts/data.txt", "artifacts/../Cargo.toml",
                 "artifacts//data.txt", "artifacts/./data.txt", "artifacts/data.txt/",
                 "artifacts/*.txt", "artifacts/[ab].txt", "artifacts/what?.txt",
                 "artifacts/:magic.txt", "artifacts/back\\slash.txt", "artifacts/new\nline.txt",
                 "artifacts/.cargo/config.txt", "artifacts/.git/config.txt", "artifacts/test.rs",
                 ".cargo/config.toml", "Cargo.toml", "crates/other/src/lib.rs", ""]
        for path in paths:
            with self.subTest(path=path), self.assertRaises(archives.AuditError):
                packages.workspace_test_files({"member": {"workspace_test_files": {path: "a" * 40}}})

    def test_invalid_mappings_and_blob_ids_are_rejected(self):
        for value in (None, [], "path", 42):
            with self.subTest(value=value), self.assertRaises(archives.AuditError):
                packages.workspace_test_files({"member": {"workspace_test_files": value}})
        for oid in (None, 42, "main", "A" * 40, "a" * 39, "a" * 41):
            with self.subTest(oid=oid), self.assertRaises(archives.AuditError):
                packages.workspace_test_files({"member": {"workspace_test_files": {"artifacts/data.txt": oid}}})

    def test_fixture_bytes_are_part_of_the_mutation_inventory(self):
        self.fixture()
        stage, _, _ = self.prepare()
        before = packages.inventory(stage)
        path = next(iter(self.files))
        self.assertIn(path, before)
        (stage / path).write_text("mutated during a test")
        self.assertNotEqual(before, packages.inventory(stage))

    def test_missing_fixture_cli_fails_without_running_cargo(self):
        self.fixture()
        self.specs[0]["workspace_test_files"]["artifacts/missing.txt"] = "0" * 40
        contract = self.parent / "contract.json"
        contract.write_text(json.dumps({"schema": 1, "packages": self.specs}))
        toolchain = self.parent / "toolchain.toml"
        toolchain.write_bytes(self.toolchain)
        output = self.parent / "cli-output"
        with contextlib.redirect_stdout(io.StringIO()):
            result = packages.main(["--repo", str(self.repo), "--family", self.family,
                "--expected-revision", self.revision, "--output", str(output),
                "--toolchain-file", str(toolchain), "--contract", str(contract)])
        report = json.loads((output / "dependency-package-receipt.json").read_text())
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["commands"])
        self.assertFalse(report["published"])
        self.assertEqual(report["unit_tests"], "not_run")


if __name__ == "__main__":
    unittest.main()
