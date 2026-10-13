#!/usr/bin/env python3
"""Offline controls for pre-publication packaging; Cargo execution is simulated."""
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

import package_registry_dependencies as packages
import verify_registry_archives as archives

VERSIONS = {"frankentorch-core": "0.1.1", "frankentorch-kernel-cpu": "0.1.2", "ft-kernel-metal": "0.1.1",
            "frankenjax-core": "0.1.0", "frankenjax-lax": "0.1.0", "frankenjax-test-utils": "0.1.0"}
TOOLCHAIN = b'[toolchain]\nchannel = "nightly-2026-08-31"\n'


def lock_bytes(local, bundled=None):
    doc = {"version": 4, "package": [{"name": name, "version": spec["version"]} for name, spec in local.items()]}
    for name, spec in (bundled or {}).items():
        doc["package"].append(dict(name=name, version=spec["version"], source=packages.REGISTRY, checksum=spec["archive_sha256"]))
    return packages.render(doc).encode()


def crate_bytes(name, version, files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, contents in files.items():
            member = tarfile.TarInfo(f"{name}-{version}/{path}")
            member.size = len(contents)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(contents))
    return buffer.getvalue()


class DependencyPackagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.repo = self.parent / "repo"
        self.repo.mkdir()
        self.toolchain = self.parent / "rust-toolchain.toml"
        self.toolchain.write_bytes(TOOLCHAIN)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.DEVNULL)

    def fixture(self, family="jax", with_patch=False):
        self.family = family
        repository, self.paths = packages.FAMILIES[family]
        root = {"workspace": {"resolver": "2", "members": list(self.paths.values()) + ["crates/unrelated"],
                "package": {"edition": "2024"}, "dependencies": {
                    "unrelated": {"path": "crates/unrelated"},
                    "serde": {"version": "1", "features": ["derive"]}}},
                "profile": {"release-perf": {"inherits": "release", "debug": "line-tables-only"}}}
        if family == "torch":
            root["workspace"]["dependencies"]["ft-core"] = dict(package="frankentorch-core", version="=0.1.1", path="crates/ft-core")
        if with_patch:
            root["patch"] = {"crates-io": {"block": {"git": "https://github.com/example/block", "rev": "a" * 40}}}
        (self.repo / "Cargo.toml").write_text(packages.render(root))
        (self.repo / "LICENSE").write_text("fixture license\n")
        (self.repo / "Cargo.lock").write_text("DO NOT COPY A DEVELOPMENT LOCK\n")
        (self.repo / ".cargo").mkdir()
        (self.repo / ".cargo/config.toml").write_text('[source.crates-io]\nreplace-with = "private"\n')
        (self.repo / "crates/unrelated").mkdir(parents=True)
        (self.repo / "crates/unrelated/Cargo.toml").write_text("invalid, must not enter the selected workspace")
        self.specs = []
        self.manifests = {}
        for name, path in self.paths.items():
            directory = self.repo / path
            (directory / "src").mkdir(parents=True)
            (directory / "src/lib.rs").write_text(f"// {name}\npub fn original() -> u8 {{ 42 }}\n")
            (directory / "tests").mkdir()
            (directory / "tests/check.rs").write_text("#[test] fn contract() { assert_eq!(2+2, 4); }\n")
            (directory / "LICENSE").write_text("fixture license\n")
            manifest = {"package": dict(name=name, version=VERSIONS[name], repository="https://github.com/Dicklesworthstone/" + repository,
                        edition={"workspace": True}, **{"license-file": "LICENSE"}),
                        "features": {"default": [], "extra": []},
                        "dependencies": {"serde": {"workspace": True, "features": ["alloc"]}}}
            if name == "frankentorch-kernel-cpu":
                manifest["dependencies"]["ft-core"] = {"workspace": True}
                manifest["lib"] = {"name": "ft_kernel_cpu"}
            if name in ("frankenjax-core", "frankenjax-lax"):
                manifest["lib"] = {"name": name.replace("frankenjax", "fj").replace("-", "_")}
                manifest["dev-dependencies"] = {"fj-test-utils": dict(package="frankenjax-test-utils", version="=0.1.0", path="../fj-test-utils")}
            if name == "frankenjax-lax":
                manifest["dependencies"]["fj-core"] = dict(package="frankenjax-core", version="=0.1.0", path="../fj-core")
                manifest["dependencies"]["mimalloc"] = dict(version="0.1", optional=True, **{"default-features": False})
                manifest["features"]["mimalloc-alloc"] = ["dep:mimalloc"]
            (directory / "Cargo.toml").write_text(packages.render(manifest))
            self.manifests[name] = manifest
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
        self.revision = self.git("rev-parse", "HEAD").decode().strip()
        for name, path in self.paths.items():
            self.specs.append(dict(name=name, version=VERSIONS[name], repository="https://github.com/Dicklesworthstone/" + repository,
                source_revision=self.revision, src_tree=self.git("rev-parse", f"HEAD:{path}/src").decode().strip()))
        self.contract = self.parent / "contract.json"
        self.contract.write_text(json.dumps(dict(schema=1, packages=self.specs)))
        return root

    def prepare(self, without=False):
        output = self.parent / "candidate"
        output.mkdir()
        report = {"commands": []}
        stage, workspace = packages.prepare(self.repo, output, self.family, self.specs, TOOLCHAIN,
                                            self.revision, without, report)
        return stage, output, workspace, report

    def cli(self, *extras):
        args = ["--repo", str(self.repo), "--family", self.family, "--expected-revision", self.revision,
                "--output", str(self.parent / "candidate"), "--toolchain-file", str(self.toolchain),
                "--contract", str(self.contract), *extras]
        with contextlib.redirect_stdout(io.StringIO()):
            return packages.main(args)

    def test_selects_oracle_helper_and_preserves_every_selected_byte(self):
        root = self.fixture()
        stage, _, workspace, report = self.prepare()
        self.assertEqual(report["publication_order"], ["frankenjax-test-utils", "frankenjax-core", "frankenjax-lax"])
        self.assertEqual(set(workspace["members"]), set(self.paths.values()))
        self.assertFalse((stage / "crates/unrelated").exists())
        self.assertFalse((stage / ".cargo").exists())
        self.assertFalse((stage / "Cargo.lock").exists())
        for path in self.paths.values():
            for source in (self.repo / path).rglob("*"):
                if source.is_file():
                    self.assertEqual(source.read_bytes(), (stage / source.relative_to(self.repo)).read_bytes())
        self.assertEqual(tomllib.loads((stage / "Cargo.toml").read_text())["profile"], root["profile"])
        self.assertEqual(self.git("status", "--porcelain"), b"")

    def test_override_requires_acknowledgement_and_is_recorded_not_applied(self):
        root = self.fixture("torch", with_patch=True)
        with self.assertRaisesRegex(archives.AuditError, "explicitly select"):
            self.prepare()
        output = self.parent / "acknowledged"
        output.mkdir()
        report = {}
        stage, _ = packages.prepare(self.repo, output, "torch", self.specs, TOOLCHAIN, self.revision, True, report)
        self.assertEqual(report["omitted_workspace_overrides"], {"patch": root["patch"]})
        self.assertNotIn("patch", tomllib.loads((stage / "Cargo.toml").read_text()))
        self.assertEqual(tomllib.loads((self.repo / "Cargo.toml").read_text()), root)

    def test_preparation_never_claims_build_or_publication(self):
        self.fixture()
        self.assertEqual(self.cli("--prepare-only"), 0)
        report = json.loads((self.parent / "candidate/dependency-package-receipt.json").read_text())
        self.assertEqual(report["status"], "prepared_only")
        self.assertEqual(report["unit_tests"], "not_run")
        self.assertEqual(report["registry_availability"], "not_checked")
        self.assertFalse(report["published"])
        self.assertFalse(report["commands"])

    def test_missing_helper_or_changed_version_is_rejected(self):
        self.fixture()
        for specs in (self.specs[1:], [dict(s, version="9.0.0") for s in self.specs]):
            with self.subTest(specs=specs):
                output = self.parent / str(len(list(self.parent.iterdir())))
                output.mkdir()
                with self.assertRaises(archives.AuditError):
                    packages.prepare(self.repo, output, self.family, specs, TOOLCHAIN, self.revision, False, {})

    def test_source_contract_drift_is_rejected(self):
        self.fixture()
        self.specs[0]["src_tree"] = "0" * 40
        with self.assertRaisesRegex(archives.AuditError, "source tree"):
            self.prepare()

    def test_wrong_revision_and_dirty_checkout_are_rejected(self):
        self.fixture()
        self.revision = "0" * 40
        self.assertEqual(self.cli("--prepare-only"), 1)
        self.revision = self.git("rev-parse", "HEAD").decode().strip()
        path = self.repo / next(iter(self.paths.values())) / "src/lib.rs"
        path.write_text("dirty")
        output = self.parent / "dirty"
        output.mkdir()
        with self.assertRaisesRegex(archives.AuditError, "tracked changes"):
            packages.prepare(self.repo, output, self.family, self.specs, TOOLCHAIN, self.revision, False, {})

    def test_tracked_symlink_is_rejected(self):
        self.fixture()
        path = self.repo / next(iter(self.paths.values())) / "tests/link"
        path.symlink_to(self.toolchain)
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=test@example.invalid", "commit", "-qm", "symlink")
        self.revision = self.git("rev-parse", "HEAD").decode().strip()
        with self.assertRaisesRegex(archives.AuditError, "unsupported publication input"):
            self.prepare()

    def test_target_specific_edges_and_inherited_feature_union(self):
        root = self.fixture()
        manifest = copy.deepcopy(self.manifests["frankenjax-core"])
        manifest["target"] = {'cfg(target_os = "macos")': {"dependencies": {
            "fj-lax": dict(package="frankenjax-lax", version="=0.1.0", path="../fj-lax", optional=True)}}}
        effective = packages.effective_dependencies(manifest, root["workspace"])
        self.assertEqual(effective[("", "dependencies", "serde")]["features"], ["alloc", "derive"])
        self.assertTrue(effective[('cfg(target_os = "macos")', "dependencies", "fj-lax")]["optional"])
        manifests = dict(self.manifests, **{"frankenjax-core": manifest})
        with self.assertRaisesRegex(archives.AuditError, "cycle"):
            packages.publication_order(manifests, root["workspace"], self.paths, {s["name"]: s for s in self.specs})

    def test_network_or_compiler_failure_produces_failed_receipt(self):
        self.fixture()
        real_run = subprocess.run
        def no_cargo(command, **kwargs):
            if command[0] == "git":
                return real_run(command, **kwargs)
            raise FileNotFoundError("Cargo fixture unavailable")
        with patch.object(packages.subprocess, "run", side_effect=no_cargo):
            self.assertEqual(self.cli("--cargo-command", "cargo"), 1)
        receipt = json.loads((self.parent / "candidate/dependency-package-receipt.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["package_verification"], "not_run")
        self.assertFalse(receipt["published"])

    def test_source_overriding_runner_is_rejected(self):
        self.fixture()
        stage, output, workspace, report = self.prepare()
        with self.assertRaisesRegex(archives.AuditError, "runner must"):
            packages.qualify(stage, output, self.family, self.specs, workspace,
                             ["cargo", "--config", 'patch.crates-io.hidden.path="../other"'], 1, 1, report)

    def test_inherited_kernel_override_is_rejected(self):
        self.fixture()
        stage, output, workspace, report = self.prepare()
        with patch.dict(os.environ, {"FT_METAL_SG_GEMM": "0"}), self.assertRaisesRegex(archives.AuditError, "kernel/behavior"):
            packages.qualify(stage, output, self.family, self.specs, workspace, ["cargo"], 1, 1, report)

    def test_snapshot_ignores_hidden_worktree_changes(self):
        self.fixture()
        relative = next(iter(self.paths.values())) + "/src/lib.rs"
        original = (self.repo / relative).read_bytes()
        self.git("update-index", "--assume-unchanged", relative)
        (self.repo / relative).write_text("not committed")
        stage, *_ = self.prepare()
        self.assertEqual((stage / relative).read_bytes(), original)

    def test_existing_evidence_and_ancestor_configuration_are_rejected(self):
        self.fixture()
        output = self.parent / "candidate"
        output.mkdir()
        sentinel = output / "dependency-package-receipt.json"
        sentinel.write_text("prior evidence")
        self.assertEqual(self.cli("--prepare-only"), 1)
        self.assertEqual(sentinel.read_text(), "prior evidence")
        (self.parent / ".cargo").mkdir()
        (self.parent / ".cargo/config").write_text("inherited")
        self.assertEqual(self.cli("--output", str(self.parent / "other"), "--prepare-only"), 1)
        self.assertFalse((self.parent / "other").exists())

    def test_internal_versions_external_paths_and_cycles_fail(self):
        root = self.fixture()
        specs = {s["name"]: s for s in self.specs}
        for mutation in (dict(version="0.1"), dict(path="../../outside"),
                         dict(package="frankenjax-lax", path="../fj-lax")):
            manifests = copy.deepcopy(self.manifests)
            manifests["frankenjax-lax"]["dependencies"]["fj-core"].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(archives.AuditError):
                packages.publication_order(manifests, root["workspace"], self.paths, specs)

    def pipeline(self, family="jax", fault=None, replay=False):
        self.fixture(family)
        stage, output, workspace, report = self.prepare()
        selected = {s["name"]: s for s in self.specs}
        if replay:
            (stage / "Cargo.lock").write_bytes(lock_bytes(selected))
        calls = []

        def fake(command, **kwargs):
            calls.append(command)
            self.assertEqual(command[0], "cargo")
            self.assertEqual(kwargs["cwd"], stage)
            self.assertEqual(kwargs["env"]["CARGO_HOME"], str(output / "cargo-home"))
            self.assertNotIn("CARGO_REGISTRIES_CRATES_IO_INDEX", kwargs["env"])
            if "metadata" in command:
                self.assertEqual("--locked" in command, replay)
                (stage / "Cargo.lock").write_bytes(lock_bytes(selected))
                locals_ = [dict(id=n, name=n, version=s["version"], source=None,
                           manifest_path=str(stage / self.paths[n] / "Cargo.toml")) for n, s in selected.items()]
                if fault == "git-source":
                    locals_.append(dict(id="foreign", name="other", version="1.0.0", source="git+https://example.org"))
                kwargs["stdout"].write(json.dumps(dict(packages=locals_, workspace_members=list(selected))).encode())
            if "test" in command and fault == "test-failed":
                raise subprocess.CalledProcessError(101, command)
            if "package" in command:
                self.assertIn("--workspace", command)
                self.assertIn("--all-features", command)
                self.assertIn("--locked", command)
                self.assertNotIn("--no-verify", command)
                out = output / "target/package"
                out.mkdir(parents=True)
                previous = {}
                for name in report["publication_order"]:
                    spec = selected[name]
                    manifest = copy.deepcopy(self.manifests[name])
                    manifest["package"]["edition"] = "2024"
                    for kind in packages.KINDS:
                        manifest.pop(kind, None)
                    for (target, kind, alias), dep in packages.effective_dependencies(self.manifests[name], workspace).items():
                        self.assertFalse(target)
                        manifest.setdefault(kind, {})[alias] = dep
                    contents = (stage / self.paths[name] / "src/lib.rs").read_bytes()
                    if fault == "source-drift":
                        contents += b"// changed"
                    if fault == "lost-feature":
                        manifest["features"]["extra"] = ["bad-feature"]
                    if fault == "changed-dependency":
                        manifest["dependencies"]["serde"]["features"] = []
                    if fault == "lib-name":
                        manifest["lib"] = dict(name="wrong_import")
                    lock = lock_bytes({name: spec}, previous)
                    if fault == "missing-lock-dependency":
                        lock = lock_bytes({name: spec})
                    if fault == "wrong-lock" and previous:
                        lock = lock.replace(next(iter(previous.values()))["archive_sha256"].encode(), b"0" * 64)
                    data = crate_bytes(name, spec["version"], {"Cargo.toml": packages.render(manifest).encode(),
                                       "Cargo.lock": lock, "src/lib.rs": contents})
                    if fault != "missing-archive":
                        (out / f"{name}-{spec['version']}.crate").write_bytes(data)
                    previous[name] = dict(version=spec["version"], archive_sha256=hashlib.sha256(data).hexdigest())
                if fault == "mutated-input":
                    (stage / "Cargo.lock").write_bytes(b"mutated")
            return subprocess.CompletedProcess(command, 0)

        with patch.dict(os.environ, {"CARGO_REGISTRIES_CRATES_IO_INDEX": "private"}), \
             patch.object(packages.subprocess, "run", side_effect=fake), \
             patch.object(archives, "audit", side_effect=AssertionError("must not wait for the packages to be published")):
            packages.qualify(stage, output, family, self.specs, workspace, ["cargo"], 2, 120, report)
        self.assertEqual(report["status"], "dependency_packages_verified")
        self.assertEqual(set(report["artifacts"]), set(selected))
        self.assertEqual(len([cmd for cmd in calls if "package" in cmd]), 1)
        self.assertTrue(all("--all-features" in c for c in calls if "test" in c or "package" in c))

    def test_unpublished_jax_family_can_be_packaged_in_one_invocation(self):
        self.pipeline()

    def test_unpublished_torch_family_can_be_packaged_in_one_invocation(self):
        self.pipeline("torch")

    def test_family_lock_can_be_replayed(self):
        self.pipeline(replay=True)

    def test_test_failure_is_not_package_success(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.pipeline(fault="test-failed")

    def test_changed_source_is_not_a_verified_package(self):
        with self.assertRaisesRegex(archives.AuditError, "source mismatch"):
            self.pipeline(fault="source-drift")

    def test_lost_optional_feature_is_rejected(self):
        with self.assertRaisesRegex(archives.AuditError, "changed feature"):
            self.pipeline(fault="lost-feature")

    def test_changed_inherited_dependency_features_are_rejected(self):
        with self.assertRaisesRegex(archives.AuditError, "changed dependencies"):
            self.pipeline(fault="changed-dependency")

    def test_changed_rust_library_name_is_rejected(self):
        with self.assertRaisesRegex(archives.AuditError, "library identity"):
            self.pipeline(fault="lib-name")

    def test_package_lock_must_refer_to_the_other_bundled_archive(self):
        with self.assertRaisesRegex(archives.AuditError, "different bundled implementation"):
            self.pipeline(fault="wrong-lock")

    def test_package_lock_cannot_omit_a_bundled_dependency(self):
        with self.assertRaisesRegex(archives.AuditError, "omits a bundled"):
            self.pipeline(fault="missing-lock-dependency")

    def test_git_dependency_cannot_enter_the_staged_closure(self):
        with self.assertRaisesRegex(archives.AuditError, "non-crates.io"):
            self.pipeline(fault="git-source")

    def test_missing_archive_is_not_success(self):
        with self.assertRaises(FileNotFoundError):
            self.pipeline(fault="missing-archive")

    def test_cargo_mutation_of_locked_inputs_is_rejected(self):
        with self.assertRaisesRegex(archives.AuditError, "changed the locked"):
            self.pipeline(fault="mutated-input")


if __name__ == "__main__":
    unittest.main()
