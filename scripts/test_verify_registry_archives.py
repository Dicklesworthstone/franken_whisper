#!/usr/bin/env python3
"""Offline regression tests for archive identity, drift and unsafe packaging."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import verify_registry_archives as audit

NAME = "frankentorch-core"
VERSION = "0.1.1"
REPOSITORY = "https://github.com/Dicklesworthstone/frankentorch"
MANIFEST = f'''[package]
name = "{NAME}"
version = "{VERSION}"
repository = "{REPOSITORY}"
[lib]
path = "src/lib.rs"
[dependencies]
half = "2.7"
'''.encode()
SOURCE = {"src/lib.rs": (0o644, b"pub fn answer() -> u32 { 42 }\n"),
          "src/nested/mod.rs": (0o644, b"// nested source\n"),
          "src/nested.rs": (0o644, b"// Git tree ordering sentinel\n")}


def archive(files=None, *, manifest=MANIFEST, extra_members=()):
    contents = {"Cargo.toml": (0o644, manifest), **(SOURCE if files is None else files)}
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as out:
        for path, (mode, content) in contents.items():
            member = tarfile.TarInfo(f"{NAME}-{VERSION}/{path}")
            member.mode, member.size = mode, len(content)
            out.addfile(member, io.BytesIO(content))
        for member, content in extra_members:
            out.addfile(member, io.BytesIO(content))
    return result.getvalue()


def record(data, **changes):
    result = {"name": NAME, "vers": VERSION, "yanked": False,
              "cksum": hashlib.sha256(data).hexdigest()}
    return {**result, **changes}


class ArchiveAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Independent implementation: Git itself supplies the expected tree ID.
        # The fixture includes both nested.rs and nested/ to check tree ordering.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path, (_, data) in SOURCE.items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            def git(*args):
                return subprocess.check_output(["git", "-C", directory, *args], stderr=subprocess.DEVNULL).decode().strip()
            git("init", "-q")
            git("add", "src")
            tree = git("write-tree")
            cls.expected_tree = git("rev-parse", f"{tree}:src")
        cls.spec = {"name": NAME, "version": VERSION, "repository": REPOSITORY,
                    "source_revision": "a" * 40, "src_tree": cls.expected_tree}

    def verify(self, data, spec=None):
        return audit.verify_archive(spec or self.spec, record(data), data)

    def test_git_tree_matches_independent_git_implementation(self):
        self.assertEqual(audit.source_tree(SOURCE), self.expected_tree)

    def test_exact_source_passes_but_does_not_claim_conformance(self):
        data = archive()
        def get(url):
            return json.dumps(record(data)).encode() if "index.crates.io" in url else data
        result = audit.audit([self.spec], get)
        self.assertEqual(result["artifact_integrity"], "pass")
        self.assertEqual(result["build_and_conformance"], "not_run")
        self.assertEqual(result["verified"][0]["src_tree"], self.expected_tree)

    def test_same_version_changed_source_is_rejected(self):
        changed = {**SOURCE, "src/lib.rs": (0o644, b"pub fn answer() -> u32 { 0 }\n")}
        with self.assertRaisesRegex(audit.AuditError, "source mismatch"):
            self.verify(archive(changed))

    def test_new_unpublished_module_is_detected(self):
        changed = {**SOURCE, "src/dynamic_int8_x86.rs": (0o644, b"// new kernel\n")}
        with self.assertRaisesRegex(audit.AuditError, "source mismatch"):
            self.verify(archive(changed))

    def test_missing_module_is_detected(self):
        changed = {k: v for k, v in SOURCE.items() if k != "src/nested/mod.rs"}
        with self.assertRaisesRegex(audit.AuditError, "source mismatch"):
            self.verify(archive(changed))

    def test_file_mode_is_part_of_source_identity(self):
        changed = {**SOURCE, "src/lib.rs": (0o755, SOURCE["src/lib.rs"][1])}
        with self.assertRaisesRegex(audit.AuditError, "source mismatch"):
            self.verify(archive(changed))

    def test_transport_checksum_mismatch(self):
        data = archive()
        with self.assertRaisesRegex(audit.AuditError, "checksum differs"):
            audit.verify_archive(self.spec, record(data, cksum="0" * 64), data)

    def test_missing_exact_version_does_not_fall_back(self):
        data = json.dumps(record(b"", vers="0.1.0")).encode()
        with self.assertRaisesRegex(audit.AuditError, "publication required"):
            audit.index_record(data, NAME, VERSION)

    def test_unrelated_package_identity_is_rejected(self):
        data = archive(manifest=MANIFEST.replace(NAME.encode(), b"fj-core"))
        with self.assertRaisesRegex(audit.AuditError, "identity mismatch"):
            self.verify(data)

    def test_repository_identity_is_checked(self):
        data = archive(manifest=MANIFEST.replace(REPOSITORY.encode(), b"https://github.com/other/project"))
        with self.assertRaisesRegex(audit.AuditError, "repository mismatch"):
            self.verify(data)

    def test_yanked_and_invalid_index_entries(self):
        for change in ({"yanked": True}, {"yanked": None}, {"cksum": "invalid"}):
            with self.subTest(change=change), self.assertRaises(audit.AuditError):
                audit.index_record(json.dumps(record(b"", **change)).encode(), NAME, VERSION)

    def test_duplicate_index_version_is_rejected(self):
        body = json.dumps(record(b"")).encode()
        with self.assertRaisesRegex(audit.AuditError, "ambiguous"):
            audit.index_record(body + b"\n" + body, NAME, VERSION)

    def test_optional_target_specific_git_dependency_is_not_skipped(self):
        extra = b'''\n[target.'cfg(target_os = "macos")'.dependencies]
block = { git = "https://github.com/example/block", optional = true }
'''
        with self.assertRaisesRegex(audit.AuditError, "non-registry dependency"):
            self.verify(archive(manifest=MANIFEST + extra))

    def test_path_build_dependency_is_rejected(self):
        extra = b'\n[build-dependencies]\nhelper = { path = "../helper" }\n'
        with self.assertRaisesRegex(audit.AuditError, "non-registry dependency"):
            self.verify(archive(manifest=MANIFEST + extra))

    def test_new_build_script_requires_separate_review(self):
        changed = {**SOURCE, "build.rs": (0o644, b"fn main() {}\n")}
        with self.assertRaisesRegex(audit.AuditError, "unreviewed build input"):
            self.verify(archive(changed))
        spec = {**self.spec, "extra_files": {"build.rs": audit.git_object("blob", changed["build.rs"][1]).hex()}}
        self.verify(archive(changed), spec)

    def test_external_library_entry_point_cannot_escape_source_attestation(self):
        changed = {**SOURCE, "other.rs": (0o644, b"// different library\n")}
        manifest = MANIFEST.replace(b"src/lib.rs", b"other.rs")
        with self.assertRaisesRegex(audit.AuditError, "unreviewed build input"):
            self.verify(archive(changed, manifest=manifest))

    def test_unsafe_members_and_links_are_rejected_without_extraction(self):
        for name in (f"{NAME}-{VERSION}/../escape", "/absolute", f"{NAME}-{VERSION}/src/./lib.rs"):
            member = tarfile.TarInfo(name)
            with self.subTest(name=name), self.assertRaisesRegex(audit.AuditError, "unsafe archive"):
                self.verify(archive(extra_members=[(member, b"")]))
        member = tarfile.TarInfo(f"{NAME}-{VERSION}/src/link")
        member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
        with self.assertRaisesRegex(audit.AuditError, "non-regular"):
            self.verify(archive(extra_members=[(member, b"")]))

    def test_duplicate_archive_member_is_rejected(self):
        member = tarfile.TarInfo(f"{NAME}-{VERSION}/src/lib.rs")
        with self.assertRaisesRegex(audit.AuditError, "duplicate archive"):
            self.verify(archive(extra_members=[(member, b"")]))

    def test_unpacked_resource_limit_is_enforced(self):
        with patch.object(audit, "MAX_UNPACKED", 10), self.assertRaisesRegex(audit.AuditError, "size limit"):
            self.verify(archive())

    def test_network_failure_is_not_readiness(self):
        def get(_):
            raise audit.AuditError("registry unavailable")
        result = audit.audit([self.spec], get)
        self.assertEqual(result["artifact_integrity"], "fail")
        self.assertFalse(result["verified"])
        self.assertEqual(len(result["errors"]), 1)

    def test_audit_reports_all_missing_packages(self):
        result = audit.audit([self.spec, {**self.spec, "name": "fj-lax"}], lambda _: b"")
        self.assertEqual(len(result["errors"]), 2)
        self.assertEqual(result["artifact_integrity"], "fail")

    def test_index_sharding_and_invalid_names(self):
        for name, path in (("a", "1/a"), ("ab", "2/ab"), ("AbC", "3/a/abc"), ("serde", "se/rd/serde")):
            self.assertEqual(audit.index_path(name), path)
        with self.assertRaises(audit.AuditError):
            audit.index_path("../escape")

    def test_contract_is_nonempty_unique_and_exact(self):
        cases = [{"schema": 1, "packages": []}, {"schema": 1, "packages": [self.spec, self.spec]},
                 {"schema": 1, "packages": [{**self.spec, "version": "^0.1"}]}]
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "contract.json"
            for invalid in cases:
                file.write_text(json.dumps(invalid))
                with self.subTest(invalid=invalid), self.assertRaises(audit.AuditError):
                    audit.load_contract(file)
            file.write_text(json.dumps({"schema": 1, "packages": [self.spec]}))
            self.assertEqual(audit.load_contract(file), [self.spec])


if __name__ == "__main__":
    unittest.main()
