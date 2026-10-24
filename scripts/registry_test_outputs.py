"""Separate reviewed test evidence from immutable publication inputs.

Known evidence files stay mandatory. Additional JSON evidence is accepted only
inside source-tree-bound output roots used by the reviewed JAX test harness.
Every pre-existing input remains immutable and every other addition still fails.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from verify_registry_archives import AuditError

# fj-lax/src/lib.rs: e2e_bitwise_reductions_oracle,
# e2e_special_functions_oracle and e2e_special_function_error_paths_graceful.
# Match the emitter's source identity, not just a reusable package name.
REVIEWED_EMITTERS = {
    ("frankenjax-lax", "0.1.0", "7aeb83da4ffa461be93aff082643eb659a144867"): (
        "artifacts/e2e/e2e_bitwise_reductions.e2e.json",
        "artifacts/e2e/e2e_special_functions.e2e.json",
        "artifacts/e2e/e2e_special_function_error_paths.e2e.json",
        "artifacts/testing/logs/fj-lax/e2e_bitwise_reductions_oracle.json",
        "artifacts/testing/logs/fj-lax/e2e_special_functions_oracle.json",
        "artifacts/testing/logs/fj-lax/e2e_special_function_error_paths_graceful.json",
    ),
}
REVIEWED_OUTPUT_ROOTS = {
    ("frankenjax-core", "0.1.0", "de700ac07837961002f6bc1f256e0fada64a0ee6"): (
        "artifacts/testing/logs/fj-core/",
    ),
    ("frankenjax-lax", "0.1.0", "7aeb83da4ffa461be93aff082643eb659a144867"): (
        "artifacts/e2e/",
        "artifacts/testing/logs/fj-lax/",
    ),
}
MAX_EVIDENCE_BYTES = 8 * 1024 * 1024
MAX_EVIDENCE_FILES = 4096
MAX_TOTAL_EVIDENCE_BYTES = 64 * 1024 * 1024


def output_paths(specs: dict[str, dict]) -> tuple[str, ...]:
    result = set()
    for name, spec in specs.items():
        key = (name, spec["version"], spec["src_tree"])
        result.update(REVIEWED_EMITTERS.get(key, ()))
    return tuple(sorted(result))


def output_roots(specs: dict[str, dict]) -> tuple[str, ...]:
    result = set()
    for name, spec in specs.items():
        key = (name, spec["version"], spec["src_tree"])
        result.update(REVIEWED_OUTPUT_ROOTS.get(key, ()))
    return tuple(sorted(result))


def _valid_output_path(name: str, roots: tuple[str, ...] = ()) -> bool:
    path = PurePosixPath(name)
    if (path.is_absolute() or path.as_posix() != name or ".." in path.parts
            or not name.startswith("artifacts/") or path.suffix != ".json"
            or any(c in name for c in "\\:*?[]")
            or any(ord(c) < 32 for c in name)
            or any(p.startswith(".") for p in path.parts)):
        return False
    if roots and not any(name.startswith(root) for root in roots):
        return False
    if name.startswith("artifacts/e2e/") and not name.endswith(".e2e.json"):
        return False
    return True


def require_fresh_outputs(stage: Path, baseline: dict, paths: tuple[str, ...],
                          roots: tuple[str, ...] = ()) -> None:
    """An output must never exempt an existing golden, source or lock input."""
    for name in paths:
        path = PurePosixPath(name)
        if not _valid_output_path(name):
            raise AuditError(f"unsafe test output path: {name!r}")
        for parent in path.parents:
            if parent.as_posix() in baseline:
                raise AuditError(f"test output parent is an input file: {name}")
        target = stage / name
        if name in baseline or target.exists() or target.is_symlink():
            raise AuditError(f"test output already exists or overlaps an input: {name}")


def collect_outputs(stage: Path, output: Path, before: dict, after: dict,
                    paths: tuple[str, ...], report: dict[str, Any],
                    roots: tuple[str, ...] = ()) -> None:
    """Accept only fresh declared JSON outputs; reject every input mutation.

    The caller must obtain both inventories with its symlink-rejecting scanner.
    Copies are retained separately for CI upload. Nothing is deleted, rewritten,
    or hidden in the tested workspace. Packaging must freeze the post-test tree.
    """
    report["test_input_integrity"] = "failed"
    required = set(paths)
    if required & before.keys():
        raise AuditError("test output overlaps immutable inputs")
    added = after.keys() - before.keys()
    removed = before.keys() - after.keys()
    modified = {p for p in before.keys() & after.keys() if before[p] != after[p]}
    dynamic = {p for p in added if p not in required and _valid_output_path(p, roots)}
    allowed = required | dynamic
    unexpected = added - allowed
    missing = required - added
    delta = {"added": sorted(added), "removed": sorted(removed),
             "modified": sorted(modified), "unexpected": sorted(unexpected),
             "missing_outputs": sorted(missing)}
    report["test_workspace_delta"] = delta
    if removed or modified or unexpected or missing:
        raise AuditError("tests changed the locked source inputs or omitted reviewed outputs: "
                         + json.dumps(delta, sort_keys=True))

    # Validate everything before copying. A bounded, regular JSON file is data,
    # not a new build input or a symlink to material outside this workspace.
    if len(allowed) > MAX_EVIDENCE_FILES:
        raise AuditError("test evidence contains too many files")
    contents = {}
    total_bytes = 0
    for name in sorted(allowed):
        path = stage / name
        for parent in (path, *path.parents):
            if parent == stage:
                break
            if parent.is_symlink():
                raise AuditError(f"test evidence traverses a symlink: {name}")
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o111
                or info.st_nlink != 1 or info.st_size > MAX_EVIDENCE_BYTES):
            raise AuditError(f"test evidence is not bounded regular JSON data: {name}")
        data = path.read_bytes()
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_EVIDENCE_BYTES:
            raise AuditError("test evidence exceeds total size limit")
        digest = hashlib.sha256(data).hexdigest()
        if digest != after[name][0] or info.st_mode & 0o111 != after[name][1]:
            raise AuditError(f"test evidence changed during collection: {name}")
        try:
            parsed = json.loads(data)
        except (ValueError, UnicodeError) as exc:
            raise AuditError(f"invalid JSON test evidence: {name}") from exc
        if not isinstance(parsed, (dict, list)):
            raise AuditError(f"expected structured JSON test evidence: {name}")
        contents[name] = (data, digest)

    records = {}
    if contents:
        destination = output / "test-evidence"
        destination.mkdir(exist_ok=False)
        for name, (data, digest) in contents.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
            records[name] = {"sha256": digest, "bytes": len(data),
                             "artifact": target.relative_to(output).as_posix()}
    report.update(test_input_integrity="pass", workspace_test_outputs=records)
