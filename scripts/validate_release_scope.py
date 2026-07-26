#!/usr/bin/env python3
"""Validate the scientific and file boundary of a frozen v0.1 result bundle.

This gate intentionally validates ``artifacts/release_v0.1.0`` rather than the
source distribution.  Experimental all-atom code may remain in the repository,
but no all-atom result, raw trajectory, checkpoint, log, or cache is allowed in
the frozen benchmark bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_FAMILIES = frozenset({"mb_cg1d", "mb2d_analytic", "ala2_cg"})
REQUIRED_EXCLUSION = "ala2_all_atom"

FORBIDDEN_PATH_MARKERS = (
    "allatom",
    "all_atom",
    "all-atom",
    "full_atom",
    "full-atom",
    "openmm",
    "ambient66",
    "ala2_aa",
)
FORBIDDEN_TOKEN_PATTERNS = (
    ("66d", re.compile(r"(?<![0-9a-z])66d(?![0-9a-z])", re.IGNORECASE)),
)
FORBIDDEN_SUFFIXES = (
    ".ckpt",
    ".log",
    ".npy",
    ".npz",
    ".orbax-checkpoint",
    ".pkl",
    ".tar",
    ".tgz",
    ".zip",
)
MACHINE_METADATA_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_BUNDLE_BYTES = 250 * 1024 * 1024


class ScopeValidationError(RuntimeError):
    """Raised when a release bundle violates the frozen-result contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_string_set(value: Any, field: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ScopeValidationError(
            f"MANIFEST.json field {field!r} must be a list of strings"
        )
    return {item.strip().lower() for item in value}


def _release_scope(manifest: dict[str, Any]) -> tuple[set[str], set[str]]:
    scope = manifest.get("release_scope")
    if not isinstance(scope, dict):
        raise ScopeValidationError(
            "MANIFEST.json must contain an object named 'release_scope'"
        )

    included = _normalise_string_set(
        scope.get("included_benchmark_families"),
        "release_scope.included_benchmark_families",
    )
    excluded = _normalise_string_set(
        scope.get("excluded_benchmark_families"),
        "release_scope.excluded_benchmark_families",
    )
    return included, excluded


def _manifest_files(manifest: dict[str, Any]) -> dict[str, str]:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ScopeValidationError("MANIFEST.json field 'files' must be a list")

    result: dict[str, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ScopeValidationError(
                f"MANIFEST.json files[{index}] must be an object"
            )
        raw_path = record.get("path")
        raw_sha = record.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ScopeValidationError(
                f"MANIFEST.json files[{index}].path must be a non-empty string"
            )
        if not isinstance(raw_sha, str) or len(raw_sha) != 64:
            raise ScopeValidationError(
                f"MANIFEST.json files[{index}].sha256 must be a 64-character digest"
            )

        posix = PurePosixPath(raw_path.replace("\\", "/"))
        if posix.is_absolute() or ".." in posix.parts:
            raise ScopeValidationError(f"unsafe manifest path: {raw_path!r}")
        canonical = posix.as_posix()
        if canonical == "MANIFEST.json":
            raise ScopeValidationError("MANIFEST.json must not attempt to hash itself")
        if canonical in result:
            raise ScopeValidationError(f"duplicate manifest path: {canonical}")
        result[canonical] = raw_sha.lower()
    return result


def _contains_marker(value: str) -> str | None:
    lowered = value.lower()
    substring = next(
        (marker for marker in FORBIDDEN_PATH_MARKERS if marker in lowered),
        None,
    )
    if substring is not None:
        return substring
    return next(
        (label for label, pattern in FORBIDDEN_TOKEN_PATTERNS if pattern.search(value)),
        None,
    )


def _validate_paths(root: Path, files: list[Path]) -> None:
    total_bytes = 0
    family_evidence: dict[str, int] = {family: 0 for family in EXPECTED_FAMILIES}
    errors: list[str] = []

    for path in files:
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        size = path.stat().st_size
        total_bytes += size

        marker = _contains_marker(relative)
        if marker is not None:
            errors.append(
                f"forbidden all-atom result marker {marker!r} in path {relative}"
            )

        if lowered.endswith(FORBIDDEN_SUFFIXES):
            errors.append(
                f"raw/runtime artifact is not allowed in the frozen bundle: {relative}"
            )
        if size > MAX_FILE_BYTES:
            errors.append(
                f"file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MiB limit: {relative}"
            )

        for family in EXPECTED_FAMILIES:
            if family in lowered:
                family_evidence[family] += 1

        if (
            path.suffix.lower() in MACHINE_METADATA_SUFFIXES
            and relative != "MANIFEST.json"
            and any(
                part in {"metrics", "parameters", "provenance"}
                for part in path.relative_to(root).parts
            )
        ):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"machine metadata is not UTF-8: {relative}")
            else:
                content_marker = _contains_marker(text)
                if content_marker is not None:
                    errors.append(
                        f"forbidden all-atom result marker {content_marker!r} in machine metadata {relative}"
                    )

    if total_bytes > MAX_BUNDLE_BYTES:
        errors.append(
            f"bundle exceeds {MAX_BUNDLE_BYTES // (1024 * 1024)} MiB total-size limit"
        )
    for family, count in family_evidence.items():
        if count == 0:
            errors.append(
                f"bundle has no path-level evidence for required family {family!r}"
            )

    if errors:
        raise ScopeValidationError("\n".join(errors))


def validate_release_bundle(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ScopeValidationError(
            f"release bundle does not exist or is not a directory: {root}"
        )

    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ScopeValidationError(f"missing release manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScopeValidationError(f"invalid UTF-8 JSON manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ScopeValidationError("MANIFEST.json root must be an object")

    included, excluded = _release_scope(manifest)
    if included != EXPECTED_FAMILIES:
        raise ScopeValidationError(
            "release_scope.included_benchmark_families must equal "
            f"{sorted(EXPECTED_FAMILIES)}, got {sorted(included)}"
        )
    if REQUIRED_EXCLUSION not in excluded:
        raise ScopeValidationError(
            "release_scope.excluded_benchmark_families must include "
            f"{REQUIRED_EXCLUSION!r}"
        )

    actual_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != manifest_path
    )
    if not actual_files:
        raise ScopeValidationError(
            "release bundle contains no files besides MANIFEST.json"
        )
    _validate_paths(root, actual_files)

    recorded = _manifest_files(manifest)
    actual = {path.relative_to(root).as_posix(): _sha256(path) for path in actual_files}
    missing = sorted(set(actual) - set(recorded))
    stale = sorted(set(recorded) - set(actual))
    mismatched = sorted(
        path for path in set(actual) & set(recorded) if actual[path] != recorded[path]
    )
    errors: list[str] = []
    if missing:
        errors.append(f"files missing from MANIFEST.json: {missing}")
    if stale:
        errors.append(f"manifest entries without files: {stale}")
    if mismatched:
        errors.append(f"SHA-256 mismatches: {mismatched}")
    if errors:
        raise ScopeValidationError("\n".join(errors))

    return {
        "root": str(root),
        "included_benchmark_families": sorted(included),
        "excluded_benchmark_families": sorted(excluded),
        "file_count": len(actual_files),
        "total_bytes": sum(path.stat().st_size for path in actual_files),
        "verification": "PASS",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle",
        nargs="?",
        type=Path,
        default=Path("artifacts/release_v0.1.0"),
        help="frozen result bundle (default: artifacts/release_v0.1.0)",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = validate_release_bundle(arguments.bundle)
    except ScopeValidationError as exc:
        print(f"release scope verification: FAIL\n{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
