#!/usr/bin/env python3
"""Read-only publication gate for the initial cg-bms-jax v0.1.0 commit.

The scientific bundle has its own manifest validator in
``validate_release_scope.py``. This script protects the wider Git publication:
it constructs the candidate set from an explicit allowlist, rejects release
placeholders and likely secrets, checks README links, and can prove that the
Git index contains exactly the allowlisted set.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_ALLOWLIST = Path("release/publication_allowlist_v0.1.0.txt")
MAX_FILE_BYTES = 25 * 1024 * 1024
TEXT_SCAN_BYTES = 2 * 1024 * 1024

REQUIRED_FILES = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "README.md",
        "artifacts/release_v0.1.0/MANIFEST.json",
        "assets/manifest.yaml",
        "data/mb2d_equilibrium_exact_v1/endpoints.manifest.json",
        "data/mb2d_equilibrium_exact_v1/endpoints.npz",
        "docs/PUBLISHING_V0.1.0.md",
        "docs/RELEASE_RESULTS_V0.1.0.md",
        "environment.yml",
        "pyproject.toml",
        "release/publication_allowlist_v0.1.0.txt",
        "scripts/validate_publication_tree.py",
        "scripts/validate_release_scope.py",
    }
)
REQUIRED_TREES = (
    "src/",
    "tests/",
    "configs/experiment/mb",
    "configs/experiment/ala2_ambient18",
    "configs/experiment/ala2_cg",
    "artifacts/release_v0.1.0/figures/",
    "artifacts/release_v0.1.0/metrics/",
    "artifacts/release_v0.1.0/parameters/",
    "artifacts/release_v0.1.0/provenance/",
)

FORBIDDEN_TOP_LEVEL = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
        "logs",
        "outputs",
        "plots",
        "tmp",
    }
)
FORBIDDEN_ARCHIVE_SUFFIXES = frozenset({".tar", ".tgz", ".zip"})
FORBIDDEN_RUNTIME_SUFFIXES = frozenset(
    {".ckpt", ".log", ".orbax-checkpoint", ".pkl"}
)
ALL_ATOM_MARKERS = (
    "allatom",
    "all_atom",
    "all-atom",
    "fullatom",
    "full_atom",
    "full-atom",
    "ambient66",
    "ala2_aa",
)
ALL_ATOM_SCOPES = frozenset({"artifacts", "configs", "data", "docs", "scripts"})
PENDING_PATTERN = re.compile(
    r"\b(?:PENDING_VERIFIED(?:_[A-Z0-9_]+)?|TODO_RELEASE|TBD_RELEASE)\b"
)
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")

SECRET_PATTERNS = (
    (
        "private key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{24,}\b")),
)
TEXT_SUFFIXES = frozenset(
    {
        "",
        ".cff",
        ".cfg",
        ".csv",
        ".ini",
        ".json",
        ".md",
        ".py",
        ".rst",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)


class PublicationValidationError(RuntimeError):
    """Raised when the candidate publication violates the v0.1.0 contract."""


@dataclass(frozen=True)
class AllowRule:
    kind: str
    directory_or_path: str
    pattern: str | None = None

    def matches(self, relative: str) -> bool:
        if self.kind == "path":
            return relative == self.directory_or_path
        if self.kind == "tree":
            prefix = self.directory_or_path.rstrip("/") + "/"
            return relative.startswith(prefix)
        if self.kind == "glob":
            posix = PurePosixPath(relative)
            return (
                posix.parent.as_posix() == self.directory_or_path.rstrip("/")
                and self.pattern is not None
                and fnmatch.fnmatchcase(posix.name, self.pattern)
            )
        raise AssertionError(f"unknown rule kind: {self.kind}")


def _normalise(relative: str) -> str:
    return PurePosixPath(relative.replace("\\", "/")).as_posix()


def load_allowlist(path: Path) -> tuple[AllowRule, ...]:
    rules: list[AllowRule] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        kind = fields[0]
        if kind in {"path", "tree"} and len(fields) == 2:
            rules.append(AllowRule(kind, _normalise(fields[1])))
        elif kind == "glob" and len(fields) == 3:
            rules.append(AllowRule(kind, _normalise(fields[1]), fields[2]))
        else:
            raise PublicationValidationError(
                f"invalid allowlist rule at {path}:{line_number}: {raw_line!r}"
            )
    if not rules:
        raise PublicationValidationError(f"allowlist contains no rules: {path}")
    return tuple(rules)


def is_allowed(relative: str, rules: Iterable[AllowRule]) -> bool:
    return any(rule.matches(relative) for rule in rules)


def candidate_files(root: Path, rules: Iterable[AllowRule]) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(root).parts
        and is_allowed(path.relative_to(root).as_posix(), rules)
    }


def staged_files(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {
        _normalise(part.decode("utf-8"))
        for part in result.stdout.split(b"\0")
        if part
    }


def _read_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > TEXT_SCAN_BYTES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _validate_required(files: set[str]) -> list[str]:
    errors = [f"required publication file is missing: {path}" for path in sorted(REQUIRED_FILES - files)]
    for prefix in REQUIRED_TREES:
        if not any(path.startswith(prefix) for path in files):
            errors.append(f"required publication tree has no files: {prefix}")
    return errors


def _validate_paths(root: Path, files: set[str], rules: tuple[AllowRule, ...]) -> list[str]:
    errors: list[str] = []
    for relative in sorted(files):
        path = root / relative
        parts = PurePosixPath(relative).parts
        lowered = relative.lower()

        if not is_allowed(relative, rules):
            errors.append(f"file is outside the publication allowlist: {relative}")
        if parts and parts[0] in FORBIDDEN_TOP_LEVEL:
            errors.append(f"runtime/cache tree is forbidden: {relative}")
        if path.suffix.lower() in FORBIDDEN_ARCHIVE_SUFFIXES:
            errors.append(f"archive is forbidden from the initial commit: {relative}")
        if path.suffix.lower() in FORBIDDEN_RUNTIME_SUFFIXES:
            errors.append(f"runtime artifact is forbidden: {relative}")
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds the 25 MiB publication limit: {relative}")

        if parts and parts[0] in ALL_ATOM_SCOPES:
            marker = next((item for item in ALL_ATOM_MARKERS if item in lowered), None)
            if marker is not None:
                errors.append(
                    f"all-atom benchmark material ({marker}) is outside v0.1.0 scope: {relative}"
                )
        if parts and parts[0] == "artifacts" and not relative.startswith(
            "artifacts/release_v0.1.0/"
        ):
            errors.append(f"historical/probe artifact is forbidden: {relative}")
        if relative.startswith("data/") and not relative.startswith(
            "data/mb2d_equilibrium_exact_v1/"
        ):
            errors.append(f"non-frozen dataset is forbidden: {relative}")
    return errors


def _validate_text(root: Path, files: set[str]) -> list[str]:
    errors: list[str] = []
    release_facing = {
        path
        for path in files
        if path == "README.md"
        or path == "docs/RELEASE_RESULTS_V0.1.0.md"
        or path.startswith("artifacts/release_v0.1.0/")
    }
    for relative in sorted(files):
        path = root / relative
        text = _read_text(path)
        if text is None:
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible {label} in {relative}")
        if relative in release_facing:
            match = PENDING_PATTERN.search(text)
            if match is not None:
                errors.append(f"unresolved release placeholder {match.group(0)!r} in {relative}")
    return errors


def _validate_readme_links(root: Path, files: set[str]) -> list[str]:
    if "README.md" not in files:
        return ["README.md is not in the publication set"]
    text = (root / "README.md").read_text(encoding="utf-8")
    errors: list[str] = []
    for raw_target in MARKDOWN_LINK_PATTERN.findall(text):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if (
            not target
            or target.startswith(("#", "http://", "https://", "mailto:"))
            or "://" in target
        ):
            continue
        target = target.split("#", 1)[0].split("?", 1)[0]
        if not target:
            continue
        resolved = (root / target).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            errors.append(f"README link escapes the repository: {raw_target}")
            continue
        if not resolved.exists():
            errors.append(f"broken README link: {raw_target}")
    return errors


def validate(root: Path, allowlist: Path, mode: str) -> dict[str, object]:
    root = root.resolve()
    allowlist = allowlist if allowlist.is_absolute() else root / allowlist
    rules = load_allowlist(allowlist)
    candidates = candidate_files(root, rules)
    files = candidates if mode == "worktree" else staged_files(root)

    errors = _validate_required(files)
    errors.extend(_validate_paths(root, files, rules))
    errors.extend(_validate_text(root, files))
    errors.extend(_validate_readme_links(root, files))

    if mode == "staged":
        missing_from_index = sorted(candidates - files)
        extra_in_index = sorted(files - candidates)
        if missing_from_index:
            errors.append(
                "allowlisted files missing from the initial index: "
                + ", ".join(missing_from_index)
            )
        if extra_in_index:
            errors.append(
                "files staged outside the current allowlisted candidate set: "
                + ", ".join(extra_in_index)
            )

    if errors:
        raise PublicationValidationError("\n".join(errors))

    return {
        "mode": mode,
        "verification": "PASS",
        "file_count": len(files),
        "total_bytes": sum((root / relative).stat().st_size for relative in files),
        "allowlist": allowlist.relative_to(root).as_posix(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--mode", choices=("worktree", "staged"), default="worktree")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = validate(arguments.root, arguments.allowlist, arguments.mode)
    except (PublicationValidationError, OSError, subprocess.CalledProcessError) as exc:
        print(f"publication verification: FAIL\n{exc}", file=sys.stderr)
        return 1
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
