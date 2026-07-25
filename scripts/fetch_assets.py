#!/usr/bin/env python3
"""Download pinned CG-BG assets and verify their content hashes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "assets" / "manifest.yaml")
    parser.add_argument("--cache", type=Path, default=root / "assets" / "cache")
    parser.add_argument("--experiment", choices=("mb_cg1d", "ala2_ambient18_300k"))
    parser.add_argument("--all", action="store_true", help="Fetch both supported experiments")
    args = parser.parse_args()
    if not args.all and args.experiment is None:
        parser.error("select --experiment or --all")

    manifest = yaml.safe_load(args.manifest.read_text())
    args.cache.mkdir(parents=True, exist_ok=True)
    selected = []
    for logical_id, spec in manifest["files"].items():
        if args.all or spec["experiment"] == args.experiment:
            selected.append((logical_id, spec))

    for logical_id, spec in selected:
        downloaded = Path(
            hf_hub_download(
                repo_id=manifest["repo_id"],
                repo_type=manifest["repo_type"],
                revision=manifest["revision"],
                filename=spec["path"],
                local_dir=args.cache,
            )
        )
        actual_size = downloaded.stat().st_size
        actual_hash = sha256(downloaded)
        if actual_size != int(spec["size_bytes"]):
            raise RuntimeError(f"{logical_id}: size {actual_size} != {spec['size_bytes']}")
        if actual_hash != spec["sha256"]:
            raise RuntimeError(f"{logical_id}: sha256 {actual_hash} != {spec['sha256']}")
        print(f"verified {logical_id}: {downloaded}")


if __name__ == "__main__":
    main()

