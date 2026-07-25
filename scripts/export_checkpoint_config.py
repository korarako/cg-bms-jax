#!/usr/bin/env python
"""Export the immutable experiment config embedded in a checkpoint manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def export_checkpoint_config(
    checkpoint: str | Path,
    output: str | Path,
) -> Path:
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{manifest_path} does not contain an experiment config")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    print(export_checkpoint_config(arguments.checkpoint, arguments.output))


if __name__ == "__main__":
    main()
