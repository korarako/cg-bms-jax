"""Atomic Orbax storage with immutable scientific metadata and checksums."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import secrets
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp

from cg_bms_jax.checkpoint.types import CheckpointMetadata
from cg_bms_jax.training.state import ControllerTrainState

_MANIFEST = "manifest.json"
_DIGEST = "sha256.txt"
_TREE = "pytree"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RestoredCheckpoint:
    """Restored controller payload plus JSON configuration and provenance."""

    params: Any
    constants: Any
    optimizer_state: Any
    step: Any
    config: dict[str, Any]
    metadata: CheckpointMetadata
    sha256: str

    def as_train_state(self) -> ControllerTrainState:
        return ControllerTrainState(
            params=self.params,
            constants=self.constants,
            optimizer_state=self.optimizer_state,
            step=self.step,
        )


@dataclass(frozen=True)
class RestoredControllerCheckpoint:
    """Controller-only restore result for parameter initialization.

    Optimizer slots and the saved training step are intentionally absent from
    this type, so callers cannot accidentally turn initialization into resume.
    """

    params: Any
    constants: Any
    config: dict[str, Any]
    metadata: CheckpointMetadata
    sha256: str

    @property
    def variables(self) -> dict[str, Any]:
        return {"params": self.params, **dict(self.constants)}


def _jsonable(value: Any) -> Any:
    """Convert common config containers to a deterministic JSON value."""

    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # OmegaConf containers expose a plain primitive representation through
    # ``to_container``; keep OmegaConf optional at import time.
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            return _jsonable(OmegaConf.to_container(value, resolve=True))
    except ImportError:
        pass
    raise TypeError(f"configuration value of type {type(value).__name__} is not JSON serializable")


def canonical_config_sha256(config: Any) -> str:
    """Return the SHA256 used by ``CheckpointMetadata.config_sha256``."""

    payload = json.dumps(
        _jsonable(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_file(hasher: Any, path: Path, relative: str) -> None:
    encoded = relative.replace(os.sep, "/").encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)


def checkpoint_sha256(path: str | os.PathLike[str]) -> str:
    """Hash every checkpoint file except the digest file itself.

    Relative file names are included in the digest, making silent additions,
    deletions and renames detectable in addition to content changes.
    """

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.relative_to(root).as_posix() != _DIGEST
    )
    hasher = hashlib.sha256()
    for file_path in files:
        _hash_file(hasher, file_path, file_path.relative_to(root).as_posix())
    return hasher.hexdigest()


def verify_checkpoint(path: str | os.PathLike[str]) -> str:
    """Verify a checkpoint digest and return it, or raise on corruption."""

    root = Path(path)
    digest_path = root / _DIGEST
    if not digest_path.is_file():
        raise FileNotFoundError(f"checkpoint digest is missing: {digest_path}")
    expected = digest_path.read_text(encoding="ascii").strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("checkpoint sha256.txt is malformed")
    actual = checkpoint_sha256(root)
    if not secrets.compare_digest(expected, actual):
        raise ValueError(f"checkpoint checksum mismatch: expected {expected}, computed {actual}")
    return actual


def _payload_from_state(state: ControllerTrainState) -> dict[str, Any]:
    return {
        "params": state.params,
        "constants": state.constants,
        "optimizer_state": state.optimizer_state,
        "step": state.step,
    }


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    state: ControllerTrainState,
    config: Any,
    metadata: CheckpointMetadata,
) -> str:
    """Save a complete controller checkpoint using staging-directory rename.

    Existing destinations are rejected so a published checkpoint can never be
    partially overwritten.  Callers should use monotonically named step
    directories and update a separate lightweight "latest" pointer if needed.
    """

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_json = _jsonable(config)
    config_digest = canonical_config_sha256(config_json)
    if metadata.config_sha256 != config_digest:
        raise ValueError(
            "metadata.config_sha256 does not match the canonical saved config: "
            f"{metadata.config_sha256} != {config_digest}"
        )
    state_step = int(jax.device_get(state.step))
    if metadata.global_step != state_step:
        raise ValueError(
            f"metadata.global_step ({metadata.global_step}) does not match state.step ({state_step})"
        )
    staging = destination.parent / f".{destination.name}.tmp-{secrets.token_hex(8)}"
    staging.mkdir()
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        tree_path = staging / _TREE
        checkpointer.save(str(tree_path.resolve()), _payload_from_state(state))
        manifest = {
            "format": "cg-bms-jax-orbax",
            "format_version": _FORMAT_VERSION,
            "metadata": metadata.to_dict(),
            "config": config_json,
        }
        (staging / _MANIFEST).write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        digest = checkpoint_sha256(staging)
        (staging / _DIGEST).write_text(digest + "\n", encoding="ascii")
        # Rename on one filesystem is atomic.  fsync the parent directory on
        # POSIX so the directory entry survives a sudden power loss.
        os.replace(staging, destination)
        if os.name == "posix":
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return digest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        close = getattr(checkpointer, "close", None)
        if close is not None:
            close()


def restore_checkpoint(
    path: str | os.PathLike[str],
    *,
    target_state: ControllerTrainState | None = None,
    verify: bool = True,
) -> RestoredCheckpoint:
    """Restore a checkpoint, optionally using a typed target PyTree.

    Passing an initialized ``target_state`` is recommended when continuing
    training because it guarantees that custom Optax tuple types are restored
    exactly.  Target-free restore is sufficient for inference parameters.
    """

    root = Path(path).expanduser().resolve()
    digest = verify_checkpoint(root) if verify else checkpoint_sha256(root)
    manifest_path = root / _MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "cg-bms-jax-orbax":
        raise ValueError("unsupported checkpoint format")
    if manifest.get("format_version") != _FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format version: {manifest.get('format_version')}")

    target = _payload_from_state(target_state) if target_state is not None else None
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        if target is None:
            payload = checkpointer.restore(str((root / _TREE).resolve()))
        else:
            restore_args = jax.tree_util.tree_map(
                lambda value: ocp.ArrayRestoreArgs(sharding=value.sharding),
                target,
            )
            payload = checkpointer.restore(
                str((root / _TREE).resolve()),
                item=target,
                restore_args=restore_args,
            )
    finally:
        close = getattr(checkpointer, "close", None)
        if close is not None:
            close()
    required = {"params", "constants", "optimizer_state", "step"}
    if set(payload) != required:
        raise ValueError(f"checkpoint PyTree fields are invalid: {sorted(payload)}")
    metadata = CheckpointMetadata.from_dict(manifest["metadata"])
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a JSON object")
    config_digest = canonical_config_sha256(config)
    if metadata.config_sha256 != config_digest:
        raise ValueError("checkpoint metadata config SHA256 does not match manifest config")
    return RestoredCheckpoint(
        params=payload["params"],
        constants=payload["constants"],
        optimizer_state=payload["optimizer_state"],
        step=payload["step"],
        config=config,
        metadata=metadata,
        sha256=digest,
    )


def restore_controller_only(
    path: str | os.PathLike[str],
    *,
    target_variables: Mapping[str, Any],
    verify: bool = True,
) -> RestoredControllerCheckpoint:
    """Restore only Flax ``params`` and non-trainable collections.

    This is an initialization boundary, not a resume API.  Orbax receives a
    two-field target PyTree, so optimizer state and the saved step are not
    deserialized.  The checkpoint checksum and manifest/config binding are
    still verified before any controller is returned.
    """

    if "params" not in target_variables:
        raise KeyError("target_variables must contain a 'params' collection")
    root = Path(path).expanduser().resolve()
    digest = verify_checkpoint(root) if verify else checkpoint_sha256(root)
    manifest_path = root / _MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "cg-bms-jax-orbax":
        raise ValueError("unsupported checkpoint format")
    if manifest.get("format_version") != _FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version: {manifest.get('format_version')}"
        )
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a JSON object")
    metadata = CheckpointMetadata.from_dict(manifest["metadata"])
    if metadata.config_sha256 != canonical_config_sha256(config):
        raise ValueError("checkpoint metadata config SHA256 does not match manifest config")

    target = {
        "params": target_variables["params"],
        "constants": {
            name: value for name, value in target_variables.items() if name != "params"
        },
    }
    restore_args = jax.tree_util.tree_map(
        lambda value: ocp.ArrayRestoreArgs(sharding=value.sharding),
        target,
    )
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        payload = checkpointer.restore(
            str((root / _TREE).resolve()),
            item=target,
            restore_args=restore_args,
            # An explicit (empty) transform tree selects only keys present in
            # ``item``.  Without it Orbax's fast path requires the complete
            # saved tree and would also deserialize optimizer_state/step.
            transforms={},
        )
    finally:
        close = getattr(checkpointer, "close", None)
        if close is not None:
            close()
    if not isinstance(payload, Mapping) or set(payload) != {"params", "constants"}:
        fields = sorted(payload) if isinstance(payload, Mapping) else type(payload).__name__
        raise ValueError(f"controller-only checkpoint fields are invalid: {fields}")
    if not isinstance(payload["constants"], Mapping):
        raise TypeError("restored controller constants must be a mapping")
    return RestoredControllerCheckpoint(
        params=payload["params"],
        constants=payload["constants"],
        config=config,
        metadata=metadata,
        sha256=digest,
    )
