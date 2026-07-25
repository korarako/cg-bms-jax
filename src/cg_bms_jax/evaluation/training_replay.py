"""Upstream-BMS-style visual diagnostics for forward replay endpoints.

The public BMS trainer evaluates ``buffer.storage['data_1']`` every configured
number of epochs. These are lagged training endpoints, not fresh samples from
the just-saved controller. This module preserves that distinction explicitly
while producing the local CG-BG-style figures used by this project.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.data.buffer import ReplayBufferState
from cg_bms_jax.runtime import RuntimeSystem

from .ala2 import evaluate_ala2


def load_evaluation_archive(path: str | Path) -> dict[str, np.ndarray]:
    """Load only fields consumed by the unweighted Ala2 evaluator."""

    with np.load(Path(path), allow_pickle=False) as archive:
        if "R" not in archive.files:
            raise KeyError(f"Evaluation reference {path} does not contain R")
        result = {"R": np.asarray(archive["R"])}
        if "U" in archive.files:
            result["U"] = np.asarray(archive["U"])
    return result


def replay_endpoint_states(
    replay: ReplayBufferState,
    *,
    max_samples: int | None,
) -> np.ndarray:
    """Return valid endpoint_1 states in deterministic ring-buffer order."""

    if "endpoint_1" not in replay.storage:
        raise KeyError("Forward replay buffer does not contain endpoint_1")
    size = int(jax.device_get(replay.size))
    capacity = replay.capacity
    if size <= 0:
        raise ValueError("Cannot visualize an empty forward replay buffer")
    cursor = int(jax.device_get(replay.cursor))
    if size < capacity:
        order = np.arange(size, dtype=np.int64)
    else:
        order = np.concatenate(
            (
                np.arange(cursor, capacity, dtype=np.int64),
                np.arange(0, cursor, dtype=np.int64),
            )
        )
    if "valid" in replay.storage:
        valid = np.asarray(jax.device_get(replay.storage["valid"]), dtype=bool)
        order = order[valid[order]]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        order = order[-max_samples:]
    if order.size == 0:
        raise ValueError("Forward replay buffer has no valid endpoint_1 samples")
    return np.asarray(jax.device_get(replay.storage["endpoint_1"][order]))


def _evaluate_target_batched(
    system: RuntimeSystem,
    standardized: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Evaluate the formal target in fixed padded batches to bound MACE memory."""

    if batch_size <= 0:
        raise ValueError("target batch_size must be positive")
    energy_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    component_parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, standardized.shape[0], batch_size):
        raw = standardized[start : start + batch_size]
        take = raw.shape[0]
        if take < batch_size:
            padding = np.repeat(raw[-1:], batch_size - take, axis=0)
            raw = np.concatenate((raw, padding), axis=0)
        state = jnp.asarray(raw)
        target = system.evaluate_target(state, include_training_wall=False)
        valid = target.valid_mask & system.support_mask(state)
        energy_parts.append(np.asarray(jax.device_get(target.energy))[:take])
        valid_parts.append(np.asarray(jax.device_get(valid), dtype=bool)[:take])
        for name, value in target.components.items():
            component_parts.setdefault(name, []).append(
                np.asarray(jax.device_get(value))[:take]
            )
    return (
        np.concatenate(energy_parts, axis=0),
        np.concatenate(valid_parts, axis=0),
        {
            name: np.concatenate(parts, axis=0)
            for name, parts in component_parts.items()
        },
    )


def write_ala2_replay_visualization(
    *,
    system: RuntimeSystem,
    replay: ReplayBufferState,
    target_reference: Mapping[str, Any],
    implicit_reference: Mapping[str, Any] | None,
    output_dir: str | Path,
    completed_outer: int,
    step: int,
    max_samples: int | None = 2048,
    target_batch_size: int = 64,
    n_bootstraps: int = 1,
    seed: int = 0,
) -> Path:
    """Write replay NPZ, four CG-BG-style figures, and unweighted metrics."""

    if system.transform is None:
        raise ValueError("Ala2 replay visualization requires the ambient transform")
    standardized = replay_endpoint_states(replay, max_samples=max_samples)
    energy, target_valid, components = _evaluate_target_batched(
        system,
        standardized,
        batch_size=target_batch_size,
    )
    physical = np.asarray(jax.device_get(system.to_physical(jnp.asarray(standardized))))
    finite = np.all(np.isfinite(physical), axis=tuple(range(1, physical.ndim)))
    keep = target_valid & finite & np.isfinite(energy)
    if not np.any(keep):
        raise ValueError("Replay diagnostic has no finite in-support Ala2 endpoints")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "R": physical[keep],
        "U": energy[keep],
        "U_target": energy[keep],
        "X_standardized": standardized[keep],
        "valid_mask": np.ones(int(np.sum(keep)), dtype=bool),
        "sampler_kind": np.asarray("training_replay_endpoint"),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "completed_outer": int(completed_outer),
                    "step": int(step),
                    "source": "forward_replay_endpoint_1",
                    "is_fresh_checkpoint_rollout": False,
                    "raw_replay_samples": int(standardized.shape[0]),
                    "retained_samples": int(np.sum(keep)),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }
    for name, value in components.items():
        payload[name] = value[keep]
    archive = output / "replay_endpoints.npz"
    np.savez_compressed(archive, **payload)

    report = evaluate_ala2(
        target=target_reference,
        sample=payload,
        output_dir=output / "formal",
        variant="ala2_cb",
        implicit=implicit_reference,
        kT=float(system.experiment["kT"]),
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    summary = {
        "completed_outer": int(completed_outer),
        "step": int(step),
        "archive": str(archive),
        "raw_replay_samples": int(standardized.shape[0]),
        "retained_samples": int(np.sum(keep)),
        "diagnostic_kind": "training_replay_endpoint",
        "fresh_checkpoint_proposal": False,
        "evaluation": report,
    }
    (output / "replay_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    return output


__all__ = [
    "load_evaluation_archive",
    "replay_endpoint_states",
    "write_ala2_replay_visualization",
]
