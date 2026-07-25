"""Ala2 bridge-matching warm-start command.

The command uses only the hash-pinned CG-BG ``flow_b`` training archive.  Its
output is a ``forward_pretrain`` checkpoint that may initialize, but never
resume, a fixed-target forward BMS run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from omegaconf import DictConfig

from cg_bms_jax.checkpoint import save_checkpoint
from cg_bms_jax.data import (
    Ala2AllAtomDataset,
    Ala2WarmStartDataset,
    load_ala2_all_atom_dataset,
    load_ala2_warmstart_dataset,
)
from cg_bms_jax.experiment.common import run_hydra_entry
from cg_bms_jax.experiment.training_support import checkpoint_metadata
from cg_bms_jax.runtime import (
    build_runtime_system,
    config_as_dict,
    resolve_project_path,
)
from cg_bms_jax.training import (
    BridgePretrainConfig,
    flax_apply,
    split_flax_variables,
    train_bridge_pretrain,
)


@dataclass(frozen=True)
class _PretrainResultView:
    state: Any
    metrics: tuple[dict[str, float | bool | int], ...]


def _indices_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _key_seed(key: jax.Array, *, namespace: bytes) -> int:
    """Convert a JAX key to a deterministic NumPy seed without shared state."""

    words = np.asarray(jax.device_get(key), dtype="<u4").reshape(-1)
    digest = hashlib.sha256(namespace + words.tobytes(order="C")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def make_ala2_endpoint_provider(
    dataset: Ala2WarmStartDataset,
    train_indices: np.ndarray,
    *,
    random_rotation: bool = True,
    add_com_noise: bool = True,
) -> Callable[[jax.Array, int], np.ndarray]:
    """Return the leakage-safe host provider consumed by JAX pretraining.

    Production Ala2 warm starts deliberately require both CG-BG augmentations.
    The boolean arguments remain explicit so a config mismatch fails before an
    expensive training run rather than silently changing the density space.
    """

    if not random_rotation:
        raise ValueError("Ala2 warm start requires CG-BG random SO(3) rotations")
    if not add_com_noise:
        raise ValueError("Ala2 warm start requires auxiliary COM noise in ambient 18D")
    candidates = np.asarray(train_indices, dtype=np.int64)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("train_indices must be a non-empty one-dimensional array")

    def provide(key: jax.Array, batch_size: int) -> np.ndarray:
        return dataset.sample_batch(
            candidates,
            batch_size,
            index_seed=_key_seed(key, namespace=b"cg-bms-jax/warm/index/v1"),
            augmentation_seed=_key_seed(
                key, namespace=b"cg-bms-jax/warm/augmentation/v1"
            ),
            replace=True,
            dtype=np.float32,
        ).endpoint_1

    return provide


def make_all_atom_ala2_endpoint_provider(
    dataset: Ala2AllAtomDataset,
    train_indices: np.ndarray,
    *,
    physical_std_angstrom: float,
    random_rotation: bool = True,
    add_com_noise: bool = True,
) -> Callable[[jax.Array, int], np.ndarray]:
    """Return official-order full-rank 66D endpoints from the MD train block."""

    if not random_rotation:
        raise ValueError("All-atom Ala2 warm start requires random SO(3) rotations")
    if not add_com_noise:
        raise ValueError(
            "All-atom ambient-66 warm start requires auxiliary Cartesian COM noise"
        )
    scale = float(physical_std_angstrom)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("physical_std_angstrom must be finite and positive")
    candidates = np.asarray(train_indices, dtype=np.int64)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("train_indices must be a non-empty one-dimensional array")

    def provide(key: jax.Array, batch_size: int) -> np.ndarray:
        batch = dataset.sample_batch(
            candidates,
            batch_size,
            index_seed=_key_seed(
                key, namespace=b"cg-bms-jax/aa66/warm/index/v1"
            ),
            augmentation_seed=_key_seed(
                key, namespace=b"cg-bms-jax/aa66/warm/augmentation/v1"
            ),
            replace=True,
            dtype=np.float32,
        )
        return np.asarray(batch.endpoint_1 / scale, dtype=np.float32)

    return provide


def _forbidden_reference_paths(experiment: Mapping[str, Any]) -> tuple[Path, ...]:
    assets = experiment.get("assets")
    if not isinstance(assets, Mapping):
        raise TypeError("experiment.assets must be a mapping")
    paths: list[Path] = []
    for name in ("reference", "implicit_reference"):
        value = assets.get(name)
        if value is not None:
            paths.append(resolve_project_path(str(value)))
    return tuple(paths)


def _pretrain_config(config: DictConfig) -> BridgePretrainConfig:
    section = config.pretrain
    return BridgePretrainConfig(
        updates=int(section.updates),
        batch_size=int(section.batch_size),
        learning_rate=float(section.learning_rate),
        final_learning_rate=float(section.final_learning_rate),
        learning_rate_warmup_updates=int(section.learning_rate_warmup_updates),
        learning_rate_decay_updates=(
            None
            if section.get("learning_rate_decay_updates") is None
            else int(section.learning_rate_decay_updates)
        ),
        learning_rate_warmup_start_factor=float(
            section.get("learning_rate_warmup_start_factor", 1.0e-3)
        ),
        gradient_clip_norm=float(section.gradient_clip_norm),
        weight_decay=float(section.weight_decay),
        max_time=float(section.bridge_matching_max_time),
        progress_every=int(config.get("progress_every", 100)),
        checkpoint_every=(
            None
            if config.get("checkpoint_every") is None
            else int(config.checkpoint_every)
        ),
        seed=int(config.seed),
    )


def _save_result(
    *,
    config: DictConfig,
    system: Any,
    result: Any,
    dataset: Ala2WarmStartDataset | Ala2AllAtomDataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray | None = None,
    gap_indices: np.ndarray | None = None,
) -> Path:
    root = config_as_dict(config)
    output_dir = resolve_project_path(str(root["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    step = int(jax.device_get(result.state.step))
    checkpoint_path = output_dir / f"forward_pretrain_step_{step:08d}"
    metadata = checkpoint_metadata(
        role="forward_pretrain",
        state=result.state,
        system=system,
        saved_config=root,
        warmstart_data_sha256=dataset.sha256,
    )
    digest = save_checkpoint(
        checkpoint_path,
        state=result.state,
        config=root,
        metadata=metadata,
    )
    metrics_path = output_dir / "forward_pretrain_metrics.json"
    data_metadata: dict[str, Any] = {
        "path": str(dataset.path),
        "sha256": dataset.sha256,
        "num_frames": dataset.num_frames,
        "train_frames": int(train_indices.size),
        "validation_frames": int(validation_indices.size),
        "train_indices_sha256": _indices_sha256(train_indices),
        "validation_indices_sha256": _indices_sha256(validation_indices),
    }
    if isinstance(dataset, Ala2WarmStartDataset):
        data_metadata["standardization_std_nm"] = (
            dataset.standardization_std_nm
        )
        data_metadata["atom_order"] = "cgbg_core_beta"
    else:
        data_metadata.update(
            {
                "coordinate_unit": "angstrom",
                "atom_order": "official_bms_22",
                "auxiliary_com_std_angstrom": (
                    dataset.auxiliary_com_std_angstrom
                ),
                "test_frames": int(
                    0 if test_indices is None else test_indices.size
                ),
                "gap_frames": int(
                    0 if gap_indices is None else gap_indices.size
                ),
                "test_indices_sha256": (
                    None
                    if test_indices is None
                    else _indices_sha256(test_indices)
                ),
                "gap_indices_sha256": (
                    None
                    if gap_indices is None
                    else _indices_sha256(gap_indices)
                ),
            }
        )
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": digest,
                "updates": step,
                "data": data_metadata,
                "history": list(result.metrics),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def run(config: DictConfig) -> Path:
    system = build_runtime_system(
        config,
        key=jax.random.PRNGKey(int(config.seed)),
        load_potential=False,
    )
    if system.transform is None or tuple(system.event_shape) not in {
        (6, 3),
        (22, 3),
    }:
        raise ValueError(
            "bridge warm start supports six-bead or all-atom Ala2"
        )
    experiment = system.experiment
    assets = experiment.get("assets")
    if not isinstance(assets, Mapping):
        raise TypeError("experiment.assets must be a mapping")
    expected_data_sha = system.identity.get("training_data_sha256")
    if not isinstance(expected_data_sha, str):
        raise ValueError("runtime identity does not pin the warm-start training data")
    if tuple(system.event_shape) == (6, 3):
        train_path = resolve_project_path(str(assets["train"]))
        dataset = load_ala2_warmstart_dataset(
            train_path,
            allowed_training_path=train_path,
            expected_sha256=expected_data_sha,
            forbidden_reference_paths=_forbidden_reference_paths(experiment),
            runtime_std_nm=float(system.transform.physical_std),
        )
        split = dataset.split_indices(
            seed=int(config.data.split_seed),
            validation_fraction=float(config.data.validation_fraction),
        )
        train_indices = split.train
        validation_indices = split.validation
        test_indices = None
        gap_indices = None
        provider = make_ala2_endpoint_provider(
            dataset,
            train_indices,
            random_rotation=bool(config.data.random_rotation),
            add_com_noise=bool(config.data.add_com_noise),
        )
    else:
        bridge_cfg = experiment.get("bridge_data")
        if not isinstance(bridge_cfg, Mapping):
            raise TypeError("all-atom experiment.bridge_data must be a mapping")
        train_path = resolve_project_path(str(bridge_cfg["path"]))
        dataset = load_ala2_all_atom_dataset(
            train_path,
            expected_sha256=expected_data_sha,
            auxiliary_com_std_angstrom=(
                float(system.transform.resolved_com_std)
                * float(system.transform.physical_std)
            ),
        )
        split_cfg = bridge_cfg.get("split", {})
        if not isinstance(split_cfg, Mapping):
            raise TypeError("all-atom experiment.bridge_data.split must be a mapping")
        split = dataset.split_time_blocks(
            validation_fraction=float(
                split_cfg.get(
                    "validation_fraction",
                    config.data.validation_fraction,
                )
            ),
            test_fraction=float(
                split_cfg.get(
                    "test_fraction",
                    config.data.get("test_fraction", 0.1),
                )
            ),
            gap_frames=int(
                split_cfg.get("gap_frames", config.data.get("gap_frames", 0))
            ),
        )
        train_indices = split.train
        validation_indices = split.validation
        test_indices = split.test
        gap_indices = split.gap
        provider = make_all_atom_ala2_endpoint_provider(
            dataset,
            train_indices,
            physical_std_angstrom=float(system.transform.physical_std),
            random_rotation=bool(config.data.random_rotation),
            add_com_noise=bool(config.data.add_com_noise),
        )
    params, constants = split_flax_variables(system.initial_variables)
    loop_config = _pretrain_config(config)

    def save_intermediate(state: Any, history: tuple[dict[str, float | bool | int], ...]) -> None:
        path = _save_result(
            config=config,
            system=system,
            result=_PretrainResultView(state=state, metrics=history),
            dataset=dataset,
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            gap_indices=gap_indices,
        )
        print(f"[bridge-pretrain] checkpoint={path}", flush=True)

    result = train_bridge_pretrain(
        initial_params=params,
        constants=constants,
        control_apply=flax_apply(system.controller.apply),
        source=system.source,
        sde=system.sde,
        config=loop_config,
        endpoint_provider=provider,
        checkpoint_callback=(
            save_intermediate if loop_config.checkpoint_every is not None else None
        ),
    )
    return _save_result(
        config=config,
        system=system,
        result=result,
        dataset=dataset,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        gap_indices=gap_indices,
    )


def main() -> None:
    result = run_hydra_entry("pretrain_bridge", run)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()


__all__ = [
    "make_ala2_endpoint_provider",
    "make_all_atom_ala2_endpoint_provider",
    "run",
]
