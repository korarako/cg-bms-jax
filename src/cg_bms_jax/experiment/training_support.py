"""Thin adaptation from Hydra experiment fields to the training/checkpoint APIs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax

from cg_bms_jax.checkpoint import (
    AssetProvenance,
    CheckpointMetadata,
    canonical_config_sha256,
    save_checkpoint,
)
from cg_bms_jax.runtime import RuntimeSystem, config_as_dict, resolve_project_path


def experiment_config(config: Any) -> dict[str, Any]:
    value = config_as_dict(config)["experiment"]
    if not isinstance(value, dict):
        raise TypeError("config.experiment must resolve to a dictionary")
    return value


def asset_provenance(system: RuntimeSystem) -> AssetProvenance:
    experiment = system.experiment
    family = str(system.identity.get("experiment_family", ""))
    is_ala2 = family == "ala2_ambient18_300k"
    is_ala2_all_atom = family == "ala2_aa_ambient66_300k"
    is_mb2d = family == "mb2d_analytic"
    if is_ala2:
        mapping_name = "ala2_core_beta"
        mapping_indices = (4, 6, 8, 10, 14, 16)
        coordinate_unit = "nm"
        standardization_std = float(system.transform.physical_std)
        num_particles = 6
        spatial_dimension = 3
        density_mode = "ambient_18d_aux_com"
        domain = "anchor_0_half_open_minimum_image"
    elif is_ala2_all_atom:
        mapping_name = "ala2_official_bms_22_identity"
        mapping_indices = tuple(range(22))
        coordinate_unit = "angstrom"
        standardization_std = float(system.transform.physical_std)
        num_particles = 22
        spatial_dimension = 3
        density_mode = "ambient_66d_aux_com_exact"
        domain = "unbounded_shape_plus_cartesian_gaussian_com"
    elif is_mb2d:
        scale = tuple(float(value) for value in experiment["affine"]["scale"])
        mapping_name = "mb2d_affine"
        mapping_indices = (0, 1)
        coordinate_unit = "cg_bg_mb_coordinate"
        standardization_std = math.sqrt(scale[0] * scale[1])
        num_particles = 1
        spatial_dimension = 2
        density_mode = "ambient_2d"
        domain = "finite_cartesian_box"
    else:
        mapping_name = "mb_cg1d_identity"
        mapping_indices = (0,)
        coordinate_unit = "reduced_mb_coordinate"
        standardization_std = 1.0
        num_particles = 1
        spatial_dimension = 1
        density_mode = "ambient_1d"
        domain = "real_line"
    return AssetProvenance(
        pmf_revision=str(system.identity["pmf_revision"]),
        pmf_sha256=str(system.identity["pmf_sha256"]),
        mapping_name=mapping_name,
        mapping_indices=mapping_indices,
        temperature_kelvin=float(experiment.get("temperature_kelvin", 1.0)),
        thermal_energy_kj_mol=float(experiment["kT"]),
        coordinate_unit=coordinate_unit,
        standardization_std=standardization_std,
        num_particles=num_particles,
        spatial_dimension=spatial_dimension,
        density_mode=density_mode,
        correction_mode="exact_ambient",
        domain=domain,
    )


def checkpoint_metadata(
    *,
    role: str,
    state: Any,
    system: RuntimeSystem,
    saved_config: Mapping[str, Any],
    parent_forward_sha256: str | None = None,
    warmstart_data_sha256: str | None = None,
    initial_controller_path: str | None = None,
    initial_controller_sha256: str | None = None,
    initial_controller_role: str | None = None,
) -> CheckpointMetadata:
    strict_target_identity = system.identity.get("target_mode") in {
        "pmf_canonical_support",
        "analytic_mb2d",
        "openmm_bms_chirality",
    }
    return CheckpointMetadata(
        role=role,
        global_step=int(jax.device_get(state.step)),
        created_at_utc=datetime.now(UTC).isoformat(),
        code_revision="cg-bms-jax-0.1.0+working-tree",
        config_sha256=canonical_config_sha256(saved_config),
        model_signature=str(system.identity["model_signature"]),
        sde_signature=str(system.identity["sde_signature"]),
        assets=asset_provenance(system),
        parent_forward_sha256=parent_forward_sha256,
        formal_target_signature=(
            str(system.identity["formal_target_signature"])
            if strict_target_identity
            else None
        ),
        training_target_signature=(
            str(system.identity["training_target_signature"])
            if strict_target_identity
            else None
        ),
        topology_signature=(
            str(system.identity["topology_signature"])
            if strict_target_identity
            else None
        ),
        coordinate_signature=(
            str(system.identity["coordinate_signature"])
            if system.identity.get("coordinate_signature") is not None
            else None
        ),
        species_signature=(
            str(system.identity["species_signature"])
            if system.identity.get("species_signature") is not None
            else None
        ),
        training_data_sha256=(
            str(system.identity["training_data_sha256"])
            if system.identity.get("training_data_sha256") is not None
            else None
        ),
        warmstart_data_sha256=warmstart_data_sha256,
        initial_controller_path=initial_controller_path,
        initial_controller_sha256=initial_controller_sha256,
        initial_controller_role=initial_controller_role,
        schema_version=2 if strict_target_identity else 1,
    )


def training_sizes(training: Mapping[str, Any]) -> tuple[int, int]:
    rollout_samples = int(training["rollout_samples"])
    batch_size = min(int(training["batch_size"]), rollout_samples)
    if rollout_samples <= 0 or batch_size <= 0:
        raise ValueError("training rollout_samples and batch_size must be positive")
    return batch_size, math.ceil(rollout_samples / batch_size)


def _save_training_checkpoint(
    *,
    config: Any,
    system: RuntimeSystem,
    state: Any,
    role: str,
    parent_forward_sha256: str | None = None,
    warmstart_data_sha256: str | None = None,
    initial_controller_path: str | None = None,
    initial_controller_sha256: str | None = None,
    initial_controller_role: str | None = None,
) -> tuple[Path, str]:
    root = config_as_dict(config)
    output_dir = resolve_project_path(str(root["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_config = experiment_config(config)
    metadata = checkpoint_metadata(
        role=role,
        state=state,
        system=system,
        saved_config=saved_config,
        parent_forward_sha256=parent_forward_sha256,
        warmstart_data_sha256=warmstart_data_sha256,
        initial_controller_path=initial_controller_path,
        initial_controller_sha256=initial_controller_sha256,
        initial_controller_role=initial_controller_role,
    )
    step = int(jax.device_get(state.step))
    checkpoint_path = output_dir / f"{role}_step_{step:08d}"
    digest = save_checkpoint(
        checkpoint_path,
        state=state,
        config=saved_config,
        metadata=metadata,
    )
    return checkpoint_path, digest


def save_intermediate_training_checkpoint(
    *,
    config: Any,
    system: RuntimeSystem,
    state: Any,
    role: str,
    parent_forward_sha256: str | None = None,
    warmstart_data_sha256: str | None = None,
    initial_controller_path: str | None = None,
    initial_controller_sha256: str | None = None,
    initial_controller_role: str | None = None,
) -> Path:
    """Write a controller snapshot without rewriting cumulative metrics."""

    path, _ = _save_training_checkpoint(
        config=config,
        system=system,
        state=state,
        role=role,
        parent_forward_sha256=parent_forward_sha256,
        warmstart_data_sha256=warmstart_data_sha256,
        initial_controller_path=initial_controller_path,
        initial_controller_sha256=initial_controller_sha256,
        initial_controller_role=initial_controller_role,
    )
    return path


def save_final_training_result(
    *,
    config: Any,
    system: RuntimeSystem,
    result: Any,
    role: str,
    parent_forward_sha256: str | None = None,
    warmstart_data_sha256: str | None = None,
    initial_controller_path: str | None = None,
    initial_controller_sha256: str | None = None,
    initial_controller_role: str | None = None,
) -> Path:
    checkpoint_path, digest = _save_training_checkpoint(
        config=config,
        system=system,
        state=result.state,
        role=role,
        parent_forward_sha256=parent_forward_sha256,
        warmstart_data_sha256=warmstart_data_sha256,
        initial_controller_path=initial_controller_path,
        initial_controller_sha256=initial_controller_sha256,
        initial_controller_role=initial_controller_role,
    )
    root = config_as_dict(config)
    output_dir = resolve_project_path(str(root["output_dir"]))
    step = int(jax.device_get(result.state.step))
    metrics_path = output_dir / f"{role}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": digest,
                "updates": step,
                "initial_controller": (
                    None
                    if initial_controller_sha256 is None
                    else {
                        "path": initial_controller_path,
                        "sha256": initial_controller_sha256,
                        "role": initial_controller_role,
                        "warmstart_data_sha256": warmstart_data_sha256,
                    }
                ),
                "history": list(result.metrics),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return checkpoint_path


__all__ = [
    "asset_provenance",
    "checkpoint_metadata",
    "experiment_config",
    "save_final_training_result",
    "save_intermediate_training_checkpoint",
    "training_sizes",
]
