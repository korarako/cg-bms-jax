"""Versioned bridge warm starts for analytic flat-state experiments.

Unlike the Ala2 entry point, this command never reads a molecular trajectory
or PMF asset. It either regenerates the historical synthetic endpoint bank or
loads the canonical SHA-pinned exact-grid MB2D endpoint dataset, as selected
by ``experiment.bridge_data``. The complete data specification is retained in
the checkpoint config and its canonical digest is recorded as
``warmstart_data_sha256``.

The output role is ``forward_pretrain``.  It is a parameters-only initializer
for a subsequent energy BMS run with the same controller, SDE, coordinate
transform and synthetic-data identity; it is not a resumed energy run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from omegaconf import DictConfig

from cg_bms_jax.checkpoint import save_checkpoint
from cg_bms_jax.data import (
    EquilibriumEndpointSource,
    FlatBridgeEndpointBank,
    FullSupportGaussianMixture,
    bridge_endpoint_source_from_config,
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


def _bridge_distribution(
    system: Any,
) -> tuple[
    FullSupportGaussianMixture | EquilibriumEndpointSource,
    FlatBridgeEndpointBank,
]:
    experiment = system.experiment
    if system.transform is not None:
        raise ValueError("flat bridge pretraining does not accept molecular transforms")
    if len(system.event_shape) != 1 or system.event_shape[0] <= 0:
        raise ValueError("flat bridge pretraining requires a vector event shape")
    if str(experiment.get("coordinate_mode")) != "mb2d_affine":
        raise ValueError(
            "the released flat bridge entry point requires coordinate_mode=mb2d_affine"
        )
    affine = experiment.get("affine")
    if not isinstance(affine, Mapping):
        raise TypeError("experiment.affine must be a mapping")
    bridge_data = experiment.get("bridge_data")
    if not isinstance(bridge_data, Mapping):
        raise TypeError("experiment.bridge_data must be a mapping")
    expected_target = None
    if str(bridge_data.get("distribution")) == "equilibrium_endpoint_npz_v1":
        target = experiment.get("target")
        if not isinstance(target, Mapping):
            raise TypeError(
                "equilibrium flat bridge requires experiment.target"
            )
        expected_target = {
            "implementation_abi": "analytic_muller_brown_finite_box_v1",
            "energy": "cg_bg_muller_brown_unbiased",
            "beta": float(target["beta"]),
            "affine_offset": [float(value) for value in affine["offset"]],
            "affine_scale": [float(value) for value in affine["scale"]],
            "physical_box": [
                [float(bound) for bound in axis] for axis in target["box"]
            ],
        }
    distribution = bridge_endpoint_source_from_config(
        bridge_data,
        affine_offset=affine["offset"],
        affine_scale=affine["scale"],
        base_dir=resolve_project_path("."),
        expected_target=expected_target,
    )
    if distribution.dimension != system.event_shape[0]:
        raise ValueError(
            "bridge-data dimension does not match the configured controller state"
        )
    expected_digest = system.identity.get("training_data_sha256")
    if expected_digest is None:
        raise ValueError(
            "the flat runtime identity must pin experiment.bridge_data as "
            "training_data_sha256"
        )
    if str(expected_digest) != distribution.provenance_sha256:
        raise ValueError(
            "runtime training-data identity does not match experiment.bridge_data"
        )
    return distribution, distribution.generate(dtype=np.float32)


def _save_endpoint_artifacts(
    *,
    output_dir: Path,
    distribution: FullSupportGaussianMixture | EquilibriumEndpointSource,
    bank: FlatBridgeEndpointBank,
    save_endpoint_bank: bool,
) -> None:
    if isinstance(distribution, FullSupportGaussianMixture):
        manifest: dict[str, Any] = {
            "distribution": "full_support_diagonal_gaussian_mixture_v1",
            "full_support": True,
            "coordinate_space": "physical",
            "seed": bank.seed,
            "num_endpoints": bank.num_endpoints,
            "event_shape": list(bank.event_shape),
            "bridge_data_sha256": bank.bridge_data_sha256,
            "affine": {
                "offset": list(distribution.affine_offset),
                "scale": list(distribution.affine_scale),
            },
            "components": [
                {
                    "label": component.label,
                    "mean": list(component.mean),
                    "scale": list(component.scale),
                    "weight": component.weight,
                    "sample_count": bank.component_counts[index],
                }
                for index, component in enumerate(distribution.components)
            ],
        }
    else:
        manifest = {
            "distribution": bank.distribution,
            "full_support": False,
            "formal_support": "finite_cartesian_box",
            "coordinate_space": "state",
            "seed": bank.seed,
            "num_endpoints": bank.num_endpoints,
            "event_shape": list(bank.event_shape),
            "bridge_data_sha256": bank.bridge_data_sha256,
            "dataset_sha256": bank.dataset_sha256,
            "split": bank.split_name,
            "affine": {
                "offset": list(distribution.affine_offset),
                "scale": list(distribution.affine_scale),
            },
            "dataset": dict(bank.source_metadata or {}),
        }
    manifest_path = output_dir / "flat_bridge_endpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    if save_endpoint_bank:
        np.savez_compressed(
            output_dir / "flat_bridge_endpoints.npz",
            endpoint_1=bank.endpoint_1,
            physical=bank.physical,
            component_index=bank.component_index,
            component_labels=np.asarray(bank.component_labels),
            bridge_data_sha256=np.asarray(bank.bridge_data_sha256),
            distribution=np.asarray(bank.distribution),
            dataset_sha256=np.asarray(bank.dataset_sha256 or ""),
            split_name=np.asarray(bank.split_name or ""),
            seed=np.asarray(bank.seed, dtype=np.int64),
        )


def _save_result(
    *,
    config: DictConfig,
    system: Any,
    result: Any,
    distribution: FullSupportGaussianMixture | EquilibriumEndpointSource,
    bank: FlatBridgeEndpointBank,
    write_endpoint_artifacts: bool,
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
        warmstart_data_sha256=bank.bridge_data_sha256,
    )
    digest = save_checkpoint(
        checkpoint_path,
        state=result.state,
        config=root,
        metadata=metadata,
    )
    metrics_path = output_dir / "forward_pretrain_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": digest,
                "updates": step,
                "endpoint_data": {
                    "bridge_data_sha256": bank.bridge_data_sha256,
                    "distribution": bank.distribution,
                    "dataset_sha256": bank.dataset_sha256,
                    "split": bank.split_name,
                    "seed": bank.seed,
                    "num_endpoints": bank.num_endpoints,
                    "component_labels": list(bank.component_labels),
                    "component_counts": list(bank.component_counts),
                    "full_support": bool(
                        (bank.source_metadata or {}).get("full_support", False)
                    ),
                },
                "history": list(result.metrics),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    if write_endpoint_artifacts:
        _save_endpoint_artifacts(
            output_dir=output_dir,
            distribution=distribution,
            bank=bank,
            save_endpoint_bank=bool(config.get("save_endpoint_bank", True)),
        )
    return checkpoint_path


def run(config: DictConfig) -> Path:
    system = build_runtime_system(
        config,
        key=jax.random.PRNGKey(int(config.seed)),
        load_potential=False,
    )
    distribution, bank = _bridge_distribution(system)
    params, constants = split_flax_variables(system.initial_variables)
    loop_config = _pretrain_config(config)

    def save_intermediate(
        state: Any,
        history: tuple[dict[str, float | bool | int], ...],
    ) -> None:
        path = _save_result(
            config=config,
            system=system,
            result=_PretrainResultView(state=state, metrics=history),
            distribution=distribution,
            bank=bank,
            write_endpoint_artifacts=False,
        )
        print(f"[flat-bridge-pretrain] checkpoint={path}", flush=True)

    result = train_bridge_pretrain(
        initial_params=params,
        constants=constants,
        control_apply=flax_apply(system.controller.apply),
        source=system.source,
        sde=system.sde,
        config=loop_config,
        target_endpoints=jax.device_put(bank.endpoint_1),
        checkpoint_callback=(
            save_intermediate if loop_config.checkpoint_every is not None else None
        ),
    )
    return _save_result(
        config=config,
        system=system,
        result=result,
        distribution=distribution,
        bank=bank,
        write_endpoint_artifacts=True,
    )


def main() -> None:
    result = run_hydra_entry("pretrain_flat_bridge", run)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()


__all__ = ["run"]
