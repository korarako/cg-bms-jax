"""Generate and validate the versioned analytic MB2D endpoint dataset."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import jax
import numpy as np
from omegaconf import DictConfig

from cg_bms_jax.data import (
    generate_mb2d_equilibrium_dataset,
    save_mb2d_equilibrium_dataset,
)
from cg_bms_jax.evaluation.mb2d import (
    DEFAULT_BASIN_CENTERS,
    analytic_mb2d_reference,
    muller_brown_energy_numpy,
)
from cg_bms_jax.experiment.common import run_hydra_entry
from cg_bms_jax.potential import AnalyticMB2DPotential
from cg_bms_jax.runtime import (
    build_runtime_system,
    config_as_dict,
    resolve_project_path,
)


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    p = np.asarray(first, dtype=np.float64).reshape(-1)
    q = np.asarray(second, dtype=np.float64).reshape(-1)
    p /= np.sum(p)
    q /= np.sum(q)
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0.0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return 0.5 * (kl(p, midpoint) + kl(q, midpoint))


def _basin_masses(coordinates: np.ndarray) -> dict[str, float]:
    names = tuple(DEFAULT_BASIN_CENTERS)
    centers = np.asarray(
        [DEFAULT_BASIN_CENTERS[name] for name in names],
        dtype=np.float64,
    )
    distances = np.sum(
        (np.asarray(coordinates, dtype=np.float64)[:, None, :] - centers[None]) ** 2,
        axis=-1,
    )
    assignment = np.argmin(distances, axis=1)
    return {
        name: float(np.mean(assignment == index))
        for index, name in enumerate(names)
    }


def _validation_metrics(
    *,
    physical: np.ndarray,
    beta: float,
    bins: int,
    domain: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, Any]:
    reference = analytic_mb2d_reference(
        beta=beta,
        bins=bins,
        domain=domain,
    )
    counts, _, _ = np.histogram2d(
        physical[:, 0],
        physical[:, 1],
        bins=(reference.x_edges, reference.y_edges),
    )
    sample_probability = counts / np.sum(counts)
    exact_basins = {}
    flat_points = reference.points.reshape(-1, 2)
    names = tuple(DEFAULT_BASIN_CENTERS)
    centers = np.asarray(
        [DEFAULT_BASIN_CENTERS[name] for name in names],
        dtype=np.float64,
    )
    assignment = np.argmin(
        np.sum((flat_points[:, None, :] - centers[None]) ** 2, axis=-1),
        axis=1,
    )
    flat_probability = reference.probability.reshape(-1)
    for index, name in enumerate(names):
        exact_basins[name] = float(np.sum(flat_probability[assignment == index]))
    observed_basins = _basin_masses(physical)
    # Keep the diagnostic implementation independent of JAX compilation.
    sample_energy = muller_brown_energy_numpy(physical)
    exact_energy_mean = float(
        np.sum(reference.probability * reference.energy)
    )
    return {
        "histogram_bins": bins,
        "histogram_js": _jensen_shannon(
            sample_probability,
            reference.probability,
        ),
        "histogram_tv": float(
            0.5
            * np.sum(
                np.abs(sample_probability - reference.probability)
            )
        ),
        "energy_mean": float(np.mean(sample_energy)),
        "exact_energy_mean": exact_energy_mean,
        "energy_mean_error": float(np.mean(sample_energy) - exact_energy_mean),
        "basin_mass": observed_basins,
        "exact_basin_mass": exact_basins,
        "basin_l1_error": float(
            sum(
                abs(observed_basins[name] - exact_basins[name])
                for name in names
            )
        ),
    }


def _reference_summary(
    *,
    beta: float,
    bins: int,
    domain: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, Any]:
    reference = analytic_mb2d_reference(
        beta=beta,
        bins=bins,
        domain=domain,
    )
    names = tuple(DEFAULT_BASIN_CENTERS)
    centers = np.asarray(
        [DEFAULT_BASIN_CENTERS[name] for name in names],
        dtype=np.float64,
    )
    points = reference.points.reshape(-1, 2)
    assignment = np.argmin(
        np.sum((points[:, None, :] - centers[None]) ** 2, axis=-1),
        axis=1,
    )
    probability = reference.probability.reshape(-1)
    basins = {
        name: float(np.sum(probability[assignment == index]))
        for index, name in enumerate(names)
    }
    return {
        "bins": bins,
        "energy_mean": float(
            np.sum(reference.probability * reference.energy)
        ),
        "basin_mass": basins,
    }


def _grid_convergence(
    *,
    beta: float,
    resolutions: tuple[int, int],
    domain: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, Any]:
    if len(resolutions) != 2 or resolutions[0] >= resolutions[1]:
        raise ValueError(
            "grid_convergence_resolution must contain increasing coarse/fine bins"
        )
    coarse = _reference_summary(
        beta=beta,
        bins=resolutions[0],
        domain=domain,
    )
    fine = _reference_summary(
        beta=beta,
        bins=resolutions[1],
        domain=domain,
    )
    basin_l1 = float(
        sum(
            abs(
                coarse["basin_mass"][name]
                - fine["basin_mass"][name]
            )
            for name in DEFAULT_BASIN_CENTERS
        )
    )
    return {
        "coarse": coarse,
        "fine": fine,
        "energy_mean_abs_delta": abs(
            float(coarse["energy_mean"]) - float(fine["energy_mean"])
        ),
        "basin_l1_delta": basin_l1,
    }


def run(config: DictConfig) -> Path:
    root = config_as_dict(config)
    output_path = resolve_project_path(root["output"])
    manifest_path = output_path.with_suffix(".manifest.json")
    overwrite = bool(root.get("overwrite", False))
    if (output_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite frozen endpoint dataset: {output_path}. "
            "Choose a new versioned output path or set overwrite=true explicitly."
        )
    system = build_runtime_system(
        config,
        key=jax.random.PRNGKey(int(root["seed"])),
        load_potential=True,
    )
    if not isinstance(system.potential, AnalyticMB2DPotential):
        raise TypeError(
            "generate_mb2d_equilibrium requires an analytic MB2D experiment"
        )
    dataset_cfg = root["dataset"]
    dataset = generate_mb2d_equilibrium_dataset(
        system.potential,
        num_train=int(dataset_cfg["num_train"]),
        num_validation=int(dataset_cfg["num_validation"]),
        num_test=int(dataset_cfg["num_test"]),
        grid_resolution=tuple(int(value) for value in dataset_cfg["grid_resolution"]),
        seed=int(root["seed"]),
        dtype=np.float32,
    )
    token = uuid.uuid4().hex
    candidate_path = output_path.with_name(
        f".{output_path.stem}.{token}.candidate{output_path.suffix}"
    )
    candidate_manifest_path = manifest_path.with_name(
        f".{manifest_path.stem}.{token}.candidate{manifest_path.suffix}"
    )
    try:
        digest = save_mb2d_equilibrium_dataset(dataset, candidate_path)
        train_physical = dataset.split_physical("train")
        validation = _validation_metrics(
            physical=train_physical,
            beta=dataset.target_beta,
            bins=int(dataset_cfg["validation_bins"]),
            domain=dataset.physical_box,
        )
        maximum_js = float(dataset_cfg["maximum_histogram_js"])
        maximum_basin_l1 = float(dataset_cfg["maximum_basin_l1"])
        convergence = _grid_convergence(
            beta=dataset.target_beta,
            resolutions=tuple(
                int(value)
                for value in dataset_cfg["grid_convergence_resolution"]
            ),
            domain=dataset.physical_box,
        )
        maximum_grid_energy_delta = float(
            dataset_cfg["maximum_grid_energy_mean_delta"]
        )
        maximum_grid_basin_l1 = float(dataset_cfg["maximum_grid_basin_l1"])
        accepted = (
            validation["histogram_js"] <= maximum_js
            and validation["basin_l1_error"] <= maximum_basin_l1
            and convergence["energy_mean_abs_delta"]
            <= maximum_grid_energy_delta
            and convergence["basin_l1_delta"] <= maximum_grid_basin_l1
        )
        manifest = {
            **dataset.metadata(),
            "dataset_sha256": digest,
            "path": str(root["output"]),
            "validation": {
                **validation,
                "maximum_histogram_js": maximum_js,
                "maximum_basin_l1": maximum_basin_l1,
                "grid_convergence": convergence,
                "maximum_grid_energy_mean_delta": maximum_grid_energy_delta,
                "maximum_grid_basin_l1": maximum_grid_basin_l1,
                "accepted": accepted,
            },
            "experiment_identity": {
                "name": str(root["experiment"]["name"]),
                "formal_target_signature": str(
                    system.identity["formal_target_signature"]
                ),
                "coordinate_signature": str(
                    system.identity["coordinate_signature"]
                ),
            },
        }
        candidate_manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        if not accepted:
            raise RuntimeError(
                "generated MB2D endpoint dataset failed validation: "
                f"JS={validation['histogram_js']:.6g}, "
                f"basin_l1={validation['basin_l1_error']:.6g}, "
                "grid_energy_delta="
                f"{convergence['energy_mean_abs_delta']:.6g}, "
                f"grid_basin_l1={convergence['basin_l1_delta']:.6g}"
            )
        if (output_path.exists() or manifest_path.exists()) and not overwrite:
            raise FileExistsError(
                "Endpoint dataset destination appeared while generating the "
                f"validated candidate: {output_path}"
            )
        os.replace(candidate_path, output_path)
        os.replace(candidate_manifest_path, manifest_path)
    finally:
        candidate_path.unlink(missing_ok=True)
        candidate_manifest_path.unlink(missing_ok=True)

    print(
        "[mb2d-equilibrium] "
        f"samples={dataset.num_samples} sha256={digest} "
        f"JS={validation['histogram_js']:.6g} "
        f"basin_l1={validation['basin_l1_error']:.6g} "
        f"grid_energy_delta={convergence['energy_mean_abs_delta']:.6g}",
        flush=True,
    )
    print(
        "Hydra override: "
        f"experiment.bridge_data.sha256={digest}",
        flush=True,
    )
    return output_path


def main() -> None:
    result = run_hydra_entry("generate_mb2d_equilibrium", run)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()


__all__ = ["run"]
