"""CG-BG-style proposal and reweighting evaluation command."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from cg_bms_jax.data import load_ala2_all_atom_dataset
from cg_bms_jax.evaluation import (
    evaluate_ala2,
    evaluate_all_atom_ala2,
    evaluate_mb,
    evaluate_mb2d,
)
from cg_bms_jax.experiment.common import run_hydra_entry
from cg_bms_jax.runtime import (
    build_runtime_system,
    config_as_dict,
    resolve_project_path,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _clip_diagnostic_view(sample: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    """Use raw log weights so the requested percentile is actually applied.

    The evaluation helpers correctly give a stored normalized ``weights`` field
    priority.  CG-BG's historical clip diagnostic instead starts from raw log
    weights, so this separate mapping deliberately removes the formal weights.
    """

    if "logw_raw" not in sample and "logw" not in sample:
        return None
    result = {name: np.asarray(value) for name, value in sample.items() if name not in {"weights", "logw"}}
    result["logw"] = np.asarray(sample.get("logw_raw", sample["logw"]))
    return result


def _deterministic_subset(indices: np.ndarray, maximum: int | None) -> np.ndarray:
    """Retain an ordered, evenly spaced subset without sampling leakage."""

    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("held-out reference indices must be non-empty")
    if maximum is None or int(maximum) >= values.size:
        return values
    maximum = int(maximum)
    if maximum <= 0:
        raise ValueError("reference_max_frames must be positive when configured")
    offsets = np.linspace(0, values.size - 1, maximum, dtype=np.int64)
    return values[offsets]


def _all_atom_reference(
    config: DictConfig,
    experiment: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load only the chronological AA test block and evaluate its target U."""

    bridge_data = experiment.get("bridge_data")
    if not isinstance(bridge_data, Mapping):
        raise TypeError("all-atom evaluation requires experiment.bridge_data")
    reference_override = config.get("implicit_reference_override")
    reference_path = resolve_project_path(
        str(
            bridge_data["path"]
            if reference_override is None
            else reference_override
        )
    )
    dataset = load_ala2_all_atom_dataset(
        reference_path,
        expected_sha256=str(bridge_data["sha256"]),
        dtype=np.float32,
    )
    split_cfg = bridge_data.get("split", {})
    if not isinstance(split_cfg, Mapping):
        raise TypeError("experiment.bridge_data.split must be a mapping")
    split = dataset.split_time_blocks(
        validation_fraction=float(split_cfg.get("validation_fraction", 0.1)),
        test_fraction=float(split_cfg.get("test_fraction", 0.1)),
        gap_frames=int(split_cfg.get("gap_frames", 0)),
    )
    configured_maximum = config.get("reference_max_frames")
    if configured_maximum is None:
        configured_maximum = bridge_data.get("evaluation_max_frames")
    reference_split = str(config.get("reference_split", "test")).lower()
    if reference_split == "validation":
        reference_indices = split.validation
    elif reference_split == "test":
        reference_indices = split.test
    else:
        raise ValueError(
            "all-atom reference_split must be 'validation' or 'test'"
        )
    if reference_indices.size == 0:
        raise ValueError(
            f"all-atom reference split {reference_split!r} is empty"
        )
    selected = _deterministic_subset(
        reference_indices,
        None if configured_maximum is None else int(configured_maximum),
    )
    coordinates = np.asarray(dataset.coordinates_angstrom[selected])

    # The evaluator plots the same molecular target carried by proposal U.
    # The auxiliary COM is intentionally absent from this dimensional energy
    # distribution; it remains present in target_reduced_energy and weights.
    system = build_runtime_system(
        config,
        key=jax.random.PRNGKey(int(config.get("seed", 0))),
    )
    physical_std = float(system.transform.physical_std)
    batch_size = int(config.get("reference_target_batch_size", 64))
    if batch_size <= 0:
        raise ValueError("reference_target_batch_size must be positive")
    energies: list[np.ndarray] = []
    for start in range(0, coordinates.shape[0], batch_size):
        state = jnp.asarray(
            coordinates[start : start + batch_size] / physical_std,
            dtype=jnp.float32,
        )
        result = system.evaluate_target(state, include_training_wall=False)
        energies.append(np.asarray(jax.device_get(result.energy)))
    target = {
        "R": coordinates,
        "U_target": np.concatenate(energies, axis=0),
    }
    metadata = {
        "path": str(reference_path),
        "sha256": dataset.sha256,
        "split_kind": "chronological_blocks",
        "reference_split": reference_split,
        "total_frames": dataset.num_frames,
        "train_frames": int(split.train.size),
        "validation_frames": int(split.validation.size),
        "test_frames": int(split.test.size),
        "gap_frames": int(split.gap.size),
        "evaluated_test_frames": (
            int(selected.size) if reference_split == "test" else 0
        ),
        "evaluated_reference_frames": int(selected.size),
        "first_reference_index": int(reference_indices[0]),
        "last_reference_index": int(reference_indices[-1]),
        "first_test_index": int(split.test[0]),
        "last_test_index": int(split.test[-1]),
        "atom_order": "official_bms_22",
        "input_unit": "nm",
        "evaluation_unit": "angstrom",
    }
    return target, metadata


def run(config: DictConfig) -> dict[str, Any]:
    root = config_as_dict(config)
    experiment = root["experiment"]
    sample_path = resolve_project_path(str(root["samples"]))
    output_dir = resolve_project_path(str(root["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = _load_npz(sample_path)
    n_bootstraps = int(root["n_bootstraps"]) if bool(root["bootstrap"]) else 1

    coordinate_mode = str(experiment["coordinate_mode"])
    if coordinate_mode == "identity":
        target_path = resolve_project_path(str(experiment["assets"]["reference"]))
        common = {
            "target": target_path,
            "sample": sample,
            "n_bootstraps": n_bootstraps,
            "seed": int(root.get("seed", 0)),
        }
        formal = evaluate_mb(
            **common,
            output_dir=output_dir / "formal",
            kT=float(experiment["kT"]),
        )
        evaluator = evaluate_mb
        evaluator_extra: dict[str, Any] = {"kT": float(experiment["kT"])}
    elif coordinate_mode == "mb2d_affine":
        target_path = "analytic_muller_brown_2d"
        formal = evaluate_mb2d(
            sample=sample,
            output_dir=output_dir / "formal",
            kT=float(experiment["kT"]),
            domain=experiment["target"]["box"],
        )
        evaluator = None
        evaluator_extra = {}
    elif coordinate_mode == "cgbg_ambient18":
        target_path = resolve_project_path(str(experiment["assets"]["reference"]))
        common = {
            "target": target_path,
            "sample": sample,
            "n_bootstraps": n_bootstraps,
            "seed": int(root.get("seed", 0)),
        }
        implicit_path = None
        candidates = (
            root.get("implicit_reference_override"),
            experiment["assets"].get("implicit_reference"),
        )
        for candidate_value in candidates:
            if candidate_value is None:
                continue
            candidate = resolve_project_path(str(candidate_value))
            if candidate.is_file():
                implicit_path = candidate
                break
        formal = evaluate_ala2(
            **common,
            output_dir=output_dir / "formal",
            variant="ala2_cb",
            implicit=implicit_path,
            kT=float(experiment["kT"]),
        )
        evaluator = evaluate_ala2
        evaluator_extra = {
            "variant": "ala2_cb",
            "implicit": implicit_path,
            "kT": float(experiment["kT"]),
        }
    elif coordinate_mode == "ala2_aa_ambient66":
        target, target_metadata = _all_atom_reference(config, experiment)
        target_path = target_metadata["path"]
        formal = evaluate_all_atom_ala2(
            target=target,
            sample=sample,
            output_dir=output_dir / "formal",
            kT=float(experiment["kT"]),
            target_units="angstrom",
            sample_units="angstrom",
        )
        formal["reference"] = target_metadata
        evaluator = None
        evaluator_extra = {}
    else:
        raise ValueError(f"Unsupported evaluation coordinate mode: {coordinate_mode!r}")

    clipped = None
    clip_percentile = root.get("compat_clip_percentile")
    diagnostic_sample = _clip_diagnostic_view(sample)
    if (
        coordinate_mode == "mb2d_affine"
        and diagnostic_sample is not None
        and clip_percentile is not None
    ):
        clipped = evaluate_mb2d(
            sample=sample,
            output_dir=output_dir / f"cgbg_clip_{float(clip_percentile):g}",
            kT=float(experiment["kT"]),
            domain=experiment["target"]["box"],
            clip_percentile=float(clip_percentile),
            clip_mode="drop",
        )
    elif (
        evaluator is not None
        and diagnostic_sample is not None
        and clip_percentile is not None
    ):
        clipped = evaluator(
            target=target_path,
            sample=diagnostic_sample,
            output_dir=output_dir / f"cgbg_clip_{float(clip_percentile):g}",
            n_bootstraps=n_bootstraps,
            seed=int(root.get("seed", 0)),
            clip_percentile=float(clip_percentile),
            clip_mode="drop",
            **evaluator_extra,
        )

    summary = {
        "sample": str(sample_path),
        "target": str(target_path),
        "formal": formal,
        "cgbg_clip_diagnostic": clipped,
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True, default=str),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    result = run_hydra_entry("evaluate", run)
    if result is not None:
        print("Evaluation complete")


if __name__ == "__main__":
    main()
