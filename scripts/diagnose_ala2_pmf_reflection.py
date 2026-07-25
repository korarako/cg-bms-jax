#!/usr/bin/env python3
"""Measure whether the CG-BG Ala2 PMF distinguishes mirror chirality."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.runtime import build_runtime_system, compose_config, resolve_project_path


def _chirality(coordinates: np.ndarray) -> np.ndarray:
    ca = coordinates[:, 2]
    return np.einsum(
        "bi,bi->b",
        np.cross(coordinates[:, 1] - ca, coordinates[:, 4] - ca),
        coordinates[:, 3] - ca,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="ala2_ambient18_300k_cold_clip_c150_2k")
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()

    config = compose_config("train_forward", [f"experiment={args.experiment}"])
    system = build_runtime_system(config, key=jax.random.PRNGKey(0))
    reference = resolve_project_path(str(config.experiment.assets.reference))
    with np.load(reference, allow_pickle=False) as archive:
        physical = np.asarray(archive["R"][: args.frames], dtype=np.float32)
    if system.transform is None:
        raise RuntimeError("Ala2 reflection diagnostic requires an ambient transform")

    standardized = system.transform.centre_and_standardize(jnp.asarray(physical))
    reflection = jnp.asarray([-1.0, 1.0, 1.0], dtype=standardized.dtype)
    mirrored = standardized * reflection
    combined = jnp.concatenate((standardized, mirrored), axis=0)

    for include_wall in (False, True):
        target = system.evaluate_target(combined, include_training_wall=include_wall)
        energy = np.asarray(jax.device_get(target.reduced_energy), dtype=np.float64)
        score = np.asarray(jax.device_get(target.score), dtype=np.float64)
        half = standardized.shape[0]
        energy_delta = energy[half:] - energy[:half]
        expected_score = score[:half] * np.asarray(reflection)
        score_error = score[half:] - expected_score
        print(
            f"include_training_wall={include_wall} "
            f"max_abs_reduced_energy_delta={np.max(np.abs(energy_delta)):.9e} "
            f"rms_reduced_energy_delta={np.sqrt(np.mean(energy_delta**2)):.9e} "
            f"max_abs_score_covariance_error={np.max(np.abs(score_error)):.9e} "
            f"rms_score_covariance_error={np.sqrt(np.mean(score_error**2)):.9e}"
        )

    original_physical = np.asarray(system.to_physical(standardized))
    mirrored_physical = np.asarray(system.to_physical(mirrored))
    original_chirality = _chirality(original_physical)
    mirrored_chirality = _chirality(mirrored_physical)
    print(
        f"chirality_sign_flip_fraction="
        f"{np.mean(np.sign(original_chirality) == -np.sign(mirrored_chirality)):.6f}"
    )


if __name__ == "__main__":
    main()
