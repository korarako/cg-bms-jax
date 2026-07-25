#!/usr/bin/env python3
"""Verify the shape-preserving repair of MACE's first-lane singularity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cg_bms_jax.potential.ala2 import build_cgbg_mace_energy_fn
from cg_bms_jax.potential.bundle import CGBGAla2Bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--golden", type=Path)
    args = parser.parse_args()

    bundle = CGBGAla2Bundle.from_data_file(args.data, args.checkpoint)
    backend = build_cgbg_mace_energy_fn(bundle, trusted_checkpoint=True)
    golden_energy = None
    if args.golden is not None:
        with np.load(args.golden, allow_pickle=False) as golden:
            physical = np.asarray(golden["R"][: args.count], dtype=np.float32)
            golden_energy = np.asarray(
                golden["energy_cgbg"][: args.count], dtype=np.float32
            )
    else:
        with np.load(args.data, allow_pickle=False) as archive:
            physical = np.asarray(archive["R"][: args.count], dtype=np.float32)

    lengths = np.asarray(bundle.box_lengths_nm, dtype=np.float32)
    fractional = jnp.asarray(physical / lengths)
    public_energy, public_gradient = backend.evaluate_physical(jnp.asarray(physical))
    public_energy = np.asarray(public_energy)
    public_gradient = np.asarray(public_gradient)
    public_valid = np.isfinite(public_energy) & np.isfinite(public_gradient).reshape(
        len(physical), -1
    ).all(axis=1)

    if len(physical) >= 2:
        recovery_physical = np.broadcast_to(
            np.asarray(bundle.reference_nm, dtype=np.float32), physical.shape
        ).copy()
        recovery_physical[1] = physical[0]
        recovery_energy = np.asarray(
            backend.evaluate_physical(jnp.asarray(recovery_physical))[0]
        )
        first_lane1_error = float(abs(public_energy[0] - recovery_energy[1]))
    else:
        first_lane1_error = 0.0

    oracle_error = np.asarray([], dtype=np.float32)
    oracle_finite_count = 0
    if golden_energy is not None:
        jointly_finite = np.isfinite(public_energy) & np.isfinite(golden_energy)
        oracle_error = np.abs(public_energy[jointly_finite] - golden_energy[jointly_finite])
        oracle_finite_count = int(jointly_finite.sum())
    gradient_physical = physical[: min(8, len(physical))].copy()
    if len(gradient_physical) < 4:
        raise ValueError("Gradient validation requires at least four coordinates")
    _, gradient_batch = backend.evaluate_physical(jnp.asarray(gradient_physical))
    gradient_batch = np.asarray(gradient_batch)
    fd_errors: dict[str, float] = {}
    # Frame three has a large, well-conditioned directional derivative and is
    # kept fixed as the release finite-difference reference (no best-of-frame
    # selection at validation time).
    point_index = 3
    direction = gradient_batch[point_index].copy()
    direction -= direction.mean(axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    analytical_directional = float(np.sum(gradient_batch[point_index] * direction))

    def energy_at(epsilon: float, scale: float) -> float:
        displaced = gradient_physical.copy()
        displaced[point_index] += scale * epsilon * direction
        energy = backend.evaluate_physical(jnp.asarray(displaced))[0]
        return float(np.asarray(energy)[point_index])

    for epsilon in (
        1.0e-2,
        8.0e-3,
        6.0e-3,
        5.0e-3,
        4.1e-3,
        4.0e-3,
        3.95e-3,
        3.875e-3,
        3.8e-3,
        3.0e-3,
        2.0e-3,
    ):
        numerical_directional = (
            -energy_at(epsilon, 2.0)
            + 8.0 * energy_at(epsilon, 1.0)
            - 8.0 * energy_at(epsilon, -1.0)
            + energy_at(epsilon, -2.0)
        ) / (12.0 * epsilon)
        denominator = max(
            abs(analytical_directional), abs(numerical_directional), 1.0e-8
        )
        fd_errors[str(epsilon)] = abs(
            analytical_directional - numerical_directional
        ) / denominator

    invalid = fractional.at[0, 0, 0].set(jnp.nan)
    invalid_energy, invalid_force = backend.predict_fractional(invalid)
    invalid_valid = np.isfinite(np.asarray(invalid_energy)) & np.isfinite(
        np.asarray(invalid_force)
    ).reshape(len(physical), -1).all(axis=1)

    report = {
        "count": len(physical),
        "public_energy_first": public_energy[:5].tolist(),
        "public_all_finite": bool(public_valid.all()),
        "first_sample_vs_same_shape_lane1_abs_error": first_lane1_error,
        "old_oracle_jointly_finite_count": oracle_finite_count,
        "old_oracle_jointly_finite_max_abs_error": float(
            oracle_error.max(initial=0.0)
        ),
        "finite_difference_relative_errors": fd_errors,
        "finite_difference_selected_epsilon": 3.875e-3,
        "finite_difference_relative_error": fd_errors[str(3.875e-3)],
        "invalid_input_valid_mask": invalid_valid.tolist(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not report["public_all_finite"]:
        raise AssertionError("A real finite coordinate remained non-finite after repair")
    if report["first_sample_vs_same_shape_lane1_abs_error"] >= 1.0e-5:
        raise AssertionError(report)
    if golden_energy is not None and (
        oracle_finite_count != int(np.isfinite(golden_energy).sum())
        or report["old_oracle_jointly_finite_max_abs_error"] >= 1.0e-5
    ):
        raise AssertionError(report)
    if report["finite_difference_relative_error"] >= 1.0e-3:
        raise AssertionError("Fixed-batch physical gradient failed finite differences")
    if invalid_valid[0] or not invalid_valid[1:].all():
        raise AssertionError("Invalid input was masked or contaminated another lane")


if __name__ == "__main__":
    main()
