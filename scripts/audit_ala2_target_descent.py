#!/usr/bin/env python3
"""No-reference cold-start descent audit for a configured Ala2 target.

This diagnostic never trains a controller and never reads an MD/reference
trajectory.  It asks a narrower prerequisite question: does repeated descent
along the exact terminal score move N(0,I) coordinates into the configured
canonical flat-support region without NaNs or score explosions?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.runtime import build_runtime_system, compose_config, resolve_project_path
from cg_bms_jax.training import clip_target_score_by_bead


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="ala2_ambient18_300k_bms_eta10_s1_canonical_gate2k",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=5.0e-4)
    parser.add_argument("--score-clip-norm", type=float, default=100.0)
    parser.add_argument(
        "--method",
        choices=("fixed", "armijo"),
        default="fixed",
        help="descent update (default: fixed, preserving the original audit)",
    )
    parser.add_argument("--armijo-c1", type=float, default=1.0e-4)
    parser.add_argument(
        "--max-backtracks",
        type=int,
        default=12,
        help="maximum number of step halvings for each sample",
    )
    parser.add_argument(
        "--snapshot-step",
        type=int,
        action="append",
        default=None,
        help="step to record; repeatable (default: 0,1,5,10,25,50,100)",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _quantiles(value: Any) -> dict[str, float | None]:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {"q01": None, "q50": None, "q99": None, "mean": None}
    q01, q50, q99 = np.quantile(flat, (0.01, 0.5, 0.99))
    return {
        "q01": float(q01),
        "q50": float(q50),
        "q99": float(q99),
        "mean": float(np.mean(flat)),
    }


def _minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.round(delta / box)


def _angles(
    coordinates: np.ndarray,
    indices: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    left = _minimum_image(
        coordinates[:, indices[:, 0]] - coordinates[:, indices[:, 1]],
        box,
    )
    right = _minimum_image(
        coordinates[:, indices[:, 2]] - coordinates[:, indices[:, 1]],
        box,
    )
    sine = np.linalg.norm(np.cross(left, right), axis=-1)
    cosine = np.sum(left * right, axis=-1)
    return np.arctan2(sine, cosine)


def _snapshot(
    *,
    step: int,
    state: jax.Array,
    result: Any,
    support: Any,
    physical_std_nm: float,
    score_clip_norm: float,
    pmf_window_minimum_kj_mol: float | None,
    pmf_window_maximum_kj_mol: float | None,
) -> dict[str, Any]:
    standardized = np.asarray(jax.device_get(state), dtype=np.float64)
    physical = standardized * float(physical_std_nm)
    box = np.asarray(support.box_lengths_nm, dtype=np.float64)
    bond_indices = np.asarray(support.bond_indices, dtype=np.int64)
    bond_delta = _minimum_image(
        physical[:, bond_indices[:, 0]] - physical[:, bond_indices[:, 1]],
        box,
    )
    bonds = np.linalg.norm(bond_delta, axis=-1)
    bond_lower = np.asarray(support.bond_lower_nm, dtype=np.float64)
    bond_upper = np.asarray(support.bond_upper_nm, dtype=np.float64)
    bond_excess = np.maximum(bond_lower - bonds, 0.0) + np.maximum(
        bonds - bond_upper, 0.0
    )

    angle_indices = np.asarray(support.angle_indices, dtype=np.int64)
    angles = _angles(physical, angle_indices, box)
    angle_lower = np.asarray(support.angle_lower_rad, dtype=np.float64)
    angle_upper = np.asarray(support.angle_upper_rad, dtype=np.float64)
    angle_excess = np.maximum(angle_lower - angles, 0.0) + np.maximum(
        angles - angle_upper, 0.0
    )

    repulsion_indices = np.asarray(support.repulsion_indices, dtype=np.int64)
    repulsion_delta = _minimum_image(
        physical[:, repulsion_indices[:, 0]]
        - physical[:, repulsion_indices[:, 1]],
        box,
    )
    nonbonded = np.linalg.norm(repulsion_delta, axis=-1)
    repulsion_excess = np.maximum(float(support.repulsion_min_nm) - nonbonded, 0.0)

    bond_flat = np.all(bond_excess == 0.0, axis=-1)
    angle_flat = np.all(angle_excess == 0.0, axis=-1)
    repulsion_flat = np.all(repulsion_excess == 0.0, axis=-1)
    all_flat = bond_flat & angle_flat & repulsion_flat
    score = np.asarray(jax.device_get(result.score), dtype=np.float64)
    score_norm = np.linalg.norm(score, axis=-1)
    valid = np.asarray(jax.device_get(result.valid_mask), dtype=bool)
    components = {
        name: np.asarray(jax.device_get(value), dtype=np.float64)
        for name, value in result.components.items()
    }
    component_metrics = {
        name: _quantiles(value) for name, value in components.items()
    }
    raw_pmf = components["U_pmf_raw"]
    below_window = (
        None
        if pmf_window_minimum_kj_mol is None
        else float(np.mean(raw_pmf < pmf_window_minimum_kj_mol))
    )
    above_window = (
        None
        if pmf_window_maximum_kj_mol is None
        else float(np.mean(raw_pmf > pmf_window_maximum_kj_mol))
    )
    return {
        "step": int(step),
        "valid_fraction": float(np.mean(valid)),
        "score_norm_per_bead": _quantiles(score_norm),
        "score_clipped_bead_fraction": float(np.mean(score_norm > score_clip_norm)),
        "bond_flat_fraction": float(np.mean(bond_flat)),
        "angle_flat_fraction": float(np.mean(angle_flat)),
        "repulsion_flat_fraction": float(np.mean(repulsion_flat)),
        "all_canonical_support_flat_fraction": float(np.mean(all_flat)),
        "bond_excess_nm": _quantiles(bond_excess),
        "angle_excess_rad": _quantiles(angle_excess),
        "repulsion_excess_nm": _quantiles(repulsion_excess),
        "minimum_nonbonded_nm": _quantiles(np.min(nonbonded, axis=-1)),
        "pmf_below_window_fraction": below_window,
        "pmf_above_window_fraction": above_window,
        "components": component_metrics,
    }


def _transition_report(
    *,
    step: int,
    energy_before: np.ndarray,
    energy_after: np.ndarray,
    method: str,
    accepted: np.ndarray | None = None,
    backtracks: np.ndarray | None = None,
) -> dict[str, Any]:
    before = np.asarray(energy_before, dtype=np.float64)
    after = np.asarray(energy_after, dtype=np.float64)
    finite = np.isfinite(before) & np.isfinite(after)
    decrease = before - after
    relative_decrease = decrease / np.maximum(np.abs(before), 1.0)
    report: dict[str, Any] = {
        "step": int(step),
        "method": method,
        "finite_fraction": float(np.mean(finite)),
        "energy_decrease_fraction": float(np.mean(finite & (decrease > 0.0))),
        "energy_nonincrease_fraction": float(np.mean(finite & (decrease >= 0.0))),
        "energy_decrease_kj_mol": _quantiles(decrease[finite]),
        "relative_energy_decrease": _quantiles(relative_decrease[finite]),
    }
    if method == "armijo":
        if accepted is None or backtracks is None:
            raise ValueError("Armijo transition diagnostics are missing")
        accepted_array = np.asarray(accepted, dtype=bool)
        backtrack_array = np.asarray(backtracks, dtype=np.int64)
        report.update(
            {
                "accepted_fraction": float(np.mean(accepted_array)),
                "failure_fraction": float(np.mean(~accepted_array)),
                "backtracks_accepted": _quantiles(
                    backtrack_array[accepted_array]
                ),
                "maximum_backtracks_observed": int(np.max(backtrack_array)),
            }
        )
    return report


def main() -> int:
    args = _parser().parse_args()
    if args.samples <= 0 or args.steps <= 0:
        raise ValueError("samples and steps must be positive")
    if args.step_size <= 0.0 or args.score_clip_norm <= 0.0:
        raise ValueError("step-size and score-clip-norm must be positive")
    if not 0.0 < args.armijo_c1 < 1.0:
        raise ValueError("armijo-c1 must lie strictly between zero and one")
    if args.max_backtracks < 0:
        raise ValueError("max-backtracks must be non-negative")
    config = compose_config(
        "train_forward",
        [f"experiment={args.experiment}", *args.override],
    )
    system = build_runtime_system(config, key=jax.random.PRNGKey(args.seed))
    support = getattr(system.potential, "canonical_support", None)
    if support is None or system.transform is None:
        raise ValueError("This audit requires target.mode=pmf_canonical_support")
    pmf_transform = getattr(system.potential, "pmf_lower_bound", None)
    pmf_window_minimum = getattr(pmf_transform, "minimum_kj_mol", None)
    pmf_window_maximum = getattr(pmf_transform, "maximum_kj_mol", None)

    state = system.source.sample(jax.random.PRNGKey(args.seed + 1), args.samples)

    def clipped_direction(result: Any) -> jax.Array:
        mean = jnp.mean(result.score, axis=-2, keepdims=True)
        shape_score = result.score - mean
        clipped = clip_target_score_by_bead(shape_score, args.score_clip_norm)
        return clipped - jnp.mean(clipped, axis=-2, keepdims=True) + mean

    @jax.jit
    def fixed_step(
        value: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # The wall is a training stabilizer, not part of the molecular target
        # whose cold-start recoverability this audit is intended to measure.
        result = system.evaluate_target(value, include_training_wall=False)
        direction = clipped_direction(result)
        proposed = value + jnp.asarray(args.step_size, dtype=value.dtype) * direction
        proposed_result = system.evaluate_target(
            proposed,
            include_training_wall=False,
        )
        return proposed, result.energy, proposed_result.energy

    @jax.jit
    def armijo_step(
        value: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        result = system.evaluate_target(value, include_training_wall=False)
        direction = clipped_direction(result)

        # result.gradient also contains the auxiliary COM gradient.  Removing
        # its bead mean recovers the gradient of the translation-invariant
        # formal molecular energy used in the Armijo condition.
        molecular_gradient = result.gradient - jnp.mean(
            result.gradient,
            axis=-2,
            keepdims=True,
        )
        shape_direction = direction - jnp.mean(
            direction,
            axis=-2,
            keepdims=True,
        )
        slope = jnp.sum(molecular_gradient * shape_direction, axis=(-2, -1))
        energy_before = result.energy
        batch_size = value.shape[0]
        alpha = jnp.full(
            (batch_size,),
            args.step_size,
            dtype=value.dtype,
        )
        accepted = jnp.zeros((batch_size,), dtype=bool)
        accepted_state = value
        accepted_energy = energy_before
        backtracks = jnp.full(
            (batch_size,),
            args.max_backtracks,
            dtype=jnp.int32,
        )

        def condition(loop_state: tuple[Any, ...]) -> jax.Array:
            index, _alpha, current_accepted, *_rest = loop_state
            return (index <= args.max_backtracks) & jnp.any(~current_accepted)

        def body(loop_state: tuple[Any, ...]) -> tuple[Any, ...]:
            index, current_alpha, current_accepted, output_state, output_energy, counts = (
                loop_state
            )
            candidate = value + current_alpha[:, None, None] * direction
            candidate_result = system.evaluate_target(
                candidate,
                include_training_wall=False,
            )
            armijo_bound = energy_before + (
                jnp.asarray(args.armijo_c1, dtype=value.dtype)
                * current_alpha
                * slope
            )
            newly_accepted = (
                (~current_accepted)
                & result.valid_mask
                & candidate_result.valid_mask
                & jnp.isfinite(slope)
                & (slope < 0.0)
                & (candidate_result.energy < energy_before)
                & (candidate_result.energy <= armijo_bound)
            )
            output_state = jnp.where(
                newly_accepted[:, None, None],
                candidate,
                output_state,
            )
            output_energy = jnp.where(
                newly_accepted,
                candidate_result.energy,
                output_energy,
            )
            counts = jnp.where(newly_accepted, index, counts)
            current_accepted = current_accepted | newly_accepted
            current_alpha = jnp.where(
                current_accepted,
                current_alpha,
                current_alpha * jnp.asarray(0.5, dtype=value.dtype),
            )
            return (
                index + 1,
                current_alpha,
                current_accepted,
                output_state,
                output_energy,
                counts,
            )

        initial = (
            jnp.asarray(0, dtype=jnp.int32),
            alpha,
            accepted,
            accepted_state,
            accepted_energy,
            backtracks,
        )
        _, _, accepted, accepted_state, accepted_energy, backtracks = (
            jax.lax.while_loop(condition, body, initial)
        )
        return (
            accepted_state,
            energy_before,
            accepted_energy,
            accepted,
            backtracks,
        )

    requested = args.snapshot_step or [0, 1, 5, 10, 25, 50, 100]
    snapshots = sorted({step for step in requested if 0 <= step <= args.steps} | {0, args.steps})
    reports: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    armijo_accepted_all: list[np.ndarray] = []
    armijo_backtracks_all: list[np.ndarray] = []
    energy_before_all: list[np.ndarray] = []
    energy_after_all: list[np.ndarray] = []
    for step in range(args.steps + 1):
        if step in snapshots:
            result = system.evaluate_target(state, include_training_wall=False)
            jax.block_until_ready(result.energy)
            report = _snapshot(
                step=step,
                state=state,
                result=result,
                support=support,
                physical_std_nm=system.transform.physical_std,
                score_clip_norm=args.score_clip_norm,
                pmf_window_minimum_kj_mol=pmf_window_minimum,
                pmf_window_maximum_kj_mol=pmf_window_maximum,
            )
            reports.append(report)
            print(
                f"[descent] step={step}/{args.steps} "
                f"valid={report['valid_fraction']:.6f} "
                f"all_flat={report['all_canonical_support_flat_fraction']:.6f} "
                f"bond_flat={report['bond_flat_fraction']:.6f} "
                f"angle_flat={report['angle_flat_fraction']:.6f} "
                f"repulsion_flat={report['repulsion_flat_fraction']:.6f} "
                f"score_clip={report['score_clipped_bead_fraction']:.6f}",
                flush=True,
            )
        if step < args.steps:
            if args.method == "armijo":
                state, before, after, accepted, backtracks = armijo_step(state)
                jax.block_until_ready(state)
                accepted_host = np.asarray(jax.device_get(accepted), dtype=bool)
                backtracks_host = np.asarray(
                    jax.device_get(backtracks),
                    dtype=np.int64,
                )
                armijo_accepted_all.append(accepted_host)
                armijo_backtracks_all.append(backtracks_host)
            else:
                state, before, after = fixed_step(state)
                jax.block_until_ready(state)
                accepted_host = None
                backtracks_host = None
            before_host = np.asarray(jax.device_get(before), dtype=np.float64)
            after_host = np.asarray(jax.device_get(after), dtype=np.float64)
            energy_before_all.append(before_host)
            energy_after_all.append(after_host)
            transitions.append(
                _transition_report(
                    step=step + 1,
                    energy_before=before_host,
                    energy_after=after_host,
                    method=args.method,
                    accepted=accepted_host,
                    backtracks=backtracks_host,
                )
            )

    all_before = np.concatenate(energy_before_all)
    all_after = np.concatenate(energy_after_all)
    all_finite = np.isfinite(all_before) & np.isfinite(all_after)
    all_decrease = all_before - all_after
    all_relative_decrease = all_decrease / np.maximum(np.abs(all_before), 1.0)
    transition_summary: dict[str, Any] = {
        "finite_fraction": float(np.mean(all_finite)),
        "energy_decrease_fraction": float(
            np.mean(all_finite & (all_decrease > 0.0))
        ),
        "energy_nonincrease_fraction": float(
            np.mean(all_finite & (all_decrease >= 0.0))
        ),
        "energy_decrease_kj_mol": _quantiles(all_decrease[all_finite]),
        "relative_energy_decrease": _quantiles(
            all_relative_decrease[all_finite]
        ),
    }
    if args.method == "armijo":
        all_accepted = np.concatenate(armijo_accepted_all)
        all_backtracks = np.concatenate(armijo_backtracks_all)
        transition_summary.update(
            {
                "accepted_fraction": float(np.mean(all_accepted)),
                "failure_fraction": float(np.mean(~all_accepted)),
                "backtracks_accepted": _quantiles(
                    all_backtracks[all_accepted]
                ),
                "maximum_backtracks_observed": int(np.max(all_backtracks)),
            }
        )
        print(
            f"[armijo] accepted={transition_summary['accepted_fraction']:.6f} "
            f"failed={transition_summary['failure_fraction']:.6f} "
            "median_backtracks="
            f"{transition_summary['backtracks_accepted']['q50']}",
            flush=True,
        )

    payload = {
        "schema": "cg-bms-jax.ala2-canonical-target-descent.v2",
        "experiment": args.experiment,
        "overrides": list(args.override),
        "seed": args.seed,
        "samples": args.samples,
        "steps": args.steps,
        "step_size": args.step_size,
        "score_clip_norm": args.score_clip_norm,
        "method": args.method,
        "armijo_c1": args.armijo_c1,
        "max_backtracks": args.max_backtracks,
        "include_training_wall": False,
        "pmf_energy_window_kj_mol": {
            "minimum": pmf_window_minimum,
            "maximum": pmf_window_maximum,
        },
        "uses_md_reference": False,
        "target_identity": dict(system.identity),
        "snapshots": reports,
        "transitions": transitions,
        "transition_summary": transition_summary,
    }
    output = resolve_project_path(
        args.output
        if args.output is not None
        else f"outputs/{args.experiment}/canonical_target_descent.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
