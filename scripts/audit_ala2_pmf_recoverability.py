#!/usr/bin/env python3
"""Audit whether the Ala2 PMF score can recover cold BMS endpoints.

This is a read-only diagnostic.  It evaluates source samples, exact
zero-control EDM endpoints at requested sigma maxima, and optional existing
NPZ archives.  No controller is loaded and no training or rollout state is
modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.evaluation.pmf_recoverability import (
    radial_scale_coordinates,
    radial_score_direction_metrics,
    recoverability_report,
    reference_bond_medians,
)
from cg_bms_jax.process import EDMSDE
from cg_bms_jax.runtime import (
    build_runtime_system,
    compose_config,
    resolve_project_path,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="ala2_ambient18_300k")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="additional Hydra override; repeatable (diagnostic only)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-npz-frames", type=int, default=4096)
    parser.add_argument(
        "--uncontrolled-sigma-max",
        type=float,
        nargs="*",
        default=None,
        help="EDM g(0) values; defaults to the experiment sigma_max; pass the flag with no values to skip",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        action="append",
        default=[],
        help="existing Ala2 archive containing R and optionally X_standardized; repeatable",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="six-bead CG reference NPZ; defaults to experiment.assets.reference",
    )
    parser.add_argument("--reference-frames", type=int, default=256)
    parser.add_argument(
        "--reference-scales",
        type=float,
        nargs="*",
        default=[],
        help="also scan PMF energy/score along COM-centred reference radial scalings",
    )
    parser.add_argument(
        "--score-clip-norm",
        type=float,
        default=None,
        help="optional per-bead score cap; raw score is always reported too",
    )
    parser.add_argument(
        "--include-training-wall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the configured soft wall in the target score (training does)",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _quantiles(value: Any) -> dict[str, float | None]:
    finite = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"q01": None, "q50": None, "q99": None, "mean": None}
    q01, q50, q99 = np.quantile(finite, (0.01, 0.50, 0.99))
    return {
        "q01": float(q01),
        "q50": float(q50),
        "q99": float(q99),
        "mean": float(np.mean(finite)),
    }


def _load_reference(path: Path, maximum: int) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("reference-frames must be positive")
    with np.load(path, allow_pickle=False) as archive:
        if "R" not in archive.files:
            raise KeyError(f"Reference archive has no R field: {path}")
        value = np.asarray(archive["R"][:maximum], dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (6, 3):
        raise ValueError(f"Reference must contain six-bead coordinates, got {value.shape}")
    return value


def _load_npz_states(path: Path, physical_std_nm: float, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        standardized = (
            np.asarray(archive["X_standardized"][:maximum], dtype=np.float32)
            if "X_standardized" in archive.files
            else None
        )
        physical = (
            np.asarray(archive["R"][:maximum], dtype=np.float32)
            if "R" in archive.files
            else None
        )
    if standardized is None and physical is None:
        raise KeyError(f"Archive must contain R or X_standardized: {path}")
    if standardized is None:
        standardized = physical / float(physical_std_nm)
    if physical is None:
        physical = standardized * float(physical_std_nm)
    if standardized.shape != physical.shape or standardized.ndim != 3 or standardized.shape[1:] != (6, 3):
        raise ValueError(f"Invalid Ala2 archive shapes in {path}: X={standardized.shape}, R={physical.shape}")
    return standardized, physical


def _evaluate(
    system: Any,
    standardized: np.ndarray,
    *,
    batch_size: int,
    include_training_wall: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    scores: list[np.ndarray] = []
    reduced_energy: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    for start in range(0, standardized.shape[0], batch_size):
        state = jnp.asarray(standardized[start : start + batch_size])
        result = system.evaluate_target(
            state,
            include_training_wall=include_training_wall,
        )
        scores.append(np.asarray(jax.device_get(result.score), dtype=np.float64))
        reduced_energy.append(np.asarray(jax.device_get(result.reduced_energy), dtype=np.float64))
        valid.append(np.asarray(jax.device_get(result.valid_mask), dtype=bool))
    return (
        np.concatenate(scores, axis=0),
        np.concatenate(reduced_energy, axis=0),
        np.concatenate(valid, axis=0),
    )


def _clip_score_per_bead(score: np.ndarray, cap: float) -> np.ndarray:
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("score-clip-norm must be positive and finite")
    norm = np.linalg.norm(score, axis=-1, keepdims=True)
    factor = np.minimum(1.0, float(cap) / np.maximum(norm, np.finfo(np.float64).tiny))
    return score * factor


def _dataset_report(
    *,
    system: Any,
    standardized: np.ndarray,
    physical_nm: np.ndarray,
    reference_bonds_nm: np.ndarray,
    box_nm: Any,
    batch_size: int,
    include_training_wall: bool,
    score_clip_norm: float | None,
) -> dict[str, Any]:
    score, energy, valid = _evaluate(
        system,
        standardized,
        batch_size=batch_size,
        include_training_wall=include_training_wall,
    )
    canonical_support = getattr(system.potential, "canonical_support", None)
    support_lower = (
        None if canonical_support is None else canonical_support.bond_lower_nm
    )
    support_upper = (
        None if canonical_support is None else canonical_support.bond_upper_nm
    )
    raw = recoverability_report(
        physical_nm,
        score,
        score_coordinate_scale_nm=system.transform.physical_std,
        mace_box_nm=box_nm,
        reference_bond_lengths_nm=reference_bonds_nm,
        support_lower_nm=support_lower,
        support_upper_nm=support_upper,
    )
    report: dict[str, Any] = {
        "num_frames": int(standardized.shape[0]),
        "target_valid_fraction": float(np.mean(valid)),
        "reduced_energy": _quantiles(energy),
        "raw_target_score": raw,
    }
    if score_clip_norm is not None:
        clipped = _clip_score_per_bead(score, score_clip_norm)
        report["clipped_target_score"] = {
            "clip_norm": float(score_clip_norm),
            "clipped_bead_fraction": float(
                np.mean(np.linalg.norm(score, axis=-1) > float(score_clip_norm))
            ),
            **recoverability_report(
                physical_nm,
                clipped,
                score_coordinate_scale_nm=system.transform.physical_std,
                mace_box_nm=box_nm,
                reference_bond_lengths_nm=reference_bonds_nm,
                support_lower_nm=support_lower,
                support_upper_nm=support_upper,
            ),
        }
    return report


def _print_dataset(name: str, report: dict[str, Any]) -> None:
    raw = report["raw_target_score"]
    mace = raw["neighbor_graphs"]["mace_0p5_nm"]
    painn = raw["neighbor_graphs"]["painn_0p8_nm"]
    print(
        f"[{name}] n={report['num_frames']} valid={report['target_valid_fraction']:.6f} "
        f"MACE-connected={mace['connected_frame_fraction']:.4f} "
        f"MACE-isolated-frame={mace['frames_with_isolated_bead_fraction']:.4f} "
        f"PaiNN-connected={painn['connected_frame_fraction']:.4f}"
    )
    for bond, metric in raw["bond_score_directions"]["bonds"].items():
        support_text = ""
        if "above_support_fraction" in metric:
            support_text = (
                f" support-long={metric['above_support_fraction']:.3f}"
                f"->short={metric['above_support_shortening_fraction']}"
                f" support-short={metric['below_support_fraction']:.3f}"
                f"->long={metric['below_support_lengthening_fraction']}"
            )
        print(
            f"  {bond}: r50={metric['distance_nm']['q50']:.5f}nm "
            f"dr/dtau50={metric['dr_dtau_nm']['q50']:.5g}nm "
            f"restore={metric.get('restorative_fraction_active')}"
            f"{support_text}"
        )


def main() -> int:
    args = _parser().parse_args()
    if args.samples <= 0 or args.max_npz_frames <= 0:
        raise ValueError("samples and max-npz-frames must be positive")
    config = compose_config(
        "train_forward",
        [f"experiment={args.experiment}", *args.override],
    )
    system = build_runtime_system(config, key=jax.random.PRNGKey(args.seed))
    if system.transform is None or system.domain is None:
        raise ValueError("The recoverability audit requires an Ala2 ambient18 experiment")

    reference_path = resolve_project_path(
        args.reference if args.reference is not None else str(config.experiment.assets.reference)
    )
    reference = _load_reference(reference_path, args.reference_frames)
    reference_bonds = reference_bond_medians(reference)
    box_nm = np.asarray(system.domain.box, dtype=np.float64)
    physical_std = float(system.transform.physical_std)

    rng = np.random.default_rng(args.seed)
    source = rng.normal(
        loc=float(config.experiment.source.mean),
        scale=float(config.experiment.source.sigma),
        size=(args.samples, 6, 3),
    ).astype(np.float32)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "source": (source, source * physical_std)
    }

    sigma_maxima = args.uncontrolled_sigma_max
    if sigma_maxima is None:
        sigma_maxima = [float(config.experiment.sde.sigma_max)]
    common_noise = rng.normal(size=source.shape).astype(np.float32)
    sigma_min = float(config.experiment.sde.sigma_min)
    rho = float(config.experiment.sde.rho)
    uncontrolled_metadata: dict[str, Any] = {}
    for maximum in sigma_maxima:
        if maximum <= sigma_min:
            raise ValueError(f"uncontrolled sigma_max={maximum} must exceed sigma_min={sigma_min}")
        sde = EDMSDE(sigma_min=sigma_min, sigma_max=float(maximum), rho=rho)
        noise_std = float(jax.device_get(jnp.sqrt(sde.total_variance)))
        endpoint = source + noise_std * common_noise
        label = f"uncontrolled_sigma_max_{maximum:g}"
        datasets[label] = (endpoint, endpoint * physical_std)
        uncontrolled_metadata[label] = {
            "sigma_min": sigma_min,
            "sigma_max": float(maximum),
            "rho": rho,
            "accumulated_noise_std_standardized": noise_std,
        }

    for path in args.npz:
        resolved = resolve_project_path(path)
        datasets[f"npz:{resolved.name}"] = _load_npz_states(
            resolved,
            physical_std,
            args.max_npz_frames,
        )

    reports: dict[str, Any] = {}
    for label, (standardized, physical) in datasets.items():
        reports[label] = _dataset_report(
            system=system,
            standardized=standardized,
            physical_nm=physical,
            reference_bonds_nm=reference_bonds,
            box_nm=box_nm,
            batch_size=args.batch_size,
            include_training_wall=args.include_training_wall,
            score_clip_norm=args.score_clip_norm,
        )
        _print_dataset(label, reports[label])

    radial: list[dict[str, Any]] = []
    for scale in args.reference_scales:
        physical = radial_scale_coordinates(reference, scale).astype(np.float32)
        standardized = physical / physical_std
        score, energy, valid = _evaluate(
            system,
            standardized,
            batch_size=args.batch_size,
            include_training_wall=args.include_training_wall,
        )
        entry = {
            **radial_score_direction_metrics(
                reference,
                scale,
                score,
                score_coordinate_scale_nm=physical_std,
            ),
            "target_valid_fraction": float(np.mean(valid)),
            "reduced_energy": _quantiles(energy),
            "recoverability": recoverability_report(
                physical,
                score,
                score_coordinate_scale_nm=physical_std,
                mace_box_nm=box_nm,
                reference_bond_lengths_nm=reference_bonds,
                support_lower_nm=(
                    None
                    if system.potential.canonical_support is None
                    else system.potential.canonical_support.bond_lower_nm
                ),
                support_upper_nm=(
                    None
                    if system.potential.canonical_support is None
                    else system.potential.canonical_support.bond_upper_nm
                ),
            ),
        }
        radial.append(entry)
        print(
            f"[reference scale={scale:g}] U50={entry['reduced_energy']['q50']} "
            f"dalpha/dtau50={entry['dalpha_dtau']['q50']} "
            f"toward={entry['toward_reference_fraction']}"
        )

    payload = {
        "schema": "cg-bms-jax.ala2-pmf-recoverability.v1",
        "experiment": args.experiment,
        "read_only": True,
        "include_training_wall": bool(args.include_training_wall),
        "physical_std_nm": physical_std,
        "mace_box_nm": box_nm.tolist(),
        "reference": str(reference_path),
        "reference_bond_medians_nm": {
            name: float(value)
            for name, value in zip(
                (
                    "ACE-C--ALA-N",
                    "ALA-N--ALA-CA",
                    "ALA-CA--ALA-CB",
                    "ALA-CA--ALA-C",
                    "ALA-C--NME-N",
                ),
                reference_bonds,
                strict=True,
            )
        },
        "uncontrolled": uncontrolled_metadata,
        "datasets": reports,
        "reference_radial_scan": radial,
    }
    output = resolve_project_path(
        args.output
        if args.output is not None
        else f"outputs/{args.experiment}/pmf_recoverability.json"
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
