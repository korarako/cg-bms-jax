#!/usr/bin/env python3
"""Compare two unweighted Ala2 CG proposal archives on shared plot scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cg_bms_jax.evaluation.ala2 import ala2_dihedrals
from cg_bms_jax.evaluation.metrics import distribution_metrics, energy_wasserstein


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--cold-label", default="Cold start")
    parser.add_argument("--warm-label", default="Warm10k start")
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--kT", type=float, default=2.494338785445972)
    return parser


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    coordinates = np.asarray(result["R"], dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (6, 3):
        raise ValueError(f"{path}: expected R with shape (N,6,3), got {coordinates.shape}")
    return result


def _angles(data: dict[str, np.ndarray]) -> np.ndarray:
    return ala2_dihedrals(np.asarray(data["R"], dtype=np.float64), "ala2_cb")


def _fes(angles: np.ndarray, *, bins: int, kT: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histogram, phi_edges, psi_edges = np.histogram2d(
        angles[:, 0],
        angles[:, 1],
        bins=bins,
        range=((-np.pi, np.pi), (-np.pi, np.pi)),
        density=True,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -(float(kT) / 4.184) * np.log(histogram)
    finite = np.isfinite(fes)
    if np.any(finite):
        fes[finite] -= np.min(fes[finite])
    fes[~finite] = np.nan
    return fes, phi_edges, psi_edges


def _valid_fraction(data: dict[str, np.ndarray]) -> float:
    if "valid_mask" not in data:
        return float("nan")
    return float(np.mean(np.asarray(data["valid_mask"], dtype=bool)))


def _energy(data: dict[str, np.ndarray]) -> np.ndarray | None:
    if "U" not in data:
        return None
    values = np.asarray(data["U"], dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def main() -> int:
    args = _parser().parse_args()
    if args.bins <= 1:
        raise ValueError("--bins must be greater than one")

    reference = _load(args.reference)
    cold = _load(args.cold)
    warm = _load(args.warm)
    angle_sets = {
        "Reference": _angles(reference),
        args.cold_label: _angles(cold),
        args.warm_label: _angles(warm),
    }
    ranges = ((-np.pi, np.pi), (-np.pi, np.pi))
    cold_metrics = distribution_metrics(angle_sets["Reference"], angle_sets[args.cold_label], ranges=ranges)
    warm_metrics = distribution_metrics(angle_sets["Reference"], angle_sets[args.warm_label], ranges=ranges)

    energy_sets = {
        "Reference": _energy(reference),
        args.cold_label: _energy(cold),
        args.warm_label: _energy(warm),
    }
    for label, data, metrics in (
        (args.cold_label, cold, cold_metrics),
        (args.warm_label, warm, warm_metrics),
    ):
        sample_energy = energy_sets[label]
        reference_energy = energy_sets["Reference"]
        if sample_energy is not None and reference_energy is not None:
            w1, w2 = energy_wasserstein(sample_energy, reference_energy)
            metrics["Energy_W1"] = float(w1)
            metrics["Energy_W2"] = float(w2)
        metrics["Valid_Fraction"] = _valid_fraction(data)
        metrics["Num_Samples"] = int(np.asarray(data["R"]).shape[0])

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    mesh = None
    metric_lookup = {args.cold_label: cold_metrics, args.warm_label: warm_metrics}
    for axis, (label, angles) in zip(axes[0], angle_sets.items(), strict=True):
        fes, phi_edges, psi_edges = _fes(angles, bins=args.bins, kT=args.kT)
        mesh = axis.pcolormesh(phi_edges, psi_edges, fes.T, shading="auto", vmin=0.0, vmax=5.25)
        title = label
        if label in metric_lookup:
            metric = metric_lookup[label]
            title += f"\nJS={metric['JS_Divergence']:.4f}, PMF={metric['PMF_Error']:.3f}"
        axis.set(
            title=title,
            xlabel=r"$\phi$",
            ylabel=r"$\psi$",
            xlim=(-np.pi, np.pi),
            ylim=(-np.pi, np.pi),
        )
    if mesh is not None:
        colorbar_axis = fig.add_axes((0.955, 0.555, 0.012, 0.30))
        fig.colorbar(mesh, cax=colorbar_axis, label="Free energy (kcal/mol)")

    colors = {"Reference": "C2", args.cold_label: "C3", args.warm_label: "C0"}
    for axis, index, label in zip(axes[1, :2], (0, 1), (r"$\phi$", r"$\psi$"), strict=True):
        for name, angles in angle_sets.items():
            axis.hist(
                angles[:, index],
                bins=args.bins,
                range=(-np.pi, np.pi),
                density=True,
                histtype="step",
                linewidth=1.6,
                color=colors[name],
                label=name,
            )
        axis.set(xlabel=label, ylabel="Density", xlim=(-np.pi, np.pi))
        axis.legend(fontsize=8)

    energy_axis = axes[1, 2]
    reference_energy = energy_sets["Reference"]
    if reference_energy is None or reference_energy.size == 0:
        energy_axis.text(0.5, 0.5, "Energy unavailable", ha="center", va="center", transform=energy_axis.transAxes)
    else:
        lo, hi = np.quantile(reference_energy, (0.001, 0.999))
        padding = 0.10 * max(float(hi - lo), np.finfo(float).eps)
        energy_range = (float(lo - padding), float(hi + padding))
        for name, values in energy_sets.items():
            if values is None or values.size == 0:
                continue
            visible = (values >= energy_range[0]) & (values <= energy_range[1])
            energy_axis.hist(
                values[visible],
                bins=args.bins,
                range=energy_range,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=colors[name],
                label=f"{name} ({100.0 * np.mean(visible):.1f}% shown)",
            )
        energy_axis.set(xlabel="Energy (kJ/mol)", ylabel="Density", xlim=energy_range)
        energy_axis.legend(fontsize=8)

    fig.suptitle("Ala2 unweighted forward-SDE proposals", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.91, wspace=0.28, hspace=0.32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    report = {
        "reference": str(args.reference),
        "proposal_a": {
            "label": args.cold_label,
            "path": str(args.cold),
            "metrics": cold_metrics,
        },
        "proposal_b": {
            "label": args.warm_label,
            "path": str(args.warm),
            "metrics": warm_metrics,
        },
        "plot": str(args.output),
    }
    metrics_output = args.metrics_output or args.output.with_suffix(".json")
    metrics_output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    print(args.output)
    print(metrics_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
