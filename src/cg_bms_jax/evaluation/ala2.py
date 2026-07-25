"""Alanine-dipeptide plots and metrics adapted from CG-BG."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    bootstrap_distribution_metrics,
    distribution_metrics,
    energy_wasserstein,
    resolve_weights,
    summarize_bootstrap,
)
from .plot_style import (
    IMPLICIT_REFERENCE_COLOR,
    PROPOSAL_COLOR,
    REFERENCE_COLOR,
    REWEIGHTED_COLOR,
    filled_stairs,
    plot_style_metadata,
)

ALA2_INDICES: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "ala2_cb": ((0, 1, 2, 4), (1, 2, 4, 5)),
    "ala2_ha": ((1, 3, 4, 6), (3, 4, 6, 8)),
    "ala2_implicit": ((4, 6, 7, 8), (6, 7, 8, 16)),
    # Public BridgeMatchingSampler Ac-Ala-NHMe ordering (22 atoms).
    "ala2_all_atom_official": ((4, 6, 8, 14), (6, 8, 14, 16)),
}


@dataclass(frozen=True)
class Ala2Dataset:
    R: np.ndarray
    dihedrals: np.ndarray
    weights: np.ndarray | None
    logw: np.ndarray | None
    energies: np.ndarray | None
    weight_source: str


def _load(source: str | Path | Mapping[str, Any]) -> dict[str, np.ndarray]:
    if isinstance(source, Mapping):
        return {key: np.asarray(value) for key, value in source.items()}
    with np.load(source, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compute_dihedral(points: Any) -> np.ndarray:
    """Vectorized signed dihedral for arrays with shape ``(...,4,3)``."""

    p = np.asarray(points, dtype=np.float64)
    if p.shape[-2:] != (4, 3):
        raise ValueError(f"Expected (...,4,3) points, got {p.shape}")
    b0 = -(p[..., 1, :] - p[..., 0, :])
    b1 = p[..., 2, :] - p[..., 1, :]
    b2 = p[..., 3, :] - p[..., 2, :]
    norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1_unit = b1 / np.maximum(norm, np.finfo(np.float64).tiny)
    v = b0 - np.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - np.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_unit, v) * w, axis=-1)
    return np.arctan2(y, x)


def ala2_dihedrals(coordinates: Any, variant: str) -> np.ndarray:
    try:
        phi_idx, psi_idx = ALA2_INDICES[variant]
    except KeyError as error:
        raise ValueError(f"Unknown Ala2 variant {variant!r}; choose from {sorted(ALA2_INDICES)}") from error
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"Expected coordinates (B,N,3), got {coordinates.shape}")
    required = max((*phi_idx, *psi_idx)) + 1
    if coordinates.shape[1] < required:
        raise ValueError(f"Variant {variant} requires at least {required} sites, got {coordinates.shape[1]}")
    phi = compute_dihedral(coordinates[:, phi_idx, :])
    psi = compute_dihedral(coordinates[:, psi_idx, :])
    return np.stack((phi, psi), axis=-1)


def prepare_ala2_dataset(
    source: str | Path | Mapping[str, Any],
    *,
    variant: str,
    kT: float,
    clip_percentile: float | None = None,
    clip_mode: str = "drop",
) -> Ala2Dataset:
    data = _load(source)
    if "R" not in data:
        raise KeyError("Ala2 evaluation input requires R")
    coordinates = np.asarray(data["R"])
    weights, logw, weight_source = resolve_weights(
        data,
        kT=kT,
        clip_percentile=clip_percentile,
        clip_mode=clip_mode,
    )
    if weights is not None and weights.shape != (coordinates.shape[0],):
        raise ValueError("Weight count does not match coordinate count")
    energy = None if "U" not in data else np.asarray(data["U"], dtype=np.float64).reshape(-1)
    if energy is not None and energy.shape != (coordinates.shape[0],):
        raise ValueError("Energy count does not match coordinate count")
    return Ala2Dataset(
        R=coordinates,
        dihedrals=ala2_dihedrals(coordinates, variant),
        weights=weights,
        logw=logw,
        energies=energy,
        weight_source=weight_source,
    )


def _free_energy_1d(samples: np.ndarray, grid: np.ndarray, kT: float, weights: np.ndarray | None) -> np.ndarray:
    try:
        from scipy.stats import gaussian_kde

        density = gaussian_kde(samples, weights=weights)(grid)
    except Exception:
        hist, edges = np.histogram(samples, bins=100, range=(-np.pi, np.pi), density=True, weights=weights)
        density = np.interp(grid, 0.5 * (edges[:-1] + edges[1:]), hist)
    free = -float(kT) * np.log(np.maximum(density, np.finfo(float).tiny))
    free -= np.nanmin(free)
    return np.where(free <= 22.5, free, np.nan)


def _plot_hist(
    ax,
    values,
    *,
    label: str,
    role: str | None = None,
    color: str | None = None,
    weights=None,
):
    mask = np.isfinite(values)
    if weights is not None:
        mask &= np.isfinite(weights)
        weights = weights[mask]
    density, edges = np.histogram(
        values[mask],
        bins=100,
        range=(-np.pi, np.pi),
        density=True,
        weights=weights,
    )
    if role is not None:
        filled_stairs(
            ax,
            density,
            edges,
            role=role,
            label=label,
        )
        return
    ax.stairs(
        density,
        edges,
        color=color,
        linewidth=1.5,
        label=label,
    )


def _energy_plot_window(
    reference_energy: np.ndarray,
    *,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
    padding_fraction: float = 0.05,
) -> tuple[float, float]:
    finite = np.asarray(reference_energy, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("reference energy contains no finite values")
    lower, upper = np.quantile(finite, (lower_quantile, upper_quantile))
    width = max(float(upper - lower), np.finfo(np.float64).eps)
    padding = padding_fraction * width
    return float(lower - padding), float(upper + padding)


def _energy_density_in_view(
    values: np.ndarray,
    bins: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if weights is None:
        normalized = np.full(values.shape, 1.0 / max(int(np.count_nonzero(finite)), 1))
    else:
        normalized = np.asarray(weights, dtype=np.float64).reshape(-1)
        if normalized.shape != values.shape:
            raise ValueError("energy weights do not match energy values")
        finite &= np.isfinite(normalized) & (normalized >= 0.0)
        total = float(np.sum(normalized[finite]))
        if not total > 0.0:
            raise ValueError("energy weights contain no positive finite mass")
        normalized = normalized / total
    counts, _ = np.histogram(values[finite], bins=bins, weights=normalized[finite])
    return counts / np.diff(bins)


def _energy_outside_fraction(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    weights: np.ndarray | None = None,
) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    outside = finite & ((values < lower) | (values > upper))
    if weights is None:
        return float(np.count_nonzero(outside) / max(np.count_nonzero(finite), 1))
    normalized = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite &= np.isfinite(normalized) & (normalized >= 0.0)
    total = float(np.sum(normalized[finite]))
    if not total > 0.0:
        return float("nan")
    return float(np.sum(normalized[outside]) / total)


def _plot_rama_fes(
    axis,
    angles: np.ndarray,
    weights: np.ndarray | None,
    *,
    title: str,
    kT: float,
):
    hist, xedge, yedge = np.histogram2d(
        angles[:, 0],
        angles[:, 1],
        bins=100,
        range=((-np.pi, np.pi), (-np.pi, np.pi)),
        density=True,
        weights=weights,
    )
    fes = -(kT / 4.184) * np.log(np.maximum(hist, np.finfo(float).tiny))
    fes -= np.nanmin(fes)
    image = axis.pcolormesh(
        xedge,
        yedge,
        fes.T,
        shading="auto",
        vmin=0.0,
        vmax=5.25,
        cmap="viridis",
    )
    axis.set(
        title=title,
        xlabel=r"$\phi$",
        ylabel=r"$\psi$",
        xlim=(-np.pi, np.pi),
        ylim=(-np.pi, np.pi),
    )
    return image


def _save_metric_report(path: Path, metrics: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    lines: list[str] = []
    for section, values in metrics.items():
        lines.append(f"[{section}]")
        if isinstance(values, Mapping):
            for name, value in values.items():
                lines.append(f"{name}: {value}")
        else:
            lines.append(str(values))
        lines.append("")
    path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")


def evaluate_ala2(
    *,
    target: str | Path | Mapping[str, Any],
    sample: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    variant: str = "ala2_cb",
    implicit: str | Path | Mapping[str, Any] | None = None,
    implicit_variant: str = "ala2_implicit",
    kT: float = 2.494338785445972,
    clip_percentile: float | None = None,
    clip_mode: str = "drop",
    n_bootstraps: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Create CG-BG-style Ala2 figures and weighted/unweighted reports.

    A proposal-only sample is fully supported: proposal figures and unweighted
    metrics are produced, while reweighted curves/metrics are marked unavailable.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_ds = prepare_ala2_dataset(target, variant=variant, kT=kT)
    sample_ds = prepare_ala2_dataset(
        sample,
        variant=variant,
        kT=kT,
        clip_percentile=clip_percentile,
        clip_mode=clip_mode,
    )
    implicit_ds = (
        None if implicit is None else prepare_ala2_dataset(implicit, variant=implicit_variant, kT=kT)
    )
    phi_t, psi_t = target_ds.dihedrals.T
    phi_s, psi_s = sample_ds.dihedrals.T

    images: dict[str, str] = {}
    # Energy distribution.  Formal metrics retain every sample.  The common
    # reference-quantile x range is display-only and its omitted mass is
    # recorded explicitly, preventing rare bad points from flattening the
    # scientifically relevant region.
    fig, ax = plt.subplots(figsize=(4, 4))
    energy_plot_view: dict[str, Any] | None = None
    if target_ds.energies is None or sample_ds.energies is None:
        ax.text(0.5, 0.5, "Energy unavailable", ha="center", va="center", transform=ax.transAxes)
    else:
        lower, upper = _energy_plot_window(target_ds.energies)
        energy_bins = np.linspace(lower, upper, 121)
        filled_stairs(
            ax,
            _energy_density_in_view(target_ds.energies, energy_bins),
            energy_bins,
            role="reference",
            label="Reference",
        )
        implicit_outside = None
        if implicit_ds is not None and implicit_ds.energies is not None:
            ax.stairs(
                _energy_density_in_view(implicit_ds.energies, energy_bins),
                energy_bins,
                color=IMPLICIT_REFERENCE_COLOR,
                linewidth=1.3,
                label="Implicit MD",
            )
            implicit_outside = _energy_outside_fraction(
                implicit_ds.energies,
                lower=lower,
                upper=upper,
            )
        filled_stairs(
            ax,
            _energy_density_in_view(sample_ds.energies, energy_bins),
            energy_bins,
            role="proposal",
            label="Proposal",
        )
        if sample_ds.weights is not None:
            filled_stairs(
                ax,
                _energy_density_in_view(
                    sample_ds.energies,
                    energy_bins,
                    weights=sample_ds.weights,
                ),
                energy_bins,
                role="reweighted",
                label="Reweighted",
            )
        reference_outside = _energy_outside_fraction(
            target_ds.energies,
            lower=lower,
            upper=upper,
        )
        proposal_outside = _energy_outside_fraction(
            sample_ds.energies,
            lower=lower,
            upper=upper,
        )
        reweighted_outside = (
            None
            if sample_ds.weights is None
            else _energy_outside_fraction(
                sample_ds.energies,
                lower=lower,
                upper=upper,
                weights=sample_ds.weights,
            )
        )
        annotation = (
            f"proposal outside={100.0 * proposal_outside:.2f}%"
        )
        if reweighted_outside is not None:
            annotation += (
                f"\nweighted outside={100.0 * reweighted_outside:.2f}%"
            )
        ax.text(
            0.98,
            0.98,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.set_xlim(lower, upper)
        ax.legend()
        energy_plot_view = {
            "display_only": True,
            "sample_usage": "all",
            "lower_quantile": 0.005,
            "upper_quantile": 0.995,
            "limits": [lower, upper],
            "reference_outside_fraction": reference_outside,
            "implicit_outside_fraction": implicit_outside,
            "proposal_outside_fraction": proposal_outside,
            "reweighted_outside_fraction": reweighted_outside,
        }
    ax.set_xlabel("Energy (kJ/mol)")
    ax.set_ylabel("Density")
    energy_path = output / f"{variant}_energy_distribution.png"
    fig.tight_layout()
    fig.savefig(energy_path, dpi=200)
    plt.close(fig)
    images["energy_distribution"] = str(energy_path)

    # Marginal densities.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    implicit_angles = None if implicit_ds is None else implicit_ds.dihedrals
    for axis, index, label in zip(axes, (0, 1), (r"$\phi$", r"$\psi$"), strict=True):
        _plot_hist(
            axis,
            target_ds.dihedrals[:, index],
            label="Explicit MD",
            role="reference",
        )
        if implicit_angles is not None:
            _plot_hist(
                axis,
                implicit_angles[:, index],
                label="Implicit MD",
                color=IMPLICIT_REFERENCE_COLOR,
            )
        _plot_hist(
            axis,
            sample_ds.dihedrals[:, index],
            label="Proposal",
            role="proposal",
        )
        if sample_ds.weights is not None:
            _plot_hist(
                axis,
                sample_ds.dihedrals[:, index],
                label="Reweighted",
                role="reweighted",
                weights=sample_ds.weights,
            )
        axis.set(xlabel=label, ylabel="Density", xlim=(-np.pi, np.pi))
        axis.legend(fontsize=7)
    density_path = output / f"{variant}_density.png"
    fig.tight_layout()
    fig.savefig(density_path, dpi=200)
    plt.close(fig)
    images["density"] = str(density_path)
    for index, coordinate_name, coordinate_label in (
        (0, "phi", r"$\phi$"),
        (1, "psi", r"$\psi$"),
    ):
        panel_path = output / f"{variant}_{coordinate_name}_density.png"
        panel_figure, panel_axis = plt.subplots(figsize=(5.2, 4.0))
        _plot_hist(
            panel_axis,
            target_ds.dihedrals[:, index],
            label="Reference",
            role="reference",
        )
        if implicit_angles is not None:
            _plot_hist(
                panel_axis,
                implicit_angles[:, index],
                label="Implicit MD",
                color=IMPLICIT_REFERENCE_COLOR,
            )
        _plot_hist(
            panel_axis,
            sample_ds.dihedrals[:, index],
            label="Proposal",
            role="proposal",
        )
        if sample_ds.weights is not None:
            _plot_hist(
                panel_axis,
                sample_ds.dihedrals[:, index],
                label="Reweighted",
                role="reweighted",
                weights=sample_ds.weights,
            )
        panel_axis.set(
            xlabel=coordinate_label,
            ylabel="Density",
            xlim=(-np.pi, np.pi),
        )
        panel_axis.legend(frameon=True)
        panel_figure.tight_layout()
        panel_figure.savefig(panel_path, dpi=220)
        plt.close(panel_figure)
        images[f"{coordinate_name}_density"] = str(panel_path)

    # One-dimensional free energies.
    grid = np.linspace(-np.pi, np.pi, 300)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis, index, label in zip(axes, (0, 1), (r"$\phi$", r"$\psi$"), strict=True):
        axis.plot(
            grid,
            _free_energy_1d(
                target_ds.dihedrals[:, index],
                grid,
                kT,
                None,
            ),
            color=REFERENCE_COLOR,
            label="Explicit MD",
        )
        if implicit_angles is not None:
            axis.plot(
                grid,
                _free_energy_1d(
                    implicit_angles[:, index],
                    grid,
                    kT,
                    None,
                ),
                color=IMPLICIT_REFERENCE_COLOR,
                label="Implicit MD",
            )
        axis.plot(
            grid,
            _free_energy_1d(
                sample_ds.dihedrals[:, index],
                grid,
                kT,
                None,
            ),
            "--",
            color=PROPOSAL_COLOR,
            label="Proposal",
        )
        if sample_ds.weights is not None:
            axis.plot(
                grid,
                _free_energy_1d(
                    sample_ds.dihedrals[:, index], grid, kT, sample_ds.weights
                ),
                "--",
                color=REWEIGHTED_COLOR,
                label="Reweighted",
            )
        axis.set(xlabel=label, ylabel="Free Energy", xlim=(-np.pi, np.pi))
        axis.legend(fontsize=7)
    free_path = output / f"{variant}_free_energy.png"
    fig.tight_layout()
    fig.savefig(free_path, dpi=200)
    plt.close(fig)
    images["free_energy"] = str(free_path)
    for index, coordinate_name, coordinate_label in (
        (0, "phi", r"$\phi$"),
        (1, "psi", r"$\psi$"),
    ):
        panel_path = output / f"{variant}_{coordinate_name}_free_energy.png"
        panel_figure, panel_axis = plt.subplots(figsize=(5.2, 4.0))
        panel_axis.plot(
            grid,
            _free_energy_1d(
                target_ds.dihedrals[:, index],
                grid,
                kT,
                None,
            ),
            color=REFERENCE_COLOR,
            label="Reference",
        )
        if implicit_angles is not None:
            panel_axis.plot(
                grid,
                _free_energy_1d(
                    implicit_angles[:, index],
                    grid,
                    kT,
                    None,
                ),
                color=IMPLICIT_REFERENCE_COLOR,
                label="Implicit MD",
            )
        panel_axis.plot(
            grid,
            _free_energy_1d(
                sample_ds.dihedrals[:, index],
                grid,
                kT,
                None,
            ),
            color=PROPOSAL_COLOR,
            linestyle="--",
            label="Proposal",
        )
        if sample_ds.weights is not None:
            panel_axis.plot(
                grid,
                _free_energy_1d(
                    sample_ds.dihedrals[:, index],
                    grid,
                    kT,
                    sample_ds.weights,
                ),
                color=REWEIGHTED_COLOR,
                linestyle="--",
                label="Reweighted",
            )
        panel_axis.set(
            xlabel=coordinate_label,
            ylabel="Free energy (kJ/mol)",
            xlim=(-np.pi, np.pi),
        )
        panel_axis.legend(frameon=True)
        panel_figure.tight_layout()
        panel_figure.savefig(panel_path, dpi=220)
        plt.close(panel_figure)
        images[f"{coordinate_name}_free_energy"] = str(panel_path)

    # Ramachandran free-energy maps.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = (
        (target_ds.dihedrals, None, "Reference"),
        (sample_ds.dihedrals, None, "Proposal"),
        (sample_ds.dihedrals, sample_ds.weights, "Reweighted"),
    )
    for axis, (angles, weights, title) in zip(axes, panels, strict=True):
        if title == "Reweighted" and weights is None:
            axis.text(
                0.5,
                0.5,
                "Proposal only\n(no weights)",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            _plot_rama_fes(
                axis,
                angles,
                weights,
                title=title,
                kT=kT,
            )
    rama_path = output / f"{variant}_ramachandran_fes.png"
    fig.tight_layout()
    fig.savefig(rama_path, dpi=200)
    plt.close(fig)
    images["ramachandran_fes"] = str(rama_path)
    for slug, (angles, weights, title) in zip(
        ("reference", "proposal", "reweighted"),
        panels,
        strict=True,
    ):
        panel_path = output / f"{variant}_{slug}_ramachandran_fes.png"
        panel_figure, panel_axis = plt.subplots(figsize=(4.8, 4.2))
        if slug == "reweighted" and weights is None:
            panel_axis.text(
                0.5,
                0.5,
                "Unavailable",
                ha="center",
                va="center",
                transform=panel_axis.transAxes,
            )
            panel_axis.set(
                title=title,
                xlabel=r"$\phi$",
                ylabel=r"$\psi$",
                xlim=(-np.pi, np.pi),
                ylim=(-np.pi, np.pi),
            )
        else:
            panel_image = _plot_rama_fes(
                panel_axis,
                angles,
                weights,
                title=title,
                kT=kT,
            )
            panel_figure.colorbar(
                panel_image,
                ax=panel_axis,
                label="Free energy (kcal/mol)",
            )
        panel_figure.tight_layout()
        panel_figure.savefig(panel_path, dpi=220)
        plt.close(panel_figure)
        images[f"{slug}_ramachandran_fes"] = str(panel_path)

    ranges = ((-np.pi, np.pi), (-np.pi, np.pi))
    unweighted_direct = distribution_metrics(target_ds.dihedrals, sample_ds.dihedrals, ranges=ranges)
    unweighted_boot = bootstrap_distribution_metrics(
        target_ds.dihedrals,
        sample_ds.dihedrals,
        ranges=ranges,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    metrics: dict[str, Any] = {
        "plot_style": plot_style_metadata(),
        "weight_source": sample_ds.weight_source,
        "unweighted_direct": unweighted_direct,
        "unweighted_bootstrap": summarize_bootstrap(unweighted_boot),
        "energy_plot_view": energy_plot_view,
    }
    if sample_ds.weights is not None:
        weighted_direct = distribution_metrics(
            target_ds.dihedrals, sample_ds.dihedrals, ranges=ranges, sample_weights=sample_ds.weights
        )
        weighted_boot = bootstrap_distribution_metrics(
            target_ds.dihedrals,
            sample_ds.dihedrals,
            ranges=ranges,
            sample_weights=sample_ds.weights,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        metrics["weighted_direct"] = weighted_direct
        metrics["weighted_bootstrap"] = summarize_bootstrap(weighted_boot)
        metrics["max_weight"] = float(np.max(sample_ds.weights))
        metrics["logw_variance"] = float(np.nanvar(sample_ds.logw)) if sample_ds.logw is not None else float("nan")
    else:
        metrics["weighted_direct"] = None
        metrics["weighted_bootstrap"] = None
    if target_ds.energies is not None and sample_ds.energies is not None:
        w1, w2 = energy_wasserstein(sample_ds.energies, target_ds.energies)
        metrics["energy_wasserstein"] = {"W1": w1, "W2": w2}
    metrics_path = output / f"{variant}_metrics.json"
    _save_metric_report(metrics_path, metrics)
    return {"images": images, "metrics": metrics, "metrics_path": str(metrics_path)}


def plot_ala2_clip_sweep(
    *,
    target: str | Path | Mapping[str, Any],
    sample: str | Path | Mapping[str, Any],
    output_path: str | Path,
    clip_percentiles: list[float] | tuple[float, ...],
    variant: str = "ala2_cb",
    kT: float = 2.494338785445972,
    clip_mode: str = "drop",
) -> dict[str, Any]:
    """Compute and actually save the clip-sweep figure missing in CG-BG."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    clips = np.asarray(clip_percentiles, dtype=float)
    if clips.ndim != 1 or clips.size == 0:
        raise ValueError("clip_percentiles must be a non-empty one-dimensional sequence")
    target_ds = prepare_ala2_dataset(target, variant=variant, kT=kT)
    raw = _load(sample)
    sample_unweighted = prepare_ala2_dataset(raw, variant=variant, kT=kT)
    metrics = np.empty((clips.size, 3), dtype=float)
    ranges = ((-np.pi, np.pi), (-np.pi, np.pi))
    for index, clip in enumerate(clips):
        weights, _, _ = resolve_weights(raw, kT=kT, clip_percentile=float(clip), clip_mode=clip_mode)
        summary = distribution_metrics(
            target_ds.dihedrals,
            sample_unweighted.dihedrals,
            ranges=ranges,
            sample_weights=weights,
        )
        metrics[index] = summary["JS_Divergence"], summary["PMF_Error"], summary["ESS_Percent"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, values, label in zip(
        axes,
        metrics.T,
        ("JS Divergence", "PMF Error", "ESS / N"),
        strict=True,
    ):
        axis.plot(clips, values, marker="o")
        axis.set(xlabel="Clip percentile", ylabel=label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        clips=clips,
        metrics=metrics,
        metric_names=np.asarray(["JS_Divergence", "PMF_Error", "ESS_Percent"]),
    )
    return {"output_path": str(output_path), "clips": clips, "metrics": metrics}
