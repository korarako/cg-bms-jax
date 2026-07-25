"""Muller--Brown CG1D evaluation with analytic reference marginal."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cg_bms_jax.potential.mb import muller_brown_energy

from .metrics import (
    bootstrap_distribution_metrics,
    distribution_metrics,
    resolve_weights,
    summarize_bootstrap,
)
from .plot_style import (
    EXACT_COLOR,
    PROPOSAL_COLOR,
    REFERENCE_COLOR,
    REWEIGHTED_COLOR,
    filled_stairs,
    plot_style_metadata,
)


def _load(source: str | Path | Mapping[str, Any]) -> dict[str, np.ndarray]:
    if isinstance(source, Mapping):
        return {key: np.asarray(value) for key, value in source.items()}
    with np.load(source, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _extract_mb_x(coordinates: Any, *, full_reference: bool) -> np.ndarray:
    """Return the one-dimensional MB coarse coordinate without mixing in ``y``.

    CG-BMS proposals already live in the one-dimensional coarse space and are
    stored as either ``(N,)`` or singleton event shapes such as ``(N, 1)``.
    The pinned CG-BG unbiased reference instead stores the original full MB
    trajectory as ``(N, 2)``.  Its coarse coordinate is the first column.
    """

    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.ndim >= 2 and int(np.prod(values.shape[1:])) == 1:
        return values.reshape(values.shape[0])
    if full_reference and values.ndim == 2 and values.shape[1] == 2:
        return values[:, 0]
    role = "reference" if full_reference else "proposal"
    raise ValueError(
        f"MB {role} coordinates must have shape (N,), a singleton event shape,"
        f"{', or (N,2) for the full reference' if full_reference else ''}; got {values.shape}"
    )


def mb_exact_marginal(
    x_grid: Any,
    *,
    kT: float = 1.0,
    y_grid: Any | None = None,
) -> np.ndarray:
    """Numerically integrate the analytic two-dimensional MB density over y."""

    x = np.asarray(x_grid, dtype=np.float64)
    y = np.linspace(0.0, 50.0, 500) if y_grid is None else np.asarray(y_grid, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy = np.stack((xx, yy), axis=-1)
    # Use the same coefficients as the JAX potential without requiring callers
    # to know that the CG checkpoint itself is one dimensional.
    energy = np.asarray(muller_brown_energy(xy))
    shifted = energy - np.min(energy, axis=1, keepdims=True)
    conditional = np.exp(-shifted / float(kT)) * np.exp(-np.min(energy, axis=1)[:, None] / float(kT))
    marginal = np.trapz(conditional, y, axis=1)
    normalizer = np.trapz(marginal, x)
    if not normalizer > 0:
        raise ValueError("Analytic Muller--Brown marginal failed to normalize")
    return marginal / normalizer


def _kde_free_energy(samples: np.ndarray, grid: np.ndarray, kT: float, weights: np.ndarray | None) -> np.ndarray:
    try:
        from scipy.stats import gaussian_kde

        density = gaussian_kde(samples, weights=weights)(grid)
    except Exception:
        hist, edges = np.histogram(samples, bins=100, range=(grid.min(), grid.max()), density=True, weights=weights)
        density = np.interp(grid, 0.5 * (edges[:-1] + edges[1:]), hist)
    free = -float(kT) * np.log(np.maximum(density, np.finfo(float).tiny))
    return free - np.nanmin(free)


def _histogram_density(
    values: np.ndarray,
    *,
    bins: int,
    value_range: tuple[float, float],
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    density, edges = np.histogram(
        values,
        bins=bins,
        range=value_range,
        density=True,
        weights=weights,
    )
    return density, edges


def evaluate_mb(
    *,
    target: str | Path | Mapping[str, Any],
    sample: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    kT: float = 1.0,
    clip_percentile: float | None = None,
    clip_mode: str = "drop",
    n_bootstraps: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Save CG-BG's two-panel MB plot plus quantitative bootstrap metrics."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    target_data = _load(target)
    sample_data = _load(sample)
    if "R" not in target_data or "R" not in sample_data:
        raise KeyError("MB evaluation requires R in target and sample")
    x_ref = _extract_mb_x(target_data["R"], full_reference=True)
    x_gen = _extract_mb_x(sample_data["R"], full_reference=False)
    weights, logw, weight_source = resolve_weights(
        sample_data,
        kT=kT,
        clip_percentile=clip_percentile,
        clip_mode=clip_mode,
    )
    if weights is not None and weights.shape != x_gen.shape:
        raise ValueError("MB weights do not match sample coordinates")

    x_grid = np.linspace(0.0, 50.0, 300)
    exact_density = mb_exact_marginal(x_grid, kT=kT)
    exact_free = -float(kT) * np.log(np.maximum(exact_density, np.finfo(float).tiny))
    exact_free -= exact_free.min()
    ref_free = _kde_free_energy(x_ref, x_grid, kT, None)
    gen_free = _kde_free_energy(x_gen, x_grid, kT, None)
    rew_free = None if weights is None else _kde_free_energy(x_gen, x_grid, kT, weights)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / "mb_plots.png"
    fig, (ax_pdf, ax_fe) = plt.subplots(1, 2, figsize=(8, 4))
    reference_density, density_edges = _histogram_density(
        x_ref,
        bins=100,
        value_range=(0.0, 50.0),
    )
    proposal_density, _ = _histogram_density(
        x_gen,
        bins=100,
        value_range=(0.0, 50.0),
    )
    reweighted_density = (
        None
        if weights is None
        else _histogram_density(
            x_gen,
            bins=100,
            value_range=(0.0, 50.0),
            weights=weights,
        )[0]
    )
    filled_stairs(
        ax_pdf,
        reference_density,
        density_edges,
        role="reference",
        label="MD Reference",
    )
    filled_stairs(
        ax_pdf,
        proposal_density,
        density_edges,
        role="proposal",
        label="Proposal",
    )
    if reweighted_density is not None:
        filled_stairs(
            ax_pdf,
            reweighted_density,
            density_edges,
            role="reweighted",
            label="Reweighted",
        )
    ax_pdf.plot(x_grid, exact_density, color=EXACT_COLOR, linewidth=1.8, label="Exact")
    ax_pdf.set(xlabel="x", ylabel="P(x)", xlim=(10, 50))
    ax_pdf.legend(fontsize=7)
    ax_fe.plot(x_grid, exact_free, color=EXACT_COLOR, label="Exact")
    ax_fe.plot(x_grid, ref_free, color=REFERENCE_COLOR, label="MD Reference")
    ax_fe.plot(x_grid, gen_free, "--", color=PROPOSAL_COLOR, label="Proposal")
    if rew_free is not None:
        ax_fe.plot(x_grid, rew_free, "--", color=REWEIGHTED_COLOR, label="Reweighted")
    ax_fe.set(xlabel="x", ylabel="Free Energy", xlim=(10, 50), ylim=(-0.5, 15))
    ax_fe.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    density_path = output / "mb_density.png"
    density_figure, density_axis = plt.subplots(figsize=(5.2, 4.0))
    filled_stairs(
        density_axis,
        reference_density,
        density_edges,
        role="reference",
        label="Reference",
    )
    filled_stairs(
        density_axis,
        proposal_density,
        density_edges,
        role="proposal",
        label="Proposal",
    )
    if reweighted_density is not None:
        filled_stairs(
            density_axis,
            reweighted_density,
            density_edges,
            role="reweighted",
            label="Reweighted",
        )
    density_axis.plot(
        x_grid,
        exact_density,
        color=EXACT_COLOR,
        linewidth=1.8,
        label="Exact",
    )
    density_axis.set(xlabel="x", ylabel="P(x)", xlim=(10, 50))
    density_axis.legend(frameon=True)
    density_figure.tight_layout()
    density_figure.savefig(density_path, dpi=220)
    plt.close(density_figure)

    free_energy_path = output / "mb_free_energy.png"
    free_energy_figure, free_energy_axis = plt.subplots(figsize=(5.2, 4.0))
    free_energy_axis.plot(x_grid, exact_free, color=EXACT_COLOR, label="Exact")
    free_energy_axis.plot(
        x_grid,
        ref_free,
        color=REFERENCE_COLOR,
        label="Reference",
    )
    free_energy_axis.plot(
        x_grid,
        gen_free,
        color=PROPOSAL_COLOR,
        linestyle="--",
        label="Proposal",
    )
    if rew_free is not None:
        free_energy_axis.plot(
            x_grid,
            rew_free,
            color=REWEIGHTED_COLOR,
            linestyle="--",
            label="Reweighted",
        )
    free_energy_axis.set(
        xlabel="x",
        ylabel="Free energy",
        xlim=(10, 50),
        ylim=(-0.5, 15),
    )
    free_energy_axis.legend(frameon=True)
    free_energy_figure.tight_layout()
    free_energy_figure.savefig(free_energy_path, dpi=220)
    plt.close(free_energy_figure)

    ranges = ((0.0, 50.0),)
    unweighted_direct = distribution_metrics(x_ref, x_gen, ranges=ranges)
    unweighted_boot = bootstrap_distribution_metrics(
        x_ref,
        x_gen,
        ranges=ranges,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    metrics: dict[str, Any] = {
        "plot_style": plot_style_metadata(),
        "weight_source": weight_source,
        "unweighted_direct": unweighted_direct,
        "unweighted_bootstrap": summarize_bootstrap(unweighted_boot),
        "weighted_direct": None,
        "weighted_bootstrap": None,
    }
    if weights is not None:
        metrics["weighted_direct"] = distribution_metrics(x_ref, x_gen, ranges=ranges, sample_weights=weights)
        weighted_boot = bootstrap_distribution_metrics(
            x_ref,
            x_gen,
            ranges=ranges,
            sample_weights=weights,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        metrics["weighted_bootstrap"] = summarize_bootstrap(weighted_boot)
        metrics["max_weight"] = float(np.max(weights))
        metrics["logw_variance"] = float(np.nanvar(logw)) if logw is not None else float("nan")
    metrics_path = output / "mb_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    return {
        "images": {
            "mb_plots": str(output_path),
            "density": str(density_path),
            "free_energy": str(free_energy_path),
        },
        "metrics": metrics,
        "metrics_path": str(metrics_path),
    }
