"""Analytic Muller--Brown 2D evaluation and formal reweighting sweeps.

The CG1D experiment intentionally evaluates a learned marginal PMF.  This
module instead treats the original two-dimensional Muller--Brown energy as the
target, so reference probabilities, basin masses, and temperature changes can
be computed without an MD reference trajectory.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import importance_weight_diagnostics, normalized_weights, resolve_weights
from .plot_style import (
    EXACT_COLOR,
    filled_curve,
    filled_stairs,
    plot_style_metadata,
)

DEFAULT_DOMAIN = ((0.0, 50.0), (0.0, 50.0))
DEFAULT_BASIN_CENTERS = {
    # Standard Muller--Brown minima transformed by
    # x_phys=16*(x+2), y_phys=16*(y+0.5), matching CG-BG's coordinates.
    "upper_left": (23.072, 31.072),
    "center": (31.200, 15.472),
    "lower_right": (41.968, 8.448),
}
EXACT_ENERGY_DISPLAY_SMOOTH_SIGMA_BINS = 2.0


@dataclass(frozen=True)
class MB2DReferenceGrid:
    """Cell-centred quadrature of the normalized analytic target."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    energy: np.ndarray
    probability: np.ndarray
    beta: float

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (float(self.x_edges[0]), float(self.x_edges[-1])),
            (float(self.y_edges[0]), float(self.y_edges[-1])),
        )

    @property
    def points(self) -> np.ndarray:
        xx, yy = np.meshgrid(self.x_centers, self.y_centers, indexing="ij")
        return np.stack((xx, yy), axis=-1)


def muller_brown_energy_numpy(coordinates: Any, *, biased: bool = False) -> np.ndarray:
    """NumPy equivalent of :func:`potential.mb.muller_brown_energy`."""

    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.shape[-1:] != (2,):
        raise ValueError(f"Expected Muller--Brown coordinates (...,2), got {xy.shape}")
    x, y = xy[..., 0], xy[..., 1]
    energy = (
        -17.3 * np.exp(-0.0039 * (x - 48.0) ** 2 - 0.0391 * (y - 8.0) ** 2)
        - 8.7 * np.exp(-0.0039 * (x - 32.0) ** 2 - 0.0391 * (y - 16.0) ** 2)
        - 14.7
        * np.exp(
            -0.0254 * (x - 24.0) ** 2
            + 0.043 * (x - 24.0) * (y - 32.0)
            - 0.0254 * (y - 32.0) ** 2
        )
        + 1.3
        * np.exp(
            0.00273 * (x - 16.0) ** 2
            + 0.0023 * (x - 16.0) * (y - 24.0)
            + 0.00273 * (y - 24.0) ** 2
        )
    )
    if biased:
        energy = energy - 4.0 * np.exp(-((x - 32.0) ** 2) / (2.0 * 5.0**2))
    return energy


def analytic_mb2d_reference(
    *,
    beta: float = 1.0,
    bins: int | tuple[int, int] = 160,
    domain: Sequence[Sequence[float]] = DEFAULT_DOMAIN,
) -> MB2DReferenceGrid:
    """Build a normalized cell-centred quadrature on a finite physical box."""

    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    if isinstance(bins, int):
        bins = (bins, bins)
    if len(bins) != 2 or any(int(value) < 4 for value in bins):
        raise ValueError("bins must contain two integers >= 4")
    if len(domain) != 2:
        raise ValueError("domain must contain x and y limits")
    limits = tuple((float(axis[0]), float(axis[1])) for axis in domain)
    if any(
        not np.isfinite(axis).all() or axis[0] >= axis[1]
        for axis in map(np.asarray, limits)
    ):
        raise ValueError("domain limits must be finite and increasing")
    x_edges = np.linspace(*limits[0], int(bins[0]) + 1)
    y_edges = np.linspace(*limits[1], int(bins[1]) + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    points = np.stack((xx, yy), axis=-1)
    energy = muller_brown_energy_numpy(points)
    log_mass = -beta * energy
    mass = np.exp(log_mass - np.max(log_mass))
    probability = mass / np.sum(mass)
    return MB2DReferenceGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        energy=energy,
        probability=probability,
        beta=beta,
    )


def _load(source: str | Path | Mapping[str, Any]) -> dict[str, np.ndarray]:
    if isinstance(source, Mapping):
        return {str(key): np.asarray(value) for key, value in source.items()}
    with np.load(source, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _resolve_mb2d_weights(
    data: Mapping[str, Any],
    *,
    kT: float,
    clip_percentile: float | None,
    clip_mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Resolve formal weights or an explicit CG-BG-style clip diagnostic."""

    if clip_percentile is None:
        return resolve_weights(data, kT=kT)
    diagnostic = {
        str(name): np.asarray(value)
        for name, value in data.items()
        if name not in {"weights", "logw"}
    }
    raw_source = "computed_from_U_logp"
    if "logw_raw" in data:
        diagnostic["logw"] = np.asarray(data["logw_raw"])
        raw_source = "logw_raw"
    elif "logw" in data:
        diagnostic["logw"] = np.asarray(data["logw"])
        raw_source = "logw"
    weights, raw_logw, _ = resolve_weights(
        diagnostic,
        kT=kT,
        clip_percentile=float(clip_percentile),
        clip_mode=clip_mode,
    )
    if weights is None:
        raise ValueError(
            "Clipped MB2D evaluation requires logw_raw/logw or U with logp"
        )
    return (
        weights,
        raw_logw,
        f"{raw_source}_{clip_mode}_p{float(clip_percentile):g}",
    )


def extract_mb2d_coordinates(coordinates: Any) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 2:
        return values
    if values.ndim > 2 and int(np.prod(values.shape[1:])) == 2:
        return values.reshape(values.shape[0], 2)
    raise ValueError(
        f"MB2D coordinates must contain two event dimensions, got {values.shape}"
    )


def _normalized_histogram(
    coordinates: np.ndarray,
    reference: MB2DReferenceGrid,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    coordinates = extract_mb2d_coordinates(coordinates)
    hist, _, _ = np.histogram2d(
        coordinates[:, 0],
        coordinates[:, 1],
        bins=(reference.x_edges, reference.y_edges),
        weights=weights,
    )
    total_input = (
        float(coordinates.shape[0]) if weights is None else float(np.sum(weights))
    )
    retained = float(np.sum(hist))
    if retained <= 0.0:
        raise ValueError("No proposal mass lies inside the analytic MB2D domain")
    return hist / retained, retained / total_input


def _probability_js(
    left: np.ndarray, right: np.ndarray, *, baseline: float = 1.0e-15
) -> float:
    p = np.asarray(left, dtype=np.float64).reshape(-1) + baseline
    q = np.asarray(right, dtype=np.float64).reshape(-1) + baseline
    p /= np.sum(p)
    q /= np.sum(q)
    mixture = 0.5 * (p + q)
    return float(
        0.5 * (np.sum(p * np.log(p / mixture)) + np.sum(q * np.log(q / mixture)))
    )


def _pmf_rmse(
    exact_probability: np.ndarray,
    estimated_probability: np.ndarray,
    *,
    beta: float,
    probability_floor: float = 1.0e-12,
) -> float:
    """Probability-weighted free-energy RMSE after fitting one additive shift.

    A fixed probability floor makes the histogram diagnostic finite and
    comparable across sample sizes.  It is an evaluation convention only and
    never changes formal importance weights.
    """

    p = np.asarray(exact_probability, dtype=np.float64).reshape(-1)
    q = np.asarray(estimated_probability, dtype=np.float64).reshape(-1)
    if not 0.0 < probability_floor < 1.0:
        raise ValueError("probability_floor must lie in (0,1)")
    support = p > np.max(p) * 1.0e-8
    p_support = p[support]
    p_support /= np.sum(p_support)
    difference = (
        -np.log(np.maximum(q[support], probability_floor))
        + np.log(np.maximum(p[support], probability_floor))
    ) / float(beta)
    difference -= np.sum(p_support * difference)
    return float(np.sqrt(np.sum(p_support * difference**2)))


def _basin_indices(
    coordinates: np.ndarray,
    basin_centers: Mapping[str, Sequence[float]],
) -> np.ndarray:
    centers = np.asarray(list(basin_centers.values()), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] < 2:
        raise ValueError("basin_centers must define at least two 2D centres")
    points = extract_mb2d_coordinates(coordinates)
    return np.argmin(
        np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=-1), axis=1
    )


def _basin_masses(
    coordinates: np.ndarray,
    *,
    basin_centers: Mapping[str, Sequence[float]],
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    assignment = _basin_indices(coordinates, basin_centers)
    names = tuple(basin_centers)
    sample_weights = (
        np.full(assignment.shape, 1.0 / assignment.size)
        if weights is None
        else normalized_weights(weights)
    )
    return {
        name: float(np.sum(sample_weights[assignment == index]))
        for index, name in enumerate(names)
    }


def _exact_basin_masses(
    reference: MB2DReferenceGrid,
    basin_centers: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    return _basin_masses(
        reference.points.reshape(-1, 2),
        basin_centers=basin_centers,
        weights=reference.probability.reshape(-1),
    )


def mb2d_distribution_metrics(
    coordinates: Any,
    *,
    reference: MB2DReferenceGrid,
    weights: Any | None = None,
    basin_centers: Mapping[str, Sequence[float]] = DEFAULT_BASIN_CENTERS,
) -> dict[str, Any]:
    """Compare one proposal view to the exact analytic grid."""

    points = extract_mb2d_coordinates(coordinates)
    normalized = None if weights is None else normalized_weights(weights)
    histogram, in_domain_mass = _normalized_histogram(points, reference, normalized)
    exact_basins = _exact_basin_masses(reference, basin_centers)
    sample_basins = _basin_masses(
        points,
        basin_centers=basin_centers,
        weights=normalized,
    )
    sample_energy = muller_brown_energy_numpy(points)
    exact_energy_mean = float(np.sum(reference.probability * reference.energy))
    sample_energy_mean = float(
        np.mean(sample_energy)
        if normalized is None
        else np.sum(normalized * sample_energy)
    )
    return {
        "js_2d": _probability_js(reference.probability, histogram),
        "pmf_rmse": _pmf_rmse(
            reference.probability,
            histogram,
            beta=reference.beta,
        ),
        "in_domain_mass": in_domain_mass,
        "energy_mean": sample_energy_mean,
        "exact_energy_mean": exact_energy_mean,
        "energy_mean_error": sample_energy_mean - exact_energy_mean,
        "basin_mass": sample_basins,
        "exact_basin_mass": exact_basins,
        "basin_l1_error": float(
            sum(abs(sample_basins[name] - exact_basins[name]) for name in exact_basins)
        ),
    }


def _free_energy(probability: np.ndarray, beta: float) -> np.ndarray:
    free = -np.log(np.maximum(probability, np.finfo(np.float64).tiny)) / beta
    return free - np.min(free)


def _weighted_quantile(
    values: Any,
    quantile: float,
    *,
    weights: Any | None = None,
) -> float:
    """Return a finite (optionally probability-weighted) scalar quantile."""

    q = float(quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must lie in [0,1]")
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        finite = np.isfinite(data)
        if not finite.any():
            raise ValueError("quantile input contains no finite values")
        return float(np.quantile(data[finite], q))

    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    if mass.shape != data.shape:
        raise ValueError(
            "weighted quantile values and weights must have the same shape"
        )
    valid = np.isfinite(data) & np.isfinite(mass) & (mass > 0.0)
    if not valid.any():
        raise ValueError("weighted quantile contains no positive finite mass")
    data = data[valid]
    mass = mass[valid]
    order = np.argsort(data, kind="stable")
    data = data[order]
    mass = mass[order]
    cumulative = np.cumsum(mass)
    cumulative /= cumulative[-1]
    return float(np.interp(q, cumulative, data))


def _energy_mass_summary(
    values: Any,
    *,
    lower: float,
    upper: float,
    weights: Any | None = None,
) -> dict[str, float]:
    """Account for every unit of probability mass relative to a display view."""

    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        mass = np.full(data.shape, 1.0 / max(data.size, 1), dtype=np.float64)
    else:
        mass = normalized_weights(weights)
        if mass.shape != data.shape:
            raise ValueError("energy values and weights must have the same shape")
    finite = np.isfinite(data)
    nonfinite_mass = float(np.sum(mass[~finite]))
    below_mass = float(np.sum(mass[finite & (data < lower)]))
    above_mass = float(np.sum(mass[finite & (data > upper)]))
    visible_mass = float(np.sum(mass[finite & (data >= lower) & (data <= upper)]))
    return {
        "below_view_mass": below_mass,
        "above_view_mass": above_mass,
        "nonfinite_mass": nonfinite_mass,
        "out_of_view_mass": below_mass + above_mass + nonfinite_mass,
        "visible_mass": visible_mass,
    }


def _energy_density_in_view(
    values: Any,
    edges: np.ndarray,
    *,
    weights: Any | None = None,
) -> np.ndarray:
    """Histogram density whose integral equals visible, not conditional, mass."""

    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        mass = np.full(data.shape, 1.0 / max(data.size, 1), dtype=np.float64)
    else:
        mass = normalized_weights(weights)
        if mass.shape != data.shape:
            raise ValueError("energy values and weights must have the same shape")
    finite = np.isfinite(data) & np.isfinite(mass)
    counts, _ = np.histogram(data[finite], bins=edges, weights=mass[finite])
    return counts / np.diff(edges)


def _smooth_density_for_display(
    density: Any,
    *,
    sigma_bins: float = EXACT_ENERGY_DISPLAY_SMOOTH_SIGMA_BINS,
) -> np.ndarray:
    """Return a mass-preserving Gaussian-bin smoothing for display only.

    The analytic MB2D energy reference is evaluated on a regular coordinate
    grid. Directly drawing its weighted energy histogram produces visually
    dominant grid spikes even though the underlying target is continuous.
    This helper smooths only the plotted exact-energy curve; formal metrics and
    the raw machine-readable histogram remain unchanged.
    """

    values = np.asarray(density, dtype=np.float64).reshape(-1)
    sigma = float(sigma_bins)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_bins must be finite and positive")
    if values.size < 2 or not np.any(values > 0.0):
        return values.copy()
    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    raw_mass = float(np.sum(values))
    smooth_mass = float(np.sum(smoothed))
    if raw_mass > 0.0 and smooth_mass > 0.0:
        smoothed *= raw_mass / smooth_mass
    return smoothed


def _energy_plot_view(
    reference_energy: Any,
    reference_weights: Any,
    *,
    lower_quantile: float,
    upper_quantile: float,
    padding_fraction: float,
) -> tuple[float, float]:
    """Target-anchored robust energy view used only for visualization."""

    lower_q = float(lower_quantile)
    upper_q = float(upper_quantile)
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("energy view quantiles must satisfy 0 <= lower < upper <= 1")
    if not np.isfinite(padding_fraction) or float(padding_fraction) < 0.0:
        raise ValueError("energy view padding_fraction must be finite and non-negative")
    lower = _weighted_quantile(
        reference_energy,
        lower_q,
        weights=reference_weights,
    )
    upper = _weighted_quantile(
        reference_energy,
        upper_q,
        weights=reference_weights,
    )
    width = upper - lower
    if not np.isfinite(width) or width <= 0.0:
        scale = max(abs(lower), abs(upper), 1.0)
        width = 1.0e-3 * scale
    padding = float(padding_fraction) * width
    return lower - padding, upper + padding


def evaluate_mb2d(
    *,
    sample: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    kT: float = 1.0,
    bins: int = 120,
    domain: Sequence[Sequence[float]] = DEFAULT_DOMAIN,
    basin_centers: Mapping[str, Sequence[float]] = DEFAULT_BASIN_CENTERS,
    energy_lower_quantile: float = 0.005,
    energy_upper_quantile: float = 0.995,
    energy_view_padding: float = 0.05,
    energy_histogram_bins: int = 160,
    clip_percentile: float | None = None,
    clip_mode: str = "drop",
    weight_view_label: str | None = None,
) -> dict[str, Any]:
    """Write exact/proposal/reweighted MB2D plots and metrics.

    The default is the authoritative no-clip result. Passing
    ``clip_percentile`` creates an explicitly labelled compatibility
    diagnostic from raw log weights.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not np.isfinite(kT) or float(kT) <= 0.0:
        raise ValueError("kT must be finite and positive")
    data = _load(sample)
    if "R" not in data:
        raise KeyError("MB2D evaluation requires R")
    coordinates = extract_mb2d_coordinates(data["R"])
    reference = analytic_mb2d_reference(
        beta=1.0 / float(kT),
        bins=bins,
        domain=domain,
    )
    weights, resolved_logw, weight_source = _resolve_mb2d_weights(
        data,
        kT=float(kT),
        clip_percentile=clip_percentile,
        clip_mode=clip_mode,
    )
    if weights is not None and weights.shape != (coordinates.shape[0],):
        raise ValueError("MB2D weights do not match the sample dimension")
    unweighted_hist, _ = _normalized_histogram(coordinates, reference)
    weighted_hist = (
        None
        if weights is None
        else _normalized_histogram(coordinates, reference, weights)[0]
    )
    unweighted = mb2d_distribution_metrics(
        coordinates,
        reference=reference,
        basin_centers=basin_centers,
    )
    weighted = (
        None
        if weights is None
        else mb2d_distribution_metrics(
            coordinates,
            reference=reference,
            weights=weights,
            basin_centers=basin_centers,
        )
    )
    raw_logw = data.get("logw_raw", resolved_logw)
    weight_metrics = (
        None
        if weights is None
        else importance_weight_diagnostics(weights, logw_raw=raw_logw)
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "mb2d_plots.png"
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    extent = (
        reference.x_edges[0],
        reference.x_edges[-1],
        reference.y_edges[0],
        reference.y_edges[-1],
    )
    fes_views = (
        ("Exact", _free_energy(reference.probability, reference.beta)),
        ("Proposal", _free_energy(unweighted_hist, reference.beta)),
        (
            "Reweighted" if weighted_hist is not None else "Reweighted (unavailable)",
            None
            if weighted_hist is None
            else _free_energy(weighted_hist, reference.beta),
        ),
    )
    image = None
    for axis, (title, free) in zip(axes[0], fes_views, strict=True):
        if free is None:
            axis.text(0.5, 0.5, "No proposal density", ha="center", va="center")
            axis.set_axis_off()
            continue
        image = axis.imshow(
            np.minimum(free.T, 12.0 * float(kT)),
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=12.0 * float(kT),
        )
        axis.set(title=title, xlabel="x", ylabel="y")
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), label="Free energy")

    exact_x = np.sum(reference.probability, axis=1)
    exact_y = np.sum(reference.probability, axis=0)
    proposal_x = np.sum(unweighted_hist, axis=1)
    proposal_y = np.sum(unweighted_hist, axis=0)
    filled_curve(
        axes[1, 0],
        reference.x_centers,
        proposal_x,
        role="proposal",
        label="Proposal",
    )
    if weighted_hist is not None:
        filled_curve(
            axes[1, 0],
            reference.x_centers,
            np.sum(weighted_hist, axis=1),
            role="reweighted",
            label="Reweighted",
        )
    axes[1, 0].plot(
        reference.x_centers,
        exact_x,
        color=EXACT_COLOR,
        linewidth=1.8,
        label="Exact",
    )
    axes[1, 0].set(xlabel="x", ylabel="Probability / bin", title="x marginal")
    axes[1, 0].legend(fontsize=8)
    filled_curve(
        axes[1, 1],
        reference.y_centers,
        proposal_y,
        role="proposal",
        label="Proposal",
    )
    if weighted_hist is not None:
        filled_curve(
            axes[1, 1],
            reference.y_centers,
            np.sum(weighted_hist, axis=0),
            role="reweighted",
            label="Reweighted",
        )
    axes[1, 1].plot(
        reference.y_centers,
        exact_y,
        color=EXACT_COLOR,
        linewidth=1.8,
        label="Exact",
    )
    axes[1, 1].set(xlabel="y", ylabel="Probability / bin", title="y marginal")
    axes[1, 1].legend(fontsize=8)

    sample_energy = muller_brown_energy_numpy(coordinates)
    exact_energy = reference.energy.reshape(-1)
    exact_weights = reference.probability.reshape(-1)
    lower, upper = _energy_plot_view(
        exact_energy,
        exact_weights,
        lower_quantile=energy_lower_quantile,
        upper_quantile=energy_upper_quantile,
        padding_fraction=energy_view_padding,
    )
    if int(energy_histogram_bins) < 10:
        raise ValueError("energy_histogram_bins must be at least 10")
    energy_bins = np.linspace(lower, upper, int(energy_histogram_bins) + 1)
    exact_energy_density = _energy_density_in_view(
        exact_energy,
        energy_bins,
        weights=exact_weights,
    )
    exact_energy_display_density = _smooth_density_for_display(
        exact_energy_density,
    )
    exact_energy_view = _energy_mass_summary(
        exact_energy,
        lower=lower,
        upper=upper,
        weights=exact_weights,
    )
    proposal_energy_view = _energy_mass_summary(
        sample_energy,
        lower=lower,
        upper=upper,
    )
    weighted_energy_view = (
        None
        if weights is None
        else _energy_mass_summary(
            sample_energy,
            lower=lower,
            upper=upper,
            weights=weights,
        )
    )
    filled_stairs(
        axes[1, 2],
        _energy_density_in_view(sample_energy, energy_bins),
        energy_bins,
        role="proposal",
        label="Proposal",
    )
    if weights is not None:
        filled_stairs(
            axes[1, 2],
            _energy_density_in_view(
                sample_energy,
                energy_bins,
                weights=weights,
            ),
            energy_bins,
            role="reweighted",
            label="Reweighted",
        )
    axes[1, 2].stairs(
        exact_energy_display_density,
        energy_bins,
        color=EXACT_COLOR,
        linewidth=1.5,
        label="Exact",
    )
    annotation = (
        f"all N={coordinates.shape[0]:,} samples\n"
        f"proposal outside={100.0 * proposal_energy_view['out_of_view_mass']:.2f}%"
    )
    if weighted_energy_view is not None:
        annotation += (
            "\n"
            f"weighted outside={100.0 * weighted_energy_view['out_of_view_mass']:.2f}%"
        )
    axes[1, 2].text(
        0.98,
        0.98,
        annotation,
        transform=axes[1, 2].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    axes[1, 2].set(xlabel="Energy", ylabel="Density", title="Energy distribution")
    axes[1, 2].set_xlim(lower, upper)
    axes[1, 2].legend(fontsize=8)
    title = f"Analytic MB2D evaluation — all {coordinates.shape[0]:,} samples"
    if weight_view_label:
        title += f"\n{weight_view_label}"
    elif clip_percentile is not None:
        title += (
            f"\nDiagnostic: {clip_mode} top "
            f"{100.0 - float(clip_percentile):g}% raw log-weights"
        )
    fig.suptitle(title, fontsize=12)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    # Release figures are saved as true standalone panels rather than crops of
    # the diagnostic grid.  All density/weight calculations above are reused,
    # so the standalone and combined views have identical scientific content.
    standalone_images: dict[str, str] = {}
    for slug, (panel_title, free) in zip(
        ("exact", "proposal", "reweighted"),
        fes_views,
        strict=True,
    ):
        panel_path = output / f"mb2d_{slug}_fes.png"
        panel_figure, panel_axis = plt.subplots(figsize=(4.8, 4.2))
        if free is None:
            panel_axis.text(
                0.5,
                0.5,
                "Unavailable",
                ha="center",
                va="center",
                transform=panel_axis.transAxes,
            )
        else:
            panel_image = panel_axis.imshow(
                np.minimum(free.T, 12.0 * float(kT)),
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="viridis",
                vmin=0.0,
                vmax=12.0 * float(kT),
            )
            panel_figure.colorbar(
                panel_image,
                ax=panel_axis,
                label="Free energy",
            )
        panel_axis.set(
            title=panel_title,
            xlabel="x",
            ylabel="y",
            xlim=(extent[0], extent[1]),
            ylim=(extent[2], extent[3]),
        )
        panel_figure.tight_layout()
        panel_figure.savefig(panel_path, dpi=220)
        plt.close(panel_figure)
        standalone_images[f"{slug}_fes"] = str(panel_path)

    marginal_specs = (
        (
            "x",
            reference.x_centers,
            exact_x,
            proposal_x,
            None if weighted_hist is None else np.sum(weighted_hist, axis=1),
        ),
        (
            "y",
            reference.y_centers,
            exact_y,
            proposal_y,
            None if weighted_hist is None else np.sum(weighted_hist, axis=0),
        ),
    )
    for (
        coordinate_name,
        centers,
        exact_marginal,
        proposal_marginal,
        weighted_marginal,
    ) in marginal_specs:
        panel_path = output / f"mb2d_{coordinate_name}_marginal.png"
        panel_figure, panel_axis = plt.subplots(figsize=(5.2, 4.0))
        filled_curve(
            panel_axis,
            centers,
            proposal_marginal,
            role="proposal",
            label="Proposal",
        )
        if weighted_marginal is not None:
            filled_curve(
                panel_axis,
                centers,
                weighted_marginal,
                role="reweighted",
                label="Reweighted",
            )
        panel_axis.plot(
            centers,
            exact_marginal,
            color=EXACT_COLOR,
            linewidth=1.8,
            label="Exact",
        )
        panel_axis.set(
            xlabel=coordinate_name,
            ylabel="Probability / bin",
            title=f"{coordinate_name} marginal",
        )
        panel_axis.legend(frameon=True)
        panel_figure.tight_layout()
        panel_figure.savefig(panel_path, dpi=220)
        plt.close(panel_figure)
        standalone_images[f"{coordinate_name}_marginal"] = str(panel_path)

    energy_path = output / "mb2d_energy_distribution.png"
    energy_figure, energy_axis = plt.subplots(figsize=(5.2, 4.0))
    filled_stairs(
        energy_axis,
        _energy_density_in_view(sample_energy, energy_bins),
        energy_bins,
        role="proposal",
        label="Proposal",
    )
    if weights is not None:
        filled_stairs(
            energy_axis,
            _energy_density_in_view(sample_energy, energy_bins, weights=weights),
            energy_bins,
            role="reweighted",
            label="Reweighted",
        )
    energy_axis.stairs(
        exact_energy_display_density,
        energy_bins,
        color=EXACT_COLOR,
        linewidth=1.5,
        label="Exact",
    )
    energy_axis.text(
        0.98,
        0.98,
        annotation,
        transform=energy_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    energy_axis.set(
        xlabel="Energy",
        ylabel="Density",
        title="Energy distribution",
        xlim=(lower, upper),
    )
    energy_axis.legend(frameon=True)
    energy_figure.tight_layout()
    energy_figure.savefig(energy_path, dpi=220)
    plt.close(energy_figure)
    standalone_images["energy_distribution"] = str(energy_path)

    metrics = {
        "plot_style": plot_style_metadata(),
        "target": {
            "kind": "analytic_muller_brown_2d",
            "beta": reference.beta,
            "kT": float(kT),
            "domain": [list(axis) for axis in reference.domain],
            "bins": [
                int(reference.probability.shape[0]),
                int(reference.probability.shape[1]),
            ],
        },
        "num_samples": int(coordinates.shape[0]),
        "weight_source": weight_source,
        "weight_transform": {
            "formal_no_clip": clip_percentile is None,
            "clip_percentile": (
                None if clip_percentile is None else float(clip_percentile)
            ),
            "removed_upper_percent": (
                None
                if clip_percentile is None
                else (
                    float(100.0 - float(clip_percentile))
                    if clip_mode == "drop"
                    else 0.0
                )
            ),
            "affected_upper_percent": (
                None
                if clip_percentile is None
                else float(100.0 - float(clip_percentile))
            ),
            "clip_mode": None if clip_percentile is None else str(clip_mode),
            "zero_weight_count": (
                None
                if weights is None
                else int(np.count_nonzero(np.asarray(weights) == 0.0))
            ),
        },
        "unweighted": unweighted,
        "weighted": weighted,
        "weight_diagnostics": weight_metrics,
        "energy_plot_view": {
            "display_only": True,
            "formal_no_clip": clip_percentile is None,
            "sample_usage": "all",
            "exact_curve_smoothing": {
                "display_only": True,
                "method": "gaussian_kernel_on_histogram_bins",
                "sigma_bins": EXACT_ENERGY_DISPLAY_SMOOTH_SIGMA_BINS,
                "mass_preserved": True,
                "formal_metrics_unchanged": True,
            },
            "lower_quantile": float(energy_lower_quantile),
            "upper_quantile": float(energy_upper_quantile),
            "padding_fraction": float(energy_view_padding),
            "limits": [float(lower), float(upper)],
            "histogram_bins": int(energy_histogram_bins),
            "exact": exact_energy_view,
            "proposal": proposal_energy_view,
            "reweighted": weighted_energy_view,
        },
    }
    metrics_path = output / "mb2d_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    return {
        "images": {
            "mb2d_plots": str(figure_path),
            **standalone_images,
        },
        "metrics": metrics,
        "metrics_path": str(metrics_path),
    }


def evaluate_pooled_mb2d(
    archives: Mapping[str, str | Path | Mapping[str, Any]],
    *,
    output_dir: str | Path,
    kT: float = 1.0,
    bins: int = 160,
    domain: Sequence[Sequence[float]] = DEFAULT_DOMAIN,
    basin_centers: Mapping[str, Sequence[float]] = DEFAULT_BASIN_CENTERS,
    energy_lower_quantile: float = 0.005,
    energy_upper_quantile: float = 0.995,
    energy_view_padding: float = 0.05,
    energy_histogram_bins: int = 200,
    clip_percentile: float | None = None,
    clip_mode: str = "drop",
) -> dict[str, Any]:
    """Create an equal-seed, all-sample visualization from formal archives.

    Each archive's selected weight view is normalized independently and then
    given equal total mass. This is a stratified across-seed visualization,
    not a claim that checkpoints from different seeds define one proposal
    density. Per-seed no-clip metrics remain the authoritative formal results.
    """

    if len(archives) < 2:
        raise ValueError("pooled MB2D evaluation requires at least two archives")
    labels: list[str] = []
    coordinate_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    energy_blocks: list[np.ndarray] = []
    sample_counts: list[int] = []
    have_energy = True
    for label, source in archives.items():
        data = _load(source)
        if "R" not in data:
            raise KeyError(f"Pooled archive {label!r} does not contain R")
        coordinates = extract_mb2d_coordinates(data["R"])
        weights, _, weight_source = _resolve_mb2d_weights(
            data,
            kT=float(kT),
            clip_percentile=clip_percentile,
            clip_mode=clip_mode,
        )
        if weights is None:
            raise ValueError(
                f"Pooled archive {label!r} has no formal weights "
                f"(source={weight_source})"
            )
        if weights.shape != (coordinates.shape[0],):
            raise ValueError(f"Pooled archive {label!r} weight shape is inconsistent")
        labels.append(str(label))
        coordinate_blocks.append(coordinates)
        weight_blocks.append(normalized_weights(weights))
        sample_counts.append(int(coordinates.shape[0]))
        if "U" in data:
            energy = np.asarray(data["U"]).reshape(-1)
            if energy.shape != (coordinates.shape[0],):
                raise ValueError(f"Pooled archive {label!r} U shape is inconsistent")
            energy_blocks.append(energy)
        else:
            have_energy = False
    if len(set(sample_counts)) != 1:
        raise ValueError(
            "Pooled MB2D visualization requires equal sample counts per seed "
            "so the unweighted view also gives every seed equal mass"
        )
    number_of_archives = len(coordinate_blocks)
    pooled: dict[str, np.ndarray] = {
        "R": np.concatenate(coordinate_blocks, axis=0),
        "weights": np.concatenate(
            [weights / number_of_archives for weights in weight_blocks],
            axis=0,
        ),
    }
    if have_energy:
        pooled["U"] = np.concatenate(energy_blocks, axis=0)

    result = evaluate_mb2d(
        sample=pooled,
        output_dir=output_dir,
        kT=kT,
        bins=bins,
        domain=domain,
        basin_centers=basin_centers,
        energy_lower_quantile=energy_lower_quantile,
        energy_upper_quantile=energy_upper_quantile,
        energy_view_padding=energy_view_padding,
        energy_histogram_bins=energy_histogram_bins,
        weight_view_label=(
            "Equal-seed stratified weights (formal no clip)"
            if clip_percentile is None
            else (
                "Equal-seed stratified diagnostic: "
                f"{clip_mode} top {100.0 - float(clip_percentile):g}% per seed"
            )
        ),
    )
    pooling = {
        "kind": "equal_seed_stratified_visualization",
        "formal_metrics_authority": "per_seed_archives",
        "labels": labels,
        "sample_counts": sample_counts,
        "total_samples": int(sum(sample_counts)),
        "all_samples_used": True,
        "clip_percentile": (
            None if clip_percentile is None else float(clip_percentile)
        ),
        "removed_upper_percent": (
            None
            if clip_percentile is None
            else (float(100.0 - float(clip_percentile)) if clip_mode == "drop" else 0.0)
        ),
        "affected_upper_percent": (
            None if clip_percentile is None else float(100.0 - float(clip_percentile))
        ),
        "clip_mode": None if clip_percentile is None else str(clip_mode),
        "clip_scope": (
            None if clip_percentile is None else "per_seed_before_equal_mass_pooling"
        ),
    }
    result["metrics"]["weight_transform"] = {
        "formal_no_clip": clip_percentile is None,
        "clip_percentile": (
            None if clip_percentile is None else float(clip_percentile)
        ),
        "removed_upper_percent": (
            None
            if clip_percentile is None
            else (float(100.0 - float(clip_percentile)) if clip_mode == "drop" else 0.0)
        ),
        "affected_upper_percent": (
            None if clip_percentile is None else float(100.0 - float(clip_percentile))
        ),
        "clip_mode": None if clip_percentile is None else str(clip_mode),
        "scope": (
            "pooled_formal_weights"
            if clip_percentile is None
            else "per_seed_before_equal_mass_pooling"
        ),
    }
    result["metrics"]["pooling"] = pooling
    metrics_path = Path(result["metrics_path"])
    metrics_path.write_text(
        json.dumps(result["metrics"], indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    manifest_path = Path(output_dir) / "pool_manifest.json"
    manifest_path.write_text(
        json.dumps(pooling, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    result["pool_manifest_path"] = str(manifest_path)
    return result


def _weights_from_raw_logw(
    data: Mapping[str, Any],
    indices: np.ndarray,
    *,
    beta: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if beta is None and "logw_raw" in data:
        raw = np.asarray(data["logw_raw"], dtype=np.float64).reshape(-1)[indices]
    else:
        if "U" not in data:
            raise KeyError("Temperature or prefix reweighting requires U")
        logq_field = "logq_ambient" if "logq_ambient" in data else "logp"
        if logq_field not in data:
            raise KeyError(
                "Temperature or prefix reweighting requires logq_ambient/logp"
            )
        resolved_beta = 1.0 if beta is None else float(beta)
        raw = (
            -resolved_beta
            * np.asarray(data["U"], dtype=np.float64).reshape(-1)[indices]
            - np.asarray(data[logq_field], dtype=np.float64).reshape(-1)[indices]
        )
    valid = np.isfinite(raw)
    for field in ("valid_mask", "support_mask"):
        if field in data:
            valid &= np.asarray(data[field], dtype=bool).reshape(-1)[indices]
    if not valid.any():
        raise ValueError("Selected samples contain no finite formal weights")
    maximum = np.max(raw[valid])
    weights = np.zeros(raw.shape, dtype=np.float64)
    weights[valid] = np.exp(raw[valid] - maximum)
    weights /= np.sum(weights)
    return weights, raw


def sample_size_convergence(
    *,
    sample: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    sizes: Sequence[int] = (1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000),
    repeats: int = 5,
    seed: int = 0,
    kT: float = 1.0,
    bins: int = 100,
    domain: Sequence[Sequence[float]] = DEFAULT_DOMAIN,
) -> dict[str, Any]:
    """Measure no-clip convergence on nested random prefixes."""

    data = _load(sample)
    if "R" not in data:
        raise KeyError("Sample-size convergence requires R")
    coordinates = extract_mb2d_coordinates(data["R"])
    selected_sizes = sorted(
        {int(size) for size in sizes if 0 < int(size) <= coordinates.shape[0]}
    )
    if not selected_sizes:
        raise ValueError("No requested sample size lies inside the archive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    reference = analytic_mb2d_reference(beta=1.0 / float(kT), bins=bins, domain=domain)
    rows: list[dict[str, float | int]] = []
    for repeat in range(int(repeats)):
        permutation = np.random.default_rng(seed + repeat).permutation(
            coordinates.shape[0]
        )
        for size in selected_sizes:
            indices = permutation[:size]
            weights, raw = _weights_from_raw_logw(data, indices)
            distribution = mb2d_distribution_metrics(
                coordinates[indices],
                reference=reference,
                weights=weights,
            )
            diagnostics = importance_weight_diagnostics(weights, logw_raw=raw)
            rows.append(
                {
                    "repeat": repeat,
                    "num_samples": size,
                    "js_2d": float(distribution["js_2d"]),
                    "pmf_rmse": float(distribution["pmf_rmse"]),
                    "basin_l1_error": float(distribution["basin_l1_error"]),
                    "energy_mean_error": float(distribution["energy_mean_error"]),
                    "ess": float(diagnostics["ess"]),
                    "ess_fraction": float(diagnostics["ess_fraction"]),
                    "max_weight": float(diagnostics["max_weight"]),
                    "top_0p1_percent_mass": float(diagnostics["top_0p1_percent_mass"]),
                    "top_1_percent_mass": float(diagnostics["top_1_percent_mass"]),
                    "logw_variance": float(diagnostics["logw_variance"]),
                }
            )
    numeric_fields = tuple(
        field for field in rows[0] if field not in {"repeat", "num_samples"}
    )
    summary = []
    for size in selected_sizes:
        current = [row for row in rows if row["num_samples"] == size]
        entry: dict[str, Any] = {"num_samples": size, "repeats": len(current)}
        for field in numeric_fields:
            values = np.asarray([row[field] for row in current], dtype=np.float64)
            entry[field] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        summary.append(entry)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "sample_size_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output / "sample_size_convergence.json"
    result = {
        "sample": str(sample) if not isinstance(sample, Mapping) else "<mapping>",
        "seed": int(seed),
        "formal_no_clip": True,
        "rows": rows,
        "summary": summary,
    }
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    _plot_convergence_summary(summary, output / "sample_size_convergence.png")
    return result


def _plot_convergence_summary(
    summary: Sequence[Mapping[str, Any]], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sizes = np.asarray([row["num_samples"] for row in summary])
    panels = (
        ("ess_fraction", "ESS / N"),
        ("js_2d", "2D JS"),
        ("basin_l1_error", "Basin L1 error"),
        ("max_weight", "Max normalized weight"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), constrained_layout=True)
    for axis, (field, label) in zip(axes.reshape(-1), panels, strict=True):
        mean = np.asarray([row[field]["mean"] for row in summary])
        std = np.asarray([row[field]["std"] for row in summary])
        axis.plot(sizes, mean, marker="o")
        axis.fill_between(sizes, mean - std, mean + std, alpha=0.2)
        axis.set_xscale("log")
        axis.set(xlabel="Number of samples", ylabel=label)
        axis.grid(alpha=0.25)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def temperature_reweighting_sweep(
    *,
    sample: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    betas: Sequence[float],
    bins: int = 100,
    domain: Sequence[Sequence[float]] = DEFAULT_DOMAIN,
) -> dict[str, Any]:
    """Reuse one PF proposal at several analytic target temperatures."""

    data = _load(sample)
    coordinates = extract_mb2d_coordinates(data["R"])
    indices = np.arange(coordinates.shape[0])
    rows = []
    for beta_value in betas:
        beta = float(beta_value)
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("All beta values must be finite and positive")
        weights, raw = _weights_from_raw_logw(data, indices, beta=beta)
        reference = analytic_mb2d_reference(beta=beta, bins=bins, domain=domain)
        distribution = mb2d_distribution_metrics(
            coordinates,
            reference=reference,
            weights=weights,
        )
        diagnostics = importance_weight_diagnostics(weights, logw_raw=raw)
        rows.append(
            {
                "beta": beta,
                "temperature_ratio_to_beta1": 1.0 / beta,
                "js_2d": distribution["js_2d"],
                "pmf_rmse": distribution["pmf_rmse"],
                "basin_l1_error": distribution["basin_l1_error"],
                "energy_mean_error": distribution["energy_mean_error"],
                "ess": diagnostics["ess"],
                "ess_fraction": diagnostics["ess_fraction"],
                "max_weight": diagnostics["max_weight"],
                "logw_variance": diagnostics["logw_variance"],
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {"formal_no_clip": True, "rows": rows}
    (output / "temperature_reweighting.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    _plot_temperature_sweep(rows, output / "temperature_reweighting.png")
    return result


def _plot_temperature_sweep(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    beta = np.asarray([row["beta"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    axes[0].plot(beta, [row["ess_fraction"] for row in rows], marker="o")
    axes[0].set(xlabel=r"$\beta$", ylabel="ESS / N")
    axes[1].plot(beta, [row["js_2d"] for row in rows], marker="o", label="2D JS")
    axes[1].plot(
        beta,
        [row["basin_l1_error"] for row in rows],
        marker="s",
        label="Basin L1",
    )
    axes[1].set(xlabel=r"$\beta$", ylabel="Error")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def compare_likelihood_archives(
    archives: Mapping[str, str | Path | Mapping[str, Any]],
    *,
    reference_label: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare ODE-tolerance runs that used identical initial samples."""

    if reference_label not in archives:
        raise KeyError(f"reference_label {reference_label!r} is absent")
    loaded = {label: _load(source) for label, source in archives.items()}
    reference = loaded[reference_label]
    if "R" not in reference:
        raise KeyError("Likelihood archives require R")
    reference_r = extract_mb2d_coordinates(reference["R"])
    reference_logq = np.asarray(
        reference.get("logq_ambient", reference.get("logp")),
        dtype=np.float64,
    ).reshape(-1)
    rows = {}
    for label, data in loaded.items():
        coordinates = extract_mb2d_coordinates(data["R"])
        logq = np.asarray(
            data.get("logq_ambient", data.get("logp")),
            dtype=np.float64,
        ).reshape(-1)
        if coordinates.shape != reference_r.shape or logq.shape != reference_logq.shape:
            raise ValueError("Likelihood sweep archives must have matching shapes")
        same_initial = None
        if "initial_X" in reference and "initial_X" in data:
            same_initial = bool(
                np.array_equal(
                    np.asarray(reference["initial_X"]),
                    np.asarray(data["initial_X"]),
                )
            )
        coordinate_error = coordinates - reference_r
        logq_error = logq - reference_logq
        entry = {
            "initial_samples_bitwise_identical": same_initial,
            "coordinate_rmse": float(np.sqrt(np.mean(coordinate_error**2))),
            "coordinate_max_abs": float(np.max(np.abs(coordinate_error))),
            "logq_rmse": float(np.sqrt(np.mean(logq_error**2))),
            "logq_max_abs": float(np.max(np.abs(logq_error))),
        }
        if "weights" in reference and "weights" in data:
            reference_weights = normalized_weights(reference["weights"])
            weights = normalized_weights(data["weights"])
            entry["weight_total_variation"] = float(
                0.5 * np.sum(np.abs(weights - reference_weights))
            )
        rows[label] = entry
    result = {"reference": reference_label, "runs": rows}
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=True),
            encoding="utf-8",
        )
    return result


def compare_sde_pf_archives(
    *,
    sde_sample: str | Path | Mapping[str, Any],
    pf_sample: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    bins: int = 60,
    domain: Sequence[Sequence[float]] = DEFAULT_DOMAIN,
) -> dict[str, Any]:
    """Check whether stochastic and probability-flow terminal laws align."""

    from .pf_validation import terminal_distribution_alignment

    sde = _load(sde_sample)
    pf = _load(pf_sample)
    sde_coordinates = extract_mb2d_coordinates(sde["R"])
    pf_coordinates = extract_mb2d_coordinates(pf["R"])
    energy_available = "U" in sde and "U" in pf
    alignment = terminal_distribution_alignment(
        sde_coordinates,
        pf_coordinates,
        bins=bins,
        ranges=tuple((float(axis[0]), float(axis[1])) for axis in domain),
        sde_energy=sde.get("U") if energy_available else None,
        pf_energy=pf.get("U") if energy_available else None,
        energy_bins=bins,
    )
    result = {
        "sde_sample": (
            str(sde_sample) if not isinstance(sde_sample, Mapping) else "<mapping>"
        ),
        "pf_sample": (
            str(pf_sample) if not isinstance(pf_sample, Mapping) else "<mapping>"
        ),
        "alignment": _array_tree_to_lists(alignment),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sde_pf_alignment.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    _plot_sde_pf_alignment(
        sde_coordinates,
        pf_coordinates,
        output / "sde_pf_alignment.png",
        domain=domain,
        bins=bins,
    )
    return result


def _array_tree_to_lists(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _array_tree_to_lists(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_array_tree_to_lists(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _plot_sde_pf_alignment(
    sde: np.ndarray,
    pf: np.ndarray,
    output: Path,
    *,
    domain: Sequence[Sequence[float]],
    bins: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ranges = tuple((float(axis[0]), float(axis[1])) for axis in domain)
    histograms = [
        np.histogram2d(values[:, 0], values[:, 1], bins=bins, range=ranges)[0]
        for values in (sde, pf)
    ]
    maximum = max(float(np.max(histogram)) for histogram in histograms)
    extent = (ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)
    for axis, title, histogram in zip(
        axes,
        ("Forward SDE", "Probability flow"),
        histograms,
        strict=True,
    ):
        image = axis.imshow(
            histogram.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=0.0,
            vmax=maximum,
            cmap="magma",
        )
        axis.set(title=title, xlabel="x", ylabel="y")
    fig.colorbar(image, ax=axes.tolist(), label="Count / bin")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def compare_mb2d_archives(
    archives: Mapping[str, str | Path | Mapping[str, Any]],
    *,
    output_dir: str | Path,
    kT: float = 1.0,
    bins: int = 100,
    domain: Sequence[Sequence[float]] = DEFAULT_DOMAIN,
) -> dict[str, Any]:
    """Evaluate labelled arms/checkpoints/budgets under one analytic contract."""

    if len(archives) < 2:
        raise ValueError("Archive comparison requires at least two labelled runs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    evaluations: dict[str, Any] = {}
    for label, archive in archives.items():
        if not label:
            raise ValueError("Archive labels must be non-empty")
        result = evaluate_mb2d(
            sample=archive,
            output_dir=output / label,
            kT=kT,
            bins=bins,
            domain=domain,
        )
        metrics = result["metrics"]
        weighted = metrics["weighted"]
        diagnostics = metrics["weight_diagnostics"]
        row = {
            "label": label,
            "num_samples": metrics["num_samples"],
            "proposal_js_2d": metrics["unweighted"]["js_2d"],
            "proposal_pmf_rmse": metrics["unweighted"]["pmf_rmse"],
            "proposal_basin_l1_error": metrics["unweighted"]["basin_l1_error"],
            "weighted_js_2d": None if weighted is None else weighted["js_2d"],
            "weighted_pmf_rmse": None if weighted is None else weighted["pmf_rmse"],
            "weighted_basin_l1_error": (
                None if weighted is None else weighted["basin_l1_error"]
            ),
            "ess_fraction": None
            if diagnostics is None
            else diagnostics["ess_fraction"],
            "max_weight": None if diagnostics is None else diagnostics["max_weight"],
            "top_1_percent_mass": (
                None if diagnostics is None else diagnostics["top_1_percent_mass"]
            ),
        }
        rows.append(row)
        evaluations[label] = result
    summary = {
        "formal_no_clip": True,
        "rows": rows,
        "evaluations": evaluations,
    }
    (output / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_archive_comparison(rows, output / "comparison.png")
    return summary


def _plot_archive_comparison(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [str(row["label"]) for row in rows]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    for offset, (proposal_field, weighted_field, title) in enumerate(
        (
            ("proposal_js_2d", "weighted_js_2d", "2D JS"),
            ("proposal_basin_l1_error", "weighted_basin_l1_error", "Basin L1"),
        )
    ):
        proposal = np.asarray([row[proposal_field] for row in rows], dtype=np.float64)
        weighted = np.asarray(
            [
                np.nan if row[weighted_field] is None else row[weighted_field]
                for row in rows
            ],
            dtype=np.float64,
        )
        axes[offset].bar(positions - 0.18, proposal, width=0.36, label="Proposal")
        axes[offset].bar(positions + 0.18, weighted, width=0.36, label="Reweighted")
        axes[offset].set(title=title, xticks=positions, xticklabels=labels)
        axes[offset].tick_params(axis="x", rotation=25)
        axes[offset].legend(fontsize=8)
    ess = np.asarray(
        [np.nan if row["ess_fraction"] is None else row["ess_fraction"] for row in rows]
    )
    axes[2].bar(positions, ess)
    axes[2].set(title="Formal ESS / N", xticks=positions, xticklabels=labels)
    axes[2].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=200)
    plt.close(fig)
