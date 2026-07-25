"""Numerical validation helpers for probability-flow likelihoods.

The direct Jacobian calculation is intentionally an eager, small-batch audit.
It materializes one dense event-space Jacobian per sample and performs host-side
validity checks, so it is not intended for production sampling or use inside
``jax.jit``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
FlowMap = Callable[[Array], Array]


class DirectFlowMapLogqResult(NamedTuple):
    """Terminal states and density changes obtained from dense Jacobians."""

    terminal_states: Array
    logq: Array
    initial_states: Array
    initial_logq: Array
    jacobians: Array
    jacobian_signs: Array
    log_abs_det_jacobians: Array


def direct_flow_map_logq(
    flow_map: FlowMap,
    initial_states: Any,
    initial_logq: Any,
) -> DirectFlowMapLogqResult:
    r"""Evaluate a batched flow map and apply the change-of-variables formula.

    ``flow_map`` receives one event-shaped state and must return the same shape.
    For each sample, this function computes its dense Jacobian with
    :func:`jax.jacrev` and then applies

    .. math::

        \log q_1 = \log q_0 - \log |\det J|.

    Both orientation-preserving and orientation-reversing maps are valid because
    a density uses the absolute determinant. Singular or non-finite Jacobians
    are rejected.
    """

    states = jnp.asarray(initial_states)
    if states.ndim < 2:
        raise ValueError("initial_states must contain batch and event dimensions")
    if states.shape[0] == 0:
        raise ValueError("initial_states must contain at least one sample")
    if not jnp.issubdtype(states.dtype, jnp.floating):
        raise TypeError("initial_states must have a floating dtype")
    if not np.isfinite(np.asarray(states)).all():
        raise ValueError("initial_states contain non-finite values")

    logq0 = jnp.asarray(initial_logq, dtype=states.dtype)
    if logq0.shape != (states.shape[0],):
        raise ValueError("initial_logq must have shape (batch,)")
    if not np.isfinite(np.asarray(logq0)).all():
        raise ValueError("initial_logq contains non-finite values")

    event_shape = states.shape[1:]
    event_size = int(np.prod(event_shape))
    flat_states = states.reshape((states.shape[0], event_size))

    def single_flow(flat_state: Array) -> Array:
        state = flat_state.reshape(event_shape)
        terminal = jnp.asarray(flow_map(state), dtype=state.dtype)
        if terminal.shape != event_shape:
            raise ValueError(
                "flow_map must return the same event shape as its input; "
                f"expected {event_shape}, got {terminal.shape}"
            )
        return terminal.reshape((event_size,))

    terminal_flat = jax.vmap(single_flow)(flat_states)
    jacobians = jax.vmap(jax.jacrev(single_flow))(flat_states)
    signs, log_abs_det = jnp.linalg.slogdet(jacobians)

    terminal_host = np.asarray(terminal_flat)
    jacobian_host = np.asarray(jacobians)
    signs_host = np.asarray(signs)
    log_abs_det_host = np.asarray(log_abs_det)
    if not np.isfinite(terminal_host).all():
        raise ValueError("flow_map produced non-finite terminal states")
    if not np.isfinite(jacobian_host).all():
        raise ValueError("flow_map Jacobian contains non-finite entries")
    valid_determinant = (
        np.isfinite(signs_host)
        & (signs_host != 0)
        & np.isfinite(log_abs_det_host)
    )
    if not valid_determinant.all():
        invalid = np.flatnonzero(~valid_determinant).tolist()
        raise ValueError(
            "flow_map Jacobian is singular or has a non-finite determinant "
            f"for batch indices {invalid}"
        )

    logq1 = logq0 - log_abs_det.astype(logq0.dtype)
    if not np.isfinite(np.asarray(logq1)).all():
        raise ValueError("direct flow-map logq contains non-finite values")
    return DirectFlowMapLogqResult(
        terminal_states=terminal_flat.reshape(states.shape),
        logq=logq1,
        initial_states=states,
        initial_logq=logq0,
        jacobians=jacobians,
        jacobian_signs=signs,
        log_abs_det_jacobians=log_abs_det,
    )


def compare_integrated_logq(
    integrated_logq: Any,
    direct_logq: Any,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> dict[str, float | int | bool]:
    """Summarize errors between integrated and direct-Jacobian PF densities.

    Errors use the signed convention ``integrated_logq - direct_logq``.
    Non-finite entries are retained in the sample counts and excluded only from
    finite-pair error statistics.
    """

    integrated = np.asarray(integrated_logq, dtype=np.float64)
    direct = np.asarray(direct_logq, dtype=np.float64)
    if integrated.ndim != 1 or direct.ndim != 1:
        raise ValueError("integrated_logq and direct_logq must both be one-dimensional")
    if integrated.shape != direct.shape:
        raise ValueError(
            "integrated_logq and direct_logq must have the same shape; "
            f"got {integrated.shape} and {direct.shape}"
        )
    if integrated.size == 0:
        raise ValueError("logq arrays must not be empty")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and non-negative")

    integrated_finite = np.isfinite(integrated)
    direct_finite = np.isfinite(direct)
    finite = integrated_finite & direct_finite
    finite_count = int(finite.sum())
    sample_count = int(integrated.size)

    if finite_count:
        errors = integrated[finite] - direct[finite]
        absolute_errors = np.abs(errors)
        tolerance = atol + rtol * np.abs(direct[finite])
        within_tolerance = absolute_errors <= tolerance
        mean_error = float(np.mean(errors))
        mean_abs_error = float(np.mean(absolute_errors))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        max_abs_error = float(np.max(absolute_errors))
        median_abs_error = float(np.median(absolute_errors))
        p95_abs_error = float(np.quantile(absolute_errors, 0.95))
        within_count = int(within_tolerance.sum())
    else:
        mean_error = float("nan")
        mean_abs_error = float("nan")
        rmse = float("nan")
        max_abs_error = float("nan")
        median_abs_error = float("nan")
        p95_abs_error = float("nan")
        within_count = 0

    return {
        "num_samples": sample_count,
        "integrated_finite_count": int(integrated_finite.sum()),
        "direct_finite_count": int(direct_finite.sum()),
        "finite_pair_count": finite_count,
        "finite_pair_fraction": finite_count / sample_count,
        "all_finite": finite_count == sample_count,
        "mean_error": mean_error,
        "mean_abs_error": mean_abs_error,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "median_abs_error": median_abs_error,
        "p95_abs_error": p95_abs_error,
        "within_tolerance_count": within_count,
        "within_tolerance_fraction": within_count / sample_count,
        "within_tolerance_fraction_finite": (
            within_count / finite_count if finite_count else float("nan")
        ),
        "all_within_tolerance": (
            finite_count == sample_count and within_count == sample_count
        ),
    }


def _sample_matrix(samples: Any, name: str) -> tuple[np.ndarray, int, int]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError(f"{name} must contain a sample dimension")
    if values.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if values.ndim == 1:
        matrix = values[:, None]
    else:
        matrix = values.reshape((values.shape[0], -1))
    finite = np.all(np.isfinite(matrix), axis=1)
    finite_count = int(finite.sum())
    if finite_count < 2:
        raise ValueError(f"{name} must contain at least two finite samples")
    return matrix[finite], int(matrix.shape[0]), finite_count


def _covariance(samples: np.ndarray) -> np.ndarray:
    dimension = samples.shape[1]
    return np.asarray(np.cov(samples, rowvar=False, ddof=1), dtype=np.float64).reshape(
        (dimension, dimension)
    )


def _validate_bins(
    bins: int | Sequence[int],
    dimension: int,
) -> int | tuple[int, ...]:
    if isinstance(bins, Integral):
        if int(bins) <= 0:
            raise ValueError("bins must be positive")
        return int(bins)
    values = tuple(int(value) for value in bins)
    if len(values) != dimension or any(value <= 0 for value in values):
        raise ValueError("bins must provide one positive count per histogram dimension")
    return values


def _inferred_ranges(first: np.ndarray, second: np.ndarray) -> tuple[tuple[float, float], ...]:
    combined = np.concatenate((first, second), axis=0)
    lower = np.min(combined, axis=0)
    upper = np.max(combined, axis=0)
    span = upper - lower
    padding = np.where(
        span > 0,
        np.maximum(0.01 * span, 1.0e-12),
        np.maximum(1.0e-6 * np.maximum(np.abs(lower), 1.0), 1.0e-12),
    )
    return tuple(
        (float(low - pad), float(high + pad))
        for low, high, pad in zip(lower, upper, padding, strict=True)
    )


def _validated_ranges(
    ranges: Sequence[tuple[float, float]] | None,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    dimension = first.shape[1]
    if ranges is None:
        return _inferred_ranges(first, second)
    values = tuple((float(bounds[0]), float(bounds[1])) for bounds in ranges)
    if len(values) != dimension:
        raise ValueError("ranges must provide one (lower, upper) pair per dimension")
    if any(
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or not lower < upper
        for lower, upper in values
    ):
        raise ValueError("each histogram range must be finite and strictly increasing")
    return values


def _histogram_js(
    first: np.ndarray,
    second: np.ndarray,
    *,
    bins: int | Sequence[int],
    ranges: Sequence[tuple[float, float]],
    baseline: float,
) -> float:
    first_histogram, _ = np.histogramdd(first, bins=bins, range=ranges)
    second_histogram, _ = np.histogramdd(second, bins=bins, range=ranges)
    if first_histogram.sum() <= 0 or second_histogram.sum() <= 0:
        raise ValueError("histogram ranges contain no samples from one distribution")
    first_probability = first_histogram.reshape(-1) + baseline
    second_probability = second_histogram.reshape(-1) + baseline
    first_probability /= first_probability.sum()
    second_probability /= second_probability.sum()
    mixture = 0.5 * (first_probability + second_probability)
    divergence = 0.5 * (
        np.sum(first_probability * np.log(first_probability / mixture))
        + np.sum(second_probability * np.log(second_probability / mixture))
    )
    return float(divergence)


def _energy_metrics(
    sde_energy: Any,
    pf_energy: Any,
    *,
    sde_sample_count: int,
    pf_sample_count: int,
    bins: int,
    energy_range: tuple[float, float] | None,
    baseline: float,
) -> dict[str, Any]:
    sde_values = np.asarray(sde_energy, dtype=np.float64)
    pf_values = np.asarray(pf_energy, dtype=np.float64)
    if sde_values.size != sde_sample_count or pf_values.size != pf_sample_count:
        raise ValueError("each energy array must contain one value per corresponding sample")
    sde_values = sde_values.reshape(-1)
    pf_values = pf_values.reshape(-1)
    sde_finite = np.isfinite(sde_values)
    pf_finite = np.isfinite(pf_values)
    if int(sde_finite.sum()) < 2 or int(pf_finite.sum()) < 2:
        raise ValueError("energy arrays must each contain at least two finite values")
    sde_valid = sde_values[sde_finite, None]
    pf_valid = pf_values[pf_finite, None]
    ranges = _validated_ranges(
        None if energy_range is None else (energy_range,),
        sde_valid,
        pf_valid,
    )
    validated_bins = _validate_bins(bins, 1)
    sde_mean = float(np.mean(sde_valid))
    pf_mean = float(np.mean(pf_valid))
    sde_variance = float(np.var(sde_valid, ddof=1))
    pf_variance = float(np.var(pf_valid, ddof=1))
    return {
        "sde_num_values": sde_sample_count,
        "pf_num_values": pf_sample_count,
        "sde_finite_count": int(sde_finite.sum()),
        "pf_finite_count": int(pf_finite.sum()),
        "sde_finite_fraction": float(np.mean(sde_finite)),
        "pf_finite_fraction": float(np.mean(pf_finite)),
        "sde_mean": sde_mean,
        "pf_mean": pf_mean,
        "mean_difference": pf_mean - sde_mean,
        "sde_variance": sde_variance,
        "pf_variance": pf_variance,
        "variance_difference": pf_variance - sde_variance,
        "histogram_js": _histogram_js(
            sde_valid,
            pf_valid,
            bins=validated_bins,
            ranges=ranges,
            baseline=baseline,
        ),
        "histogram_range": ranges[0],
    }


def terminal_distribution_alignment(
    sde_samples: Any,
    pf_samples: Any,
    *,
    bins: int | Sequence[int] = 50,
    ranges: Sequence[tuple[float, float]] | None = None,
    sde_energy: Any | None = None,
    pf_energy: Any | None = None,
    energy_bins: int = 50,
    energy_range: tuple[float, float] | None = None,
    baseline: float = 1.0e-12,
) -> dict[str, Any]:
    """Compare terminal SDE and probability-flow sample distributions.

    Event dimensions are flattened for means and covariance matrices. A common
    histogram JS divergence is additionally computed for one- and
    two-dimensional events; for higher dimensions it is reported as ``None``.
    Non-finite sample rows are counted and excluded from moment/histogram
    calculations. Optional energy arrays are compared independently as scalar
    distributions.
    """

    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("baseline must be finite and positive")
    sde, sde_count, sde_finite_count = _sample_matrix(sde_samples, "sde_samples")
    pf, pf_count, pf_finite_count = _sample_matrix(pf_samples, "pf_samples")
    if sde.shape[1] != pf.shape[1]:
        raise ValueError(
            "SDE and PF event dimensions must match; "
            f"got {sde.shape[1]} and {pf.shape[1]}"
        )

    dimension = int(sde.shape[1])
    sde_mean = np.mean(sde, axis=0)
    pf_mean = np.mean(pf, axis=0)
    sde_covariance = _covariance(sde)
    pf_covariance = _covariance(pf)
    mean_difference = pf_mean - sde_mean
    covariance_difference = pf_covariance - sde_covariance

    histogram_js: float | None
    histogram_ranges: tuple[tuple[float, float], ...] | None
    if dimension <= 2:
        histogram_ranges = _validated_ranges(ranges, sde, pf)
        histogram_js = _histogram_js(
            sde,
            pf,
            bins=_validate_bins(bins, dimension),
            ranges=histogram_ranges,
            baseline=baseline,
        )
    else:
        if ranges is not None:
            raise ValueError("histogram ranges are supported only for 1D or 2D events")
        histogram_ranges = None
        histogram_js = None

    if (sde_energy is None) != (pf_energy is None):
        raise ValueError("sde_energy and pf_energy must be provided together")
    energy = (
        None
        if sde_energy is None
        else _energy_metrics(
            sde_energy,
            pf_energy,
            sde_sample_count=sde_count,
            pf_sample_count=pf_count,
            bins=energy_bins,
            energy_range=energy_range,
            baseline=baseline,
        )
    )

    return {
        "dimension": dimension,
        "sde": {
            "num_samples": sde_count,
            "finite_sample_count": sde_finite_count,
            "finite_fraction": sde_finite_count / sde_count,
            "mean": sde_mean,
            "covariance": sde_covariance,
        },
        "pf": {
            "num_samples": pf_count,
            "finite_sample_count": pf_finite_count,
            "finite_fraction": pf_finite_count / pf_count,
            "mean": pf_mean,
            "covariance": pf_covariance,
        },
        "mean_difference": mean_difference,
        "mean_l2_error": float(np.linalg.norm(mean_difference)),
        "covariance_difference": covariance_difference,
        "covariance_frobenius_error": float(np.linalg.norm(covariance_difference)),
        "histogram_js": histogram_js,
        "histogram_ranges": histogram_ranges,
        "energy": energy,
    }
