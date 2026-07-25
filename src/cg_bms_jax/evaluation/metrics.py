"""NumPy metrics shared by the Ala2 and Muller--Brown evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def normalized_weights(weights: Any) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected one weight per sample, got {values.shape}")
    values = np.where(np.isfinite(values) & (values >= 0), values, 0.0)
    total = values.sum()
    if not total > 0:
        raise ValueError("Weights contain no positive finite mass")
    return values / total


def log_weights_to_weights(
    logw: Any,
    *,
    clip_percentile: float | None = None,
    clip_mode: str = "drop",
) -> np.ndarray:
    values = np.asarray(logw, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected one log weight per sample, got {values.shape}")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Log weights contain no finite values")
    adjusted = values.copy()
    if clip_percentile is not None:
        if not 0 <= clip_percentile <= 100:
            raise ValueError("clip_percentile must lie in [0,100]")
        threshold = np.percentile(adjusted[finite], clip_percentile)
        if clip_mode == "drop":
            finite &= adjusted < threshold
        elif clip_mode == "cap":
            adjusted[finite] = np.minimum(adjusted[finite], threshold)
        else:
            raise ValueError("clip_mode must be 'drop' or 'cap'")
    if not finite.any():
        raise ValueError("Clipping removed all samples")
    shifted = np.full_like(adjusted, -np.inf)
    maximum = np.max(adjusted[finite])
    shifted[finite] = adjusted[finite] - maximum
    weights = np.zeros_like(adjusted)
    weights[finite] = np.exp(shifted[finite])
    return normalized_weights(weights)


def resolve_weights(
    data: Mapping[str, Any],
    *,
    kT: float,
    clip_percentile: float | None = None,
    clip_mode: str = "drop",
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Resolve normalized weights, raw log weights, and their source.

    Proposal-only archives are valid and return ``(None, None, 'proposal_only')``.
    """

    if "weights" in data:
        weights = normalized_weights(data["weights"])
        logw = np.log(np.where(weights > 0, weights, np.nan))
        return weights, logw, "weights"
    if "logw" in data:
        logw = np.asarray(data["logw"], dtype=np.float64).reshape(-1)
        return (
            log_weights_to_weights(logw, clip_percentile=clip_percentile, clip_mode=clip_mode),
            logw,
            "logw",
        )
    if "logp" in data and "U" in data:
        logw = -np.asarray(data["U"], dtype=np.float64).reshape(-1) / float(kT) - np.asarray(
            data["logp"], dtype=np.float64
        ).reshape(-1)
        return (
            log_weights_to_weights(logw, clip_percentile=clip_percentile, clip_mode=clip_mode),
            logw,
            "computed_from_U_logp",
        )
    return None, None, "proposal_only"


def effective_sample_size_fraction(weights: Any | None) -> float:
    if weights is None:
        return 1.0
    values = normalized_weights(weights)
    return float(1.0 / np.sum(values * values) / values.size)


def importance_weight_diagnostics(
    weights: Any,
    *,
    logw_raw: Any | None = None,
    top_fractions: Sequence[float] = (0.001, 0.01, 0.1),
) -> dict[str, float]:
    """Summarize formal, unmodified importance weights.

    The reported top-mass values are concentration diagnostics, not clipped
    estimates.  Keeping this helper independent of plotting makes the same
    no-clip contract available to MB, Ala2, and future targets.
    """

    values = normalized_weights(weights)
    size = int(values.size)
    ess = float(1.0 / np.sum(values * values))
    positive = values > 0.0
    entropy = float(-np.sum(values[positive] * np.log(values[positive])))
    result = {
        "num_samples": float(size),
        "ess": ess,
        "ess_fraction": ess / size,
        "max_weight": float(np.max(values)),
        "entropy": entropy,
        "perplexity": float(np.exp(entropy)),
    }
    descending = np.sort(values)[::-1]
    for fraction in top_fractions:
        value = float(fraction)
        if not 0.0 < value <= 1.0:
            raise ValueError("top_fractions must lie in (0,1]")
        count = max(1, int(np.ceil(value * size)))
        label = f"top_{100.0 * value:g}_percent_mass".replace(".", "p")
        result[label] = float(np.sum(descending[:count]))
    if logw_raw is not None:
        raw = np.asarray(logw_raw, dtype=np.float64).reshape(-1)
        if raw.shape != values.shape:
            raise ValueError("logw_raw must have the same shape as weights")
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            result.update(
                {
                    "finite_logw_fraction": 0.0,
                    "logw_variance": float("nan"),
                    "logw_span": float("nan"),
                }
            )
        else:
            result.update(
                {
                    "finite_logw_fraction": float(finite.size / raw.size),
                    "logw_variance": float(np.var(finite)),
                    "logw_span": float(np.max(finite) - np.min(finite)),
                }
            )
    return result


def _histogram(
    samples: np.ndarray,
    *,
    bins: int | Sequence[int],
    ranges: Sequence[tuple[float, float]],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        hist, _ = np.histogram(samples, bins=bins, range=ranges[0], weights=weights)
    elif samples.ndim == 2:
        hist, _ = np.histogramdd(samples, bins=bins, range=ranges, weights=weights)
    else:
        raise ValueError(f"Histogram samples must be 1D or 2D, got {samples.shape}")
    return np.asarray(hist, dtype=np.float64)


def histogram_js_divergence(
    target: Any,
    sample: Any,
    *,
    bins: int = 100,
    ranges: Sequence[tuple[float, float]],
    sample_weights: Any | None = None,
    baseline: float = 1e-10,
) -> float:
    target = np.asarray(target)
    sample = np.asarray(sample)
    weights = None if sample_weights is None else normalized_weights(sample_weights)
    p = _histogram(target, bins=bins, ranges=ranges).reshape(-1) + baseline
    q = _histogram(sample, bins=bins, ranges=ranges, weights=weights).reshape(-1) + baseline
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def histogram_pmf_error(
    target: Any,
    sample: Any,
    *,
    bins: int = 100,
    ranges: Sequence[tuple[float, float]],
    sample_weights: Any | None = None,
    baseline: float = 1e-10,
) -> float:
    target = np.asarray(target)
    sample = np.asarray(sample)
    weights = None if sample_weights is None else normalized_weights(sample_weights)
    p = _histogram(target, bins=bins, ranges=ranges).reshape(-1)
    q = _histogram(sample, bins=bins, ranges=ranges, weights=weights).reshape(-1)
    p = p / (p.sum() + baseline)
    q = q / (q.sum() + baseline)
    p_safe = np.where(p > 0, p, baseline)
    q_safe = np.where(q > 0, q, baseline)
    mixture = 0.5 * (p_safe + q_safe)
    mixture /= mixture.sum()
    return float(np.sum(mixture * (np.log(p_safe) - np.log(q_safe)) ** 2))


def energy_wasserstein(predicted: Any, reference: Any, *, n_quantiles: int = 1000) -> tuple[float, float]:
    predicted = np.sort(np.asarray(predicted, dtype=np.float64).reshape(-1))
    reference = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    predicted = predicted[np.isfinite(predicted)]
    reference = reference[np.isfinite(reference)]
    if predicted.size == 0 or reference.size == 0:
        return float("nan"), float("nan")
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    pred_aligned = np.interp(quantiles, np.linspace(0.0, 1.0, predicted.size), predicted)
    ref_aligned = np.interp(quantiles, np.linspace(0.0, 1.0, reference.size), reference)
    difference = pred_aligned - ref_aligned
    return float(np.mean(np.abs(difference))), float(np.sqrt(np.mean(difference**2)))


def distribution_metrics(
    target: Any,
    sample: Any,
    *,
    ranges: Sequence[tuple[float, float]],
    sample_weights: Any | None = None,
    bins: int = 100,
) -> dict[str, float]:
    return {
        "JS_Divergence": histogram_js_divergence(
            target, sample, bins=bins, ranges=ranges, sample_weights=sample_weights
        ),
        "PMF_Error": histogram_pmf_error(
            target, sample, bins=bins, ranges=ranges, sample_weights=sample_weights
        ),
        "ESS_Percent": effective_sample_size_fraction(sample_weights),
    }


def bootstrap_distribution_metrics(
    target: Any,
    sample: Any,
    *,
    ranges: Sequence[tuple[float, float]],
    sample_weights: Any | None = None,
    bins: int = 100,
    n_bootstraps: int = 500,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Bootstrap target and proposal clouds; weighted samples use resampling."""

    target = np.asarray(target)
    sample = np.asarray(sample)
    if target.ndim != sample.ndim:
        raise ValueError("Target and sample ranks differ")
    weights = None if sample_weights is None else normalized_weights(sample_weights)
    rng = np.random.default_rng(seed)
    output = {name: np.empty(n_bootstraps, dtype=np.float64) for name in ("JS_Divergence", "PMF_Error", "ESS_Percent")}
    for index in range(n_bootstraps):
        target_boot = target[rng.choice(target.shape[0], target.shape[0], replace=True)]
        sample_indices = rng.choice(sample.shape[0], sample.shape[0], replace=True, p=weights)
        sample_boot = sample[sample_indices]
        values = distribution_metrics(target_boot, sample_boot, ranges=ranges, bins=bins)
        values["ESS_Percent"] = effective_sample_size_fraction(weights)
        for name in output:
            output[name][index] = values[name]
    return output


def summarize_bootstrap(results: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for name, values in results.items()
    }
