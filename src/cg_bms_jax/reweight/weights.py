"""Numerically stable exact-ambient and CG-BG-compatible weights."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from cg_bms_jax.coordinates import cgbg_correct_log_density

Array = jax.Array


@dataclass(frozen=True)
class WeightResult:
    logw_raw: Array
    logw: Array
    weights: Array
    proposal_log_density: Array
    valid_mask: Array
    log_normalizer: Array
    density_mode: str


def _normalize_log_weights(logw_raw: Array, valid_mask: Array) -> tuple[Array, Array, Array]:
    logw_raw = jnp.asarray(logw_raw)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)
    if logw_raw.shape != valid_mask.shape:
        raise ValueError(f"logw shape {logw_raw.shape} and valid mask {valid_mask.shape} differ")
    safe = jnp.where(valid_mask & jnp.isfinite(logw_raw), logw_raw, -jnp.inf)
    log_norm = jax.scipy.special.logsumexp(safe)
    has_support = jnp.isfinite(log_norm)
    logw = jnp.where(has_support, safe - log_norm, -jnp.inf)
    weights = jnp.where(jnp.isfinite(logw), jnp.exp(logw), 0.0)
    # In float32, subtracting a large log normalizer from similarly large raw
    # log weights can lose several bits before the exponential.  A second
    # normalization keeps the serialized probabilities on the simplex even
    # when the absolute log-density scale is large.  Recompute ``logw`` from
    # those probabilities so the two public normalized representations agree.
    weight_sum = jnp.sum(weights)
    safe_weight_sum = jnp.where(
        has_support & (weight_sum > 0.0), weight_sum, jnp.ones_like(weight_sum)
    )
    weights = jnp.where(has_support, weights / safe_weight_sum, 0.0)
    logw = jnp.where(weights > 0.0, jnp.log(weights), -jnp.inf)
    log_norm = jnp.where(
        has_support,
        log_norm + jnp.log(safe_weight_sum),
        log_norm,
    )
    return logw, weights, log_norm


def compute_importance_weights(
    reduced_target_energy: Array,
    proposal_log_density: Array,
    *,
    valid_mask: Array | None = None,
    density_mode: str,
) -> WeightResult:
    """Compute normalized self-normalized importance weights without clipping."""

    reduced = jnp.asarray(reduced_target_energy)
    logq = jnp.asarray(proposal_log_density)
    if reduced.shape != logq.shape:
        raise ValueError(f"Target shape {reduced.shape} and logq shape {logq.shape} differ")
    valid = jnp.ones(reduced.shape, dtype=bool) if valid_mask is None else jnp.asarray(valid_mask, dtype=bool)
    raw = -reduced - logq
    valid = valid & jnp.isfinite(reduced) & jnp.isfinite(logq) & jnp.isfinite(raw)
    logw, weights, log_norm = _normalize_log_weights(raw, valid)
    return WeightResult(
        logw_raw=raw,
        logw=logw,
        weights=weights,
        proposal_log_density=logq,
        valid_mask=valid,
        log_normalizer=log_norm,
        density_mode=density_mode,
    )


def compute_exact_ambient_weights(
    augmented_reduced_energy: Array,
    logq_ambient: Array,
    *,
    valid_mask: Array | None = None,
) -> WeightResult:
    """Weights on the full-rank ambient space including the auxiliary COM target."""

    return compute_importance_weights(
        augmented_reduced_energy,
        logq_ambient,
        valid_mask=valid_mask,
        density_mode="ambient_exact",
    )


def compute_cgbg_compat_weights(
    pmf_energy: Array,
    kT: float | Array,
    logq_ambient: Array,
    standardized_ambient: Array,
    physical_std: float | Array,
    *,
    valid_mask: Array | None = None,
) -> WeightResult:
    """Reproduce CG-BG's radial-COM and scalar-standardisation convention."""

    corrected_logq = cgbg_correct_log_density(logq_ambient, standardized_ambient, physical_std)
    reduced = jnp.asarray(pmf_energy) / jnp.asarray(kT, dtype=jnp.asarray(pmf_energy).dtype)
    return compute_importance_weights(
        reduced,
        corrected_logq,
        valid_mask=valid_mask,
        density_mode="cgbg_compat",
    )


def clip_log_weights_for_diagnostics(
    logw_raw: Array,
    percentile: float,
    *,
    mode: str = "cap",
) -> Array:
    """Explicit diagnostic clipping; never used by formal weight functions.

    ``mode='cap'`` winsorises large log weights. ``mode='drop'`` reproduces the
    upstream plotting behaviour by assigning the upper tail zero probability.
    """

    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must lie in [0,100]")
    values = jnp.asarray(logw_raw)
    finite = jnp.isfinite(values)
    threshold = jnp.nanpercentile(jnp.where(finite, values, jnp.nan), percentile)
    if mode == "cap":
        return jnp.where(finite, jnp.minimum(values, threshold), -jnp.inf)
    if mode == "drop":
        return jnp.where(finite & (values < threshold), values, -jnp.inf)
    raise ValueError("mode must be 'cap' or 'drop'")
