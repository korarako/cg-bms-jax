"""Closed-form full-rank Brownian bridge operations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cg_bms_jax.process.sde import EDMSDE

Array = jax.Array


def _broadcast_batch(values: Array | float, reference: Array) -> Array:
    values = jnp.asarray(values, dtype=reference.dtype)
    if values.ndim == 0:
        values = jnp.broadcast_to(values, (reference.shape[0],))
    if values.shape != (reference.shape[0],):
        raise ValueError(f"time must be scalar or have shape ({reference.shape[0]},)")
    return values.reshape((reference.shape[0],) + (1,) * (reference.ndim - 1))


def sample_bridge(
    key: Array,
    sde: EDMSDE,
    time: Array | float,
    endpoint_0: Array,
    endpoint_1: Array,
) -> Array:
    r"""Draw from the reference bridge ``p(X_t | X_0, X_1)``."""

    endpoint_0 = jnp.asarray(endpoint_0)
    endpoint_1 = jnp.asarray(endpoint_1)
    if endpoint_0.shape != endpoint_1.shape or endpoint_0.ndim < 2:
        raise ValueError("endpoints must have an identical batch-plus-event shape")

    time_b = _broadcast_batch(time, endpoint_0)
    total_variance = sde.total_variance
    remaining_variance = sde.remaining_variance(time_b)
    remaining_fraction = remaining_variance / total_variance
    elapsed_fraction = 1.0 - remaining_fraction
    variance = remaining_variance * elapsed_fraction
    mean = remaining_fraction * endpoint_0 + elapsed_fraction * endpoint_1
    noise = jax.random.normal(key, endpoint_0.shape, dtype=endpoint_0.dtype)
    return mean + jnp.sqrt(jnp.maximum(variance, 0.0)) * noise


def conditional_score_t0(
    sde: EDMSDE,
    time: Array | float,
    endpoint_0: Array,
    state_t: Array,
) -> Array:
    r"""Compute ``grad_Xt log p(X_t | X_0) = (X_0-X_t)/kappa(t)``."""

    endpoint_0 = jnp.asarray(endpoint_0)
    state_t = jnp.asarray(state_t)
    if endpoint_0.shape != state_t.shape or endpoint_0.ndim < 2:
        raise ValueError("endpoint_0 and state_t must have an identical batch-plus-event shape")
    time_b = _broadcast_batch(time, endpoint_0)
    return (endpoint_0 - state_t) / sde.integrated_variance(time_b)


def conditional_score_1t(
    sde: EDMSDE,
    time: Array | float,
    endpoint_1: Array,
    state_t: Array,
) -> Array:
    r"""Compute ``grad_Xt log p(X_1 | X_t)`` for the reference process.

    For the zero-drift EDM reference diffusion, the remaining transition
    variance is ``kappa(1) - kappa(t)``.  This is the supervised
    bridge-matching target used by WT-ASBS pretraining (its Eq. 66), before the
    variance weighting in the regression loss.
    """

    endpoint_1 = jnp.asarray(endpoint_1)
    state_t = jnp.asarray(state_t)
    if endpoint_1.shape != state_t.shape or endpoint_1.ndim < 2:
        raise ValueError("endpoint_1 and state_t must have an identical batch-plus-event shape")
    time_b = _broadcast_batch(time, endpoint_1)
    remaining_variance = sde.remaining_variance(time_b)
    return (endpoint_1 - state_t) / remaining_variance


def endpoint_conditional_score(sde: EDMSDE, endpoint_0: Array, endpoint_1: Array) -> Array:
    r"""Compute ``grad_X1 log p(X_1 | X_0)`` at reference time one."""

    endpoint_0 = jnp.asarray(endpoint_0)
    endpoint_1 = jnp.asarray(endpoint_1)
    if endpoint_0.shape != endpoint_1.shape:
        raise ValueError("endpoints must have identical shapes")
    return (endpoint_0 - endpoint_1) / sde.total_variance
