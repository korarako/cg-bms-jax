"""Nelson forward bridge-matching targets and loss helpers."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cg_bms_jax.process.bridge import endpoint_conditional_score
from cg_bms_jax.process.sde import EDMSDE

Array = jax.Array


class MatchingLoss(NamedTuple):
    """Total loss and its matching and optional damping components."""

    total: Array
    matching: Array
    damping: Array


def _broadcast_batch(values: Array | float, reference: Array) -> Array:
    values = jnp.asarray(values, dtype=reference.dtype)
    if values.ndim == 0:
        values = jnp.broadcast_to(values, (reference.shape[0],))
    if values.shape != (reference.shape[0],):
        raise ValueError(f"time must be scalar or have shape ({reference.shape[0]},)")
    return values.reshape((reference.shape[0],) + (1,) * (reference.ndim - 1))


def nelson_target(
    sde: EDMSDE,
    time: Array | float,
    endpoint_0: Array,
    endpoint_1: Array,
    source_score: Array,
    target_score: Array,
) -> Array:
    r"""Return the generalized BMS target.

    ``source_score`` is evaluated at ``endpoint_0`` and ``target_score`` at
    ``endpoint_1``.  The endpoints form the independent BMS coupling.
    """

    arrays = tuple(jnp.asarray(value) for value in (endpoint_0, endpoint_1, source_score, target_score))
    if len({array.shape for array in arrays}) != 1 or arrays[0].ndim < 2:
        raise ValueError("endpoints and scores must share one batch-plus-event shape")
    endpoint_0, endpoint_1, source_score, target_score = arrays
    time_b = _broadcast_batch(time, endpoint_0)
    gamma = sde.integrated_variance(time_b) / sde.total_variance
    reference_endpoint_score = endpoint_conditional_score(sde, endpoint_0, endpoint_1)
    return gamma * (source_score + target_score) - reference_endpoint_score


def forward_matching_loss(
    prediction: Array,
    target: Array,
    *,
    previous_prediction: Array | None = None,
    damping: float = 0.0,
) -> MatchingLoss:
    """Squared matching loss with the BMS damped fixed-point penalty."""

    prediction = jnp.asarray(prediction)
    target = jax.lax.stop_gradient(jnp.asarray(target))
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")

    matching = jnp.mean(jnp.square(prediction - target))
    if previous_prediction is None:
        if damping != 0.0:
            raise ValueError("previous_prediction is required when damping is non-zero")
        damping_loss = jnp.zeros((), dtype=matching.dtype)
    else:
        previous_prediction = jax.lax.stop_gradient(jnp.asarray(previous_prediction))
        if previous_prediction.shape != prediction.shape:
            raise ValueError("previous_prediction must have the prediction shape")
        damping_loss = jnp.mean(jnp.square(prediction - previous_prediction))
    return MatchingLoss(total=matching + damping * damping_loss, matching=matching, damping=damping_loss)
