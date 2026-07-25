"""Backward bridge-matching targets."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cg_bms_jax.process.bridge import conditional_score_t0
from cg_bms_jax.process.sde import EDMSDE

Array = jax.Array


def backward_score_target(
    sde: EDMSDE,
    time: Array | float,
    endpoint_0: Array,
    state_t: Array,
) -> Array:
    r"""Return ``grad_Xt log p(X_t | X_0)`` without a ``g(t)^2`` factor."""

    return conditional_score_t0(sde, time, endpoint_0, state_t)


def backward_matching_loss(prediction: Array, target: Array) -> Array:
    """Mean squared backward score-matching loss."""

    prediction = jnp.asarray(prediction)
    target = jax.lax.stop_gradient(jnp.asarray(target))
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    return jnp.mean(jnp.square(prediction - target))
