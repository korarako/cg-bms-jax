"""Optimizer state and Flax-controller adapters used by both BMS stages.

The training layer deliberately does not depend on a particular controller
module.  A controller is any callable with the signature
``apply(params, constants, time, state) -> control``.  ``flax_apply`` adapts a
normal ``flax.linen.Module.apply`` method to that small protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

Array = jax.Array
PyTree = Any
ControlApply = Callable[[PyTree, PyTree, Array, Array], Array]


class ControllerTrainState(NamedTuple):
    """Pure-JAX controller state suitable for ``jax.jit`` and checkpointing."""

    params: PyTree
    constants: PyTree
    optimizer_state: optax.OptState
    step: Array


def split_flax_variables(variables: Mapping[str, PyTree]) -> tuple[PyTree, dict[str, PyTree]]:
    """Split a Flax variables mapping into trainable and non-trainable trees."""

    if "params" not in variables:
        raise KeyError("Flax variables must contain a 'params' collection")
    params = variables["params"]
    constants = {name: value for name, value in variables.items() if name != "params"}
    return params, constants


def flax_apply(module_apply: Callable[..., Array], **apply_kwargs: Any) -> ControlApply:
    """Adapt ``Module.apply(variables, time, state)`` to ``ControlApply``.

    Non-parameter Flax collections (for example the faithful PaiNN constant
    atom indices) are carried separately so that optimizers never see them.
    Extra keyword arguments are static and forwarded to ``Module.apply``.
    """

    def apply(params: PyTree, constants: PyTree, time: Array, state: Array) -> Array:
        variables = {"params": params}
        if constants is not None:
            variables.update(dict(constants))
        return module_apply(variables, time, state, **apply_kwargs)

    return apply


def make_optimizer(
    *,
    learning_rate: float | optax.Schedule,
    gradient_clip_norm: float,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Construct the common clipped AdamW optimizer."""

    if isinstance(learning_rate, (int, float)) and learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    return optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )


def make_learning_rate(
    *,
    peak: float,
    final: float | None = None,
    warmup_updates: int = 0,
    decay_updates: int | None = None,
    total_updates: int | None = None,
    warmup_start_factor: float = 1.0e-3,
) -> float | optax.Schedule:
    """Build the BMS warmup/cosine schedule, or return a constant rate.

    Keeping schedule construction shared is important for Orbax: inference
    restores a typed optimizer-state template even though it only consumes the
    controller variables, and the template PyTree must match training exactly.
    """

    if peak <= 0.0:
        raise ValueError("peak learning rate must be positive")
    if warmup_updates < 0:
        raise ValueError("warmup_updates must be non-negative")
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if final is not None and not 0.0 <= final <= peak:
        raise ValueError("final learning rate must be in [0, peak]")
    if not warmup_updates and final is None:
        return peak
    horizon = decay_updates if decay_updates is not None else total_updates
    if horizon is None or horizon <= 0:
        raise ValueError("a positive learning-rate decay horizon is required")
    if horizon < warmup_updates:
        raise ValueError("learning-rate decay horizon cannot precede warmup")

    schedules: list[optax.Schedule] = []
    boundaries: list[int] = []
    if warmup_updates:
        schedules.append(
            optax.linear_schedule(
                init_value=peak * warmup_start_factor,
                end_value=peak,
                transition_steps=warmup_updates,
            )
        )
        boundaries.append(warmup_updates)
    schedules.append(
        optax.cosine_decay_schedule(
            init_value=peak,
            decay_steps=max(horizon - warmup_updates, 1),
            alpha=(peak if final is None else final) / peak,
        )
    )
    return (
        schedules[0]
        if len(schedules) == 1
        else optax.join_schedules(schedules, boundaries)
    )


def initialize_train_state(
    params: PyTree,
    constants: PyTree,
    optimizer: optax.GradientTransformation,
) -> ControllerTrainState:
    """Initialize optimizer slots while leaving Flax constants untouched."""

    return ControllerTrainState(
        params=params,
        constants={} if constants is None else constants,
        optimizer_state=optimizer.init(params),
        step=jnp.asarray(0, dtype=jnp.int64 if jax.config.jax_enable_x64 else jnp.int32),
    )


def tree_all_finite(tree: PyTree) -> Array:
    """Return one device boolean indicating whether every leaf is finite."""

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves]))
