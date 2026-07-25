"""Euler--Maruyama simulation implemented with ``jax.lax.scan``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from cg_bms_jax.process.sde import EDMSDE

Array = jax.Array
ControlApply = Callable[[Any, Array, Array], Array]


class EulerMaruyamaResult(NamedTuple):
    """Final state and trajectory including the supplied initial state."""

    final: Array
    trajectory: Array


def euler_maruyama_rollout(
    key: Array,
    initial_state: Array,
    *,
    sde: EDMSDE,
    control_apply: ControlApply,
    control_params: Any,
    steps: int,
    t0: float = 0.0,
    t1: float = 1.0,
    zero_last_noise: bool = False,
) -> EulerMaruyamaResult:
    r"""Simulate ``dX = g(t)^2 u(t,X)dt + g(t)dW`` in full rank."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if not t0 < t1:
        raise ValueError("t0 must be smaller than t1")

    initial_state = jnp.asarray(initial_state)
    if initial_state.ndim < 2:
        raise ValueError("initial_state must contain batch and event dimensions")
    times = jnp.linspace(t0, t1, steps + 1, dtype=initial_state.dtype)[:-1]
    dt = jnp.asarray((t1 - t0) / steps, dtype=initial_state.dtype)
    keys = jax.random.split(key, steps)
    indices = jnp.arange(steps, dtype=jnp.int32)

    def step(state: Array, inputs: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        time, noise_key, index = inputs
        diffusion = jnp.asarray(sde.diffusion(time), dtype=state.dtype)
        control = jnp.asarray(control_apply(control_params, time, state), dtype=state.dtype)
        noise = jax.random.normal(noise_key, state.shape, dtype=state.dtype)
        if zero_last_noise:
            noise = jnp.where(index == steps - 1, jnp.zeros_like(noise), noise)
        next_state = state + dt * jnp.square(diffusion) * control
        next_state = next_state + jnp.sqrt(dt) * diffusion * noise
        return next_state, next_state

    final, scanned = jax.lax.scan(step, initial_state, (times, keys, indices))
    trajectory = jnp.concatenate((initial_state[jnp.newaxis, ...], scanned), axis=0)
    return EulerMaruyamaResult(final=final, trajectory=trajectory)
