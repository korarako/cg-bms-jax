"""Probability-flow ODE and exact full-rank likelihood integration."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import diffrax
import jax
import jax.numpy as jnp

from cg_bms_jax.data.source import GaussianSource
from cg_bms_jax.process.sde import EDMSDE

Array = jax.Array
Velocity = Callable[[Array, Array], Array]
ControllerApply = Callable[[Any, Array, Array], Array]


@dataclass(frozen=True)
class ProbabilityFlowConfig:
    """Numerical configuration for a forward probability-flow solve."""

    t0: float = 0.0
    t1: float = 1.0
    dt0: float = 1.0e-3
    rtol: float = 1.0e-5
    atol: float = 1.0e-6
    max_steps: int = 100_000

    def __post_init__(self) -> None:
        finite_positive = (self.dt0, self.rtol, self.atol)
        if not self.t0 < self.t1:
            raise ValueError("t0 must be smaller than t1")
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in finite_positive):
            raise ValueError("dt0, rtol and atol must be finite and positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


class ProbabilityFlowResult(NamedTuple):
    """Terminal samples and their likelihood from one coupled ODE solve."""

    samples: Array
    log_prob: Array
    initial_samples: Array
    initial_log_prob: Array
    num_steps: Array


class FlowMapResult(NamedTuple):
    """Terminal coordinates from the coordinate-only PF solve."""

    samples: Array
    initial_samples: Array
    num_steps: Array


def bms_probability_flow_velocity(
    sde: EDMSDE,
    forward_apply: ControllerApply,
    forward_params: Any,
    backward_apply: ControllerApply,
    backward_params: Any,
    time: Array | float,
    state: Array,
) -> Array:
    r"""Evaluate ``0.5 g(t)^2 (u(t,x)-v(t,x))``."""

    state = jnp.asarray(state)
    time = jnp.asarray(time, dtype=state.dtype)
    forward = jnp.asarray(forward_apply(forward_params, time, state), dtype=state.dtype)
    backward = jnp.asarray(backward_apply(backward_params, time, state), dtype=state.dtype)
    if forward.shape != state.shape or backward.shape != state.shape:
        raise ValueError("forward and backward controllers must return the state shape")
    diffusion_square = jnp.square(jnp.asarray(sde.diffusion(time), dtype=state.dtype))
    return 0.5 * diffusion_square * (forward - backward)


def exact_divergence(velocity: Velocity, time: Array | float, state: Array) -> Array:
    """Compute one exact Jacobian trace per sample with ``jacrev`` and ``vmap``."""

    state = jnp.asarray(state)
    if state.ndim < 2:
        raise ValueError("state must contain batch and event dimensions")
    event_shape = state.shape[1:]
    flat_state = state.reshape((state.shape[0], -1))

    def single_velocity(flat_sample: Array) -> Array:
        sample = flat_sample.reshape((1, *event_shape))
        output = jnp.asarray(velocity(jnp.asarray(time, dtype=sample.dtype), sample))
        if output.shape != sample.shape:
            raise ValueError("velocity must preserve the state shape")
        return output.reshape((-1,))

    def single_divergence(flat_sample: Array) -> Array:
        jacobian = jax.jacrev(single_velocity)(flat_sample)
        return jnp.trace(jacobian)

    return jax.vmap(single_divergence)(flat_state)


def integrate_probability_flow(
    velocity: Velocity,
    initial_samples: Array,
    initial_log_prob: Array,
    *,
    config: ProbabilityFlowConfig | None = None,
) -> ProbabilityFlowResult:
    r"""Integrate ``(X, log q)`` together using Diffrax Dopri5.

    Keeping coordinates and likelihood in one PyTree solve prevents attaching
    a likelihood computed on one numerical path to samples from another path.
    """

    config = ProbabilityFlowConfig() if config is None else config
    initial_samples = jnp.asarray(initial_samples)
    initial_log_prob = jnp.asarray(initial_log_prob, dtype=initial_samples.dtype)
    if initial_samples.ndim < 2:
        raise ValueError("initial_samples must contain batch and event dimensions")
    if initial_log_prob.shape != (initial_samples.shape[0],):
        raise ValueError("initial_log_prob must have shape (batch,)")

    def dynamics(time: Array, state: tuple[Array, Array], _args: None) -> tuple[Array, Array]:
        samples, log_prob = state
        sample_velocity = velocity(time, samples)
        divergence = exact_divergence(velocity, time, samples)
        return sample_velocity, -divergence.astype(log_prob.dtype)

    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(dynamics),
        solver=diffrax.Dopri5(),
        t0=config.t0,
        t1=config.t1,
        dt0=config.dt0,
        y0=(initial_samples, initial_log_prob),
        args=None,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=config.rtol, atol=config.atol),
        max_steps=config.max_steps,
        throw=True,
    )
    sample_path, log_prob_path = solution.ys
    return ProbabilityFlowResult(
        samples=sample_path[-1],
        log_prob=log_prob_path[-1],
        initial_samples=initial_samples,
        initial_log_prob=initial_log_prob,
        num_steps=jnp.asarray(solution.stats["num_steps"]),
    )


def integrate_flow_map(
    velocity: Velocity,
    initial_samples: Array,
    *,
    config: ProbabilityFlowConfig | None = None,
) -> FlowMapResult:
    """Integrate only coordinates with the same Dopri5/PID contract.

    This path is intended for direct change-of-variables audits. Production
    sampling must continue to use :func:`integrate_probability_flow`, which
    binds coordinates and integrated ``logq`` in one solve.
    """

    config = ProbabilityFlowConfig() if config is None else config
    initial_samples = jnp.asarray(initial_samples)
    if initial_samples.ndim < 2:
        raise ValueError("initial_samples must contain batch and event dimensions")

    def dynamics(time: Array, samples: Array, _args: None) -> Array:
        output = jnp.asarray(velocity(time, samples), dtype=samples.dtype)
        if output.shape != samples.shape:
            raise ValueError("velocity must preserve the state shape")
        return output

    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(dynamics),
        solver=diffrax.Dopri5(),
        t0=config.t0,
        t1=config.t1,
        dt0=config.dt0,
        y0=initial_samples,
        args=None,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(
            rtol=config.rtol,
            atol=config.atol,
        ),
        max_steps=config.max_steps,
        throw=True,
    )
    return FlowMapResult(
        samples=solution.ys[-1],
        initial_samples=initial_samples,
        num_steps=jnp.asarray(solution.stats["num_steps"]),
    )


def sample_probability_flow(
    key: Array,
    source: GaussianSource,
    batch_size: int,
    velocity: Velocity,
    *,
    dtype: jnp.dtype = jnp.float32,
    config: ProbabilityFlowConfig | None = None,
) -> ProbabilityFlowResult:
    """Draw the analytic source and integrate its samples and density together."""

    initial_samples = source.sample(key, batch_size, dtype=dtype)
    initial_log_prob = source.log_prob(initial_samples)
    return integrate_probability_flow(
        velocity,
        initial_samples,
        initial_log_prob,
        config=config,
    )
