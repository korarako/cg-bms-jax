from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.data import GaussianSource
from cg_bms_jax.process import (
    EDMSDE,
    ProbabilityFlowConfig,
    bms_probability_flow_velocity,
    exact_divergence,
    integrate_flow_map,
    integrate_probability_flow,
)


def test_exact_ambient_divergence_of_linear_velocity() -> None:
    dimension = 18
    rate = jnp.asarray(0.17, dtype=jnp.float64)
    state = jax.random.normal(jax.random.PRNGKey(3), (4, 6, 3), dtype=jnp.float64)

    def velocity(_time, value):
        return rate * value

    divergence = exact_divergence(velocity, 0.4, state)
    np.testing.assert_allclose(divergence, dimension * rate, rtol=1e-12, atol=1e-12)


def test_diffrax_couples_linear_samples_and_log_density() -> None:
    source = GaussianSource(event_shape=(2, 2), scale=1.0)
    initial = source.sample(jax.random.PRNGKey(7), 3, dtype=jnp.float64)
    initial_log_prob = source.log_prob(initial)
    rate = 0.2

    def velocity(_time, value):
        return rate * value

    config = ProbabilityFlowConfig(dt0=0.02, rtol=1.0e-8, atol=1.0e-10, max_steps=10_000)
    result = integrate_probability_flow(velocity, initial, initial_log_prob, config=config)

    duration = config.t1 - config.t0
    expected_samples = initial * jnp.exp(rate * duration)
    expected_log_prob = initial_log_prob - 4 * rate * duration
    np.testing.assert_allclose(result.samples, expected_samples, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(result.log_prob, expected_log_prob, rtol=2e-7, atol=2e-8)
    np.testing.assert_array_equal(result.initial_samples, initial)
    np.testing.assert_array_equal(result.initial_log_prob, initial_log_prob)


def test_coordinate_only_flow_map_matches_coupled_coordinate_path() -> None:
    initial = jax.random.normal(
        jax.random.PRNGKey(19),
        (5, 2),
        dtype=jnp.float64,
    )

    def velocity(_time, value):
        return 0.13 * value + jnp.asarray([0.2, -0.1])

    config = ProbabilityFlowConfig(
        dt0=0.02,
        rtol=1.0e-8,
        atol=1.0e-10,
        max_steps=10_000,
    )
    source = GaussianSource(event_shape=(2,), scale=1.0)
    coupled = integrate_probability_flow(
        velocity,
        initial,
        source.log_prob(initial),
        config=config,
    )
    coordinates = integrate_flow_map(velocity, initial, config=config)
    np.testing.assert_allclose(
        coordinates.samples,
        coupled.samples,
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    np.testing.assert_array_equal(coordinates.initial_samples, initial)


def test_bms_probability_flow_velocity_uses_forward_minus_backward() -> None:
    sde = EDMSDE(sigma_min=0.02, sigma_max=0.6, rho=3.0)
    state = jnp.asarray([[[1.0, -2.0, 0.5]]], dtype=jnp.float64)
    time = jnp.asarray(0.3, dtype=jnp.float64)

    def linear_control(params, _time, value):
        return params * value

    actual = bms_probability_flow_velocity(
        sde,
        linear_control,
        2.0,
        linear_control,
        -0.5,
        time,
        state,
    )
    expected = 0.5 * jnp.square(sde.diffusion(time)) * (2.0 - (-0.5)) * state
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    velocity = partial(
        bms_probability_flow_velocity,
        sde,
        linear_control,
        2.0,
        linear_control,
        -0.5,
    )
    assert exact_divergence(velocity, time, state).shape == (1,)
