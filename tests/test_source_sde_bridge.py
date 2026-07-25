from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.data import (
    GaussianSource,
    extend_replay_buffer,
    init_replay_buffer,
    sample_replay_buffer,
)
from cg_bms_jax.process import (
    EDMSDE,
    conditional_score_1t,
    conditional_score_t0,
    endpoint_conditional_score,
    euler_maruyama_rollout,
    sample_bridge,
)


def test_full_rank_gaussian_source_log_prob_and_score() -> None:
    source = GaussianSource(event_shape=(6, 3), scale=1.7, mean=0.25)
    samples = source.sample(jax.random.PRNGKey(1), 8, dtype=jnp.float64)

    assert samples.shape == (8, 6, 3)
    expected = -0.5 * jnp.sum(jnp.square((samples - 0.25) / 1.7), axis=(1, 2))
    expected -= 18 * (jnp.log(1.7) + 0.5 * jnp.log(2.0 * jnp.pi))
    np.testing.assert_allclose(source.log_prob(samples), expected, rtol=1e-12, atol=1e-12)

    autodiff_score = jax.vmap(jax.grad(lambda value: source.log_prob(value[None, ...])[0]))(samples)
    np.testing.assert_allclose(source.score(samples), autodiff_score, rtol=1e-12, atol=1e-12)

    # The ambient source must not silently project away COM.
    assert not np.allclose(np.asarray(samples.mean(axis=1)), 0.0)


def test_edm_integrated_variance_has_correct_derivative() -> None:
    sde = EDMSDE(sigma_min=1.0e-3, sigma_max=1.2, rho=3.0)
    assert float(sde.integrated_variance(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(sde.total_variance) > 0.0

    time = jnp.asarray(0.37, dtype=jnp.float64)
    derivative = jax.grad(sde.integrated_variance)(time)
    np.testing.assert_allclose(derivative, jnp.square(sde.diffusion(time)), rtol=1e-10, atol=1e-10)


def test_edm_remaining_variance_stays_positive_in_float32_near_terminal_time() -> None:
    sde = EDMSDE(sigma_min=0.00101656, sigma_max=1.0, rho=3.0)
    time = jnp.asarray([0.98, 0.9883956909, 0.99], dtype=jnp.float32)

    stable = sde.remaining_variance(time)
    subtractive = jnp.asarray(sde.total_variance, dtype=jnp.float32) - jnp.asarray(
        sde.integrated_variance(time), dtype=jnp.float32
    )

    assert np.all(np.isfinite(np.asarray(stable)))
    assert np.all(np.asarray(stable) > 0.0)
    # This is the exact failure mode observed in real Ala2 warm pretraining:
    # float32 subtraction rounds at least one positive remainder to zero.
    assert np.any(np.asarray(subtractive) == 0.0)

    remaining_at_start = sde.remaining_variance(jnp.asarray(0.0, dtype=jnp.float32))
    remaining_at_end = sde.remaining_variance(jnp.asarray(1.0, dtype=jnp.float32))
    np.testing.assert_allclose(
        remaining_at_start,
        jnp.asarray(sde.total_variance, dtype=jnp.float32),
        rtol=1.0e-6,
        atol=0.0,
    )
    assert float(remaining_at_end) == 0.0

    endpoint_1 = jnp.ones((3, 6, 3), dtype=jnp.float32)
    state_t = endpoint_1 + jnp.asarray(1.0e-4, dtype=jnp.float32)
    score = conditional_score_1t(sde, time, endpoint_1, state_t)
    assert np.all(np.isfinite(np.asarray(score)))


def test_full_rank_bridge_and_conditional_scores_match_closed_form() -> None:
    sde = EDMSDE(sigma_min=0.01, sigma_max=0.8, rho=3.0)
    endpoint_0 = jnp.arange(4 * 6 * 3, dtype=jnp.float64).reshape(4, 6, 3) / 30.0
    endpoint_1 = endpoint_0[::-1] + 0.4
    time = jnp.asarray([0.2, 0.4, 0.6, 0.8], dtype=jnp.float64)
    key = jax.random.PRNGKey(4)

    sample = sample_bridge(key, sde, time, endpoint_0, endpoint_1)
    time_b = time[:, None, None]
    fraction = sde.integrated_variance(time_b) / sde.total_variance
    variance = sde.total_variance * fraction * (1.0 - fraction)
    expected = (1.0 - fraction) * endpoint_0 + fraction * endpoint_1
    expected += jnp.sqrt(variance) * jax.random.normal(key, endpoint_0.shape, dtype=endpoint_0.dtype)
    np.testing.assert_allclose(sample, expected, rtol=1e-12, atol=1e-12)

    expected_score = (endpoint_0 - sample) / sde.integrated_variance(time_b)
    np.testing.assert_allclose(
        conditional_score_t0(sde, time, endpoint_0, sample),
        expected_score,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        endpoint_conditional_score(sde, endpoint_0, endpoint_1),
        (endpoint_0 - endpoint_1) / sde.total_variance,
        rtol=1e-12,
        atol=1e-12,
    )


def test_euler_maruyama_rollout_uses_scan_and_preserves_ambient_state() -> None:
    sde = EDMSDE(sigma_min=0.02, sigma_max=0.3, rho=2.0)
    initial = jnp.ones((3, 6, 3), dtype=jnp.float64)

    def zero_control(_params, _time, state):
        return jnp.zeros_like(state)

    result = jax.jit(
        lambda key, value: euler_maruyama_rollout(
            key,
            value,
            sde=sde,
            control_apply=zero_control,
            control_params=None,
            steps=5,
        )
    )(jax.random.PRNGKey(9), initial)

    assert result.final.shape == initial.shape
    assert result.trajectory.shape == (6, *initial.shape)
    np.testing.assert_allclose(result.trajectory[0], initial, rtol=0.0, atol=0.0)
    assert not np.allclose(np.asarray(result.final.mean(axis=1)), 0.0)


def test_device_ring_replay_buffer_wraps_and_samples_valid_rows() -> None:
    example = {
        "endpoint_0": jnp.zeros((2, 1), dtype=jnp.float32),
        "endpoint_1": jnp.zeros((2, 1), dtype=jnp.float32),
    }
    state = init_replay_buffer(3, example)
    state = jax.jit(extend_replay_buffer)(
        state,
        {
            "endpoint_0": jnp.asarray([[0.0], [1.0]]),
            "endpoint_1": jnp.asarray([[10.0], [11.0]]),
        },
    )
    state = extend_replay_buffer(
        state,
        {
            "endpoint_0": jnp.asarray([[2.0], [3.0]]),
            "endpoint_1": jnp.asarray([[12.0], [13.0]]),
        },
    )

    assert int(state.size) == 3
    assert int(state.cursor) == 1
    np.testing.assert_array_equal(np.sort(np.asarray(state.storage["endpoint_0"]).ravel()), [1.0, 2.0, 3.0])

    sampled = jax.jit(lambda key, value: sample_replay_buffer(key, value, 20))(
        jax.random.PRNGKey(2),
        state,
    )
    assert set(np.asarray(sampled["endpoint_0"]).ravel()).issubset({1.0, 2.0, 3.0})
    np.testing.assert_allclose(sampled["endpoint_1"] - sampled["endpoint_0"], 10.0)
