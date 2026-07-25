from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.process import EDMSDE, backward_matching_loss, backward_score_target


def test_backward_target_is_reference_score_without_diffusion_square() -> None:
    sde = EDMSDE(sigma_min=0.03, sigma_max=0.9, rho=2.5)
    endpoint_0 = jnp.asarray([[[0.1, -0.2]], [[0.4, 0.7]]], dtype=jnp.float64)
    state_t = jnp.asarray([[[0.5, 0.6]], [[-0.1, 0.2]]], dtype=jnp.float64)
    time = jnp.asarray([0.35, 0.65], dtype=jnp.float64)

    expected = (endpoint_0 - state_t) / sde.integrated_variance(time[:, None, None])
    actual = backward_score_target(sde, time, endpoint_0, state_t)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    wrongly_scaled = expected * jnp.square(sde.diffusion(time))[:, None, None]
    assert not np.allclose(np.asarray(actual), np.asarray(wrongly_scaled))


def test_backward_matching_loss_is_plain_score_mse() -> None:
    prediction = jnp.asarray([[1.0, -1.0], [2.0, 3.0]])
    target = jnp.asarray([[0.0, -1.0], [1.0, 5.0]])
    expected = jnp.mean(jnp.square(prediction - target))
    assert float(backward_matching_loss(prediction, target)) == pytest.approx(float(expected))
