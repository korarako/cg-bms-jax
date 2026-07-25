from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.process import EDMSDE, forward_matching_loss, nelson_target


def test_nelson_target_matches_bms_identity() -> None:
    sde = EDMSDE(sigma_min=0.01, sigma_max=0.7, rho=3.0)
    endpoint_0 = jnp.asarray([[[0.2, -0.1]], [[0.5, 0.3]]], dtype=jnp.float64)
    endpoint_1 = jnp.asarray([[[0.8, 0.4]], [[-0.2, 0.9]]], dtype=jnp.float64)
    source_score = -endpoint_0 / 1.3**2
    target_score = -2.0 * endpoint_1
    time = jnp.asarray([0.25, 0.75], dtype=jnp.float64)

    gamma = sde.integrated_variance(time[:, None, None]) / sde.total_variance
    reference_score = (endpoint_0 - endpoint_1) / sde.total_variance
    expected = gamma * (source_score + target_score) - reference_score
    actual = nelson_target(sde, time, endpoint_0, endpoint_1, source_score, target_score)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_forward_matching_loss_includes_damped_fixed_point_term() -> None:
    prediction = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    target = jnp.asarray([[0.0, 2.0], [1.0, 6.0]])
    previous = jnp.asarray([[0.5, 1.5], [2.5, 3.5]])
    result = forward_matching_loss(prediction, target, previous_prediction=previous, damping=3.0)

    expected_matching = jnp.mean(jnp.square(prediction - target))
    expected_damping = jnp.mean(jnp.square(prediction - previous))
    assert float(result.matching) == pytest.approx(float(expected_matching))
    assert float(result.damping) == pytest.approx(float(expected_damping))
    assert float(result.total) == pytest.approx(float(expected_matching + 3.0 * expected_damping))


def test_nonzero_damping_requires_previous_iterate() -> None:
    with pytest.raises(ValueError, match="previous_prediction"):
        forward_matching_loss(jnp.zeros((2, 3)), jnp.zeros((2, 3)), damping=1.0)
