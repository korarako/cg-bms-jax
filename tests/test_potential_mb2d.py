from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.potential import (
    AnalyticMB2DPotential,
    CartesianBoxDomain,
    muller_brown_energy,
)


def test_mb2d_affine_energy_and_gradient_include_scale_chain_rule() -> None:
    potential = AnalyticMB2DPotential(
        beta=0.4,
        offset=(25.0, 25.0),
        scale=(10.0, 5.0),
        confinement_strength=0.0,
    )
    state = jnp.asarray([0.3, -0.8], dtype=jnp.float32)
    physical = potential.state_to_physical(state)
    result = potential.evaluate(state)

    physical_gradient = jax.grad(lambda xy: muller_brown_energy(xy))(physical)
    expected_state_gradient = physical_gradient * jnp.asarray(potential.scale)
    np.testing.assert_allclose(result.energy, muller_brown_energy(physical), rtol=2e-6)
    np.testing.assert_allclose(
        result.gradient,
        expected_state_gradient,
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        result.reduced_energy,
        potential.beta * result.energy,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.score,
        -potential.beta * expected_state_gradient,
        rtol=3e-5,
        atol=3e-5,
    )
    assert potential.kT == pytest.approx(2.5)
    assert bool(result.valid_mask)


def test_mb2d_gradient_matches_central_finite_difference() -> None:
    potential = AnalyticMB2DPotential(beta=0.8)
    state = jnp.asarray([0.65, -0.45], dtype=jnp.float32)
    _, automatic = potential.energy_and_grad(state)
    epsilon = 2.0e-4
    basis = jnp.eye(2, dtype=state.dtype)
    finite_difference = jnp.asarray(
        [
            (
                potential.energy(state + epsilon * basis[index])
                - potential.energy(state - epsilon * basis[index])
            )
            / (2.0 * epsilon)
            for index in range(2)
        ]
    )
    np.testing.assert_allclose(
        automatic,
        finite_difference,
        rtol=5e-3,
        atol=2e-2,
    )


def test_mb2d_inside_edge_and_outside_have_finite_restoring_scores() -> None:
    potential = AnalyticMB2DPotential()
    state = jnp.asarray(
        [
            [0.0, 0.0],
            [2.5, 0.0],
            [10.0, -10.0],
        ],
        dtype=jnp.float32,
    )
    result = potential.evaluate(state)
    np.testing.assert_array_equal(
        result.valid_mask,
        np.asarray([True, True, False]),
    )
    assert np.asarray(jnp.isfinite(result.energy)).all()
    assert np.asarray(jnp.isfinite(result.gradient)).all()
    assert np.asarray(jnp.isfinite(result.score)).all()
    # The confinement dominates far outside: score points back toward the box.
    assert float(result.score[2, 0]) < 0.0
    assert float(result.score[2, 1]) > 0.0


def test_mb2d_formal_target_excludes_training_confinement() -> None:
    potential = AnalyticMB2DPotential(confinement_strength=3.0)
    state = jnp.asarray([[2.4, 0.0]], dtype=jnp.float32)
    training = potential.evaluate(state, include_training_wall=True)
    formal = potential.evaluate(state, include_training_wall=False)

    assert float(training.components["U_confinement"][0]) > 0.0
    np.testing.assert_allclose(formal.components["U_confinement"], 0.0)
    np.testing.assert_allclose(
        formal.energy,
        formal.components["U_muller_brown"],
        rtol=1.0e-6,
    )
    assert float(training.energy[0]) > float(formal.energy[0])


def test_mb2d_outside_extension_is_gradient_continuous_at_formal_edge() -> None:
    potential = AnalyticMB2DPotential()
    edge = potential.state_domain.upper[0]
    epsilon = 1.0e-4
    states = jnp.asarray(
        [
            [edge - epsilon, 0.0],
            [edge + epsilon, 0.0],
        ],
        dtype=jnp.float32,
    )
    gradients = potential.evaluate(states).gradient
    np.testing.assert_allclose(
        gradients[0],
        gradients[1],
        rtol=2e-3,
        atol=1.0,
    )


def test_mb2d_midpoint_grid_is_discretely_normalized() -> None:
    potential = AnalyticMB2DPotential()
    reference = potential.normalized_grid((96, 80))
    assert reference.state.shape == (96, 80, 2)
    assert reference.physical.shape == (96, 80, 2)
    assert np.asarray(jnp.isfinite(reference.reduced_energy)).all()
    assert np.isfinite(float(reference.log_normalizer_state))
    np.testing.assert_allclose(
        jnp.sum(reference.probability_mass),
        1.0,
        rtol=2e-6,
        atol=2e-6,
    )
    assert np.asarray(potential.formal_support_mask(reference.state)).all()


def test_cartesian_domain_margin_is_not_part_of_formal_support() -> None:
    domain = CartesianBoxDomain(
        lower=(-2.5, -2.5),
        upper=(2.5, 2.5),
        margin=(0.25, 0.25),
    )
    point = jnp.asarray([2.4, 0.0])
    assert bool(domain.support_mask(point))
    assert not bool(domain.support_mask(point, use_margin=True))
    assert domain.metadata()["support_mode"] == "cartesian_box"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"beta": 0.0}, "beta"),
        ({"scale": (1.0, 0.0)}, "scale"),
        ({"physical_box": ((1.0, 0.0), (0.0, 1.0))}, "lower bound"),
        ({"confinement_margin": (25.0, 1.0)}, "margin"),
        ({"extension_width": (0.0, 1.0)}, "extension_width"),
    ],
)
def test_mb2d_rejects_invalid_target_parameters(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        AnalyticMB2DPotential(**kwargs)
