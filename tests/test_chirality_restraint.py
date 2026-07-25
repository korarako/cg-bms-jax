from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.potential import (
    BMS_CB_FORCE_CONSTANT_KJ_MOL,
    BMS_CB_LOCATION_RAD,
    BMS_IMPROPER_TOLERANCE_RAD,
    AmbientAla2Potential,
    BMSCBImproperRestraint,
    CGBGAla2Bundle,
    CGBGAla2PMF,
    periodic_displacement,
    torsion_angle,
)


def _core_beta_with_torsion(angle: float, *, dtype=jnp.float64) -> jax.Array:
    """Construct a non-degenerate core-beta geometry with the requested CV."""

    coordinates = jnp.zeros((6, 3), dtype=dtype)
    # Torsion indices are [CA, N, C, CB] == [2, 1, 4, 3].  With b1 along +x,
    # v along +y, and w=(0,cos(phi),sin(phi)), upstream BMS returns phi.
    coordinates = coordinates.at[2].set(jnp.asarray((0.0, 1.0, 0.0), dtype=dtype))
    coordinates = coordinates.at[1].set(jnp.asarray((0.0, 0.0, 0.0), dtype=dtype))
    coordinates = coordinates.at[4].set(jnp.asarray((1.0, 0.0, 0.0), dtype=dtype))
    coordinates = coordinates.at[3].set(
        jnp.asarray((1.0, math.cos(angle), math.sin(angle)), dtype=dtype)
    )
    # The unused ACE-C and NME-N beads are deliberately arbitrary.
    coordinates = coordinates.at[0].set(jnp.asarray((-0.3, 0.2, 0.1), dtype=dtype))
    coordinates = coordinates.at[5].set(jnp.asarray((1.3, -0.2, 0.1), dtype=dtype))
    return coordinates


def test_torsion_matches_upstream_convention_and_periodic_displacement() -> None:
    for angle in (-2.4, -0.3, 0.7, 2.5):
        value = torsion_angle(_core_beta_with_torsion(angle), (2, 1, 4, 3))
        np.testing.assert_allclose(value, angle, rtol=1.0e-7, atol=1.0e-7)

    wrapped = periodic_displacement(-math.pi + 0.05, math.pi - 0.05)
    np.testing.assert_allclose(wrapped, 0.1, rtol=1.0e-6, atol=1.0e-6)


def test_bms_flat_bottom_zero_region_and_quadratic_excess() -> None:
    restraint = BMSCBImproperRestraint()
    inside = _core_beta_with_torsion(
        BMS_CB_LOCATION_RAD + 0.5 * BMS_IMPROPER_TOLERANCE_RAD
    )
    assert float(restraint.energy(inside)) == 0.0

    excess = 0.2
    outside = _core_beta_with_torsion(
        BMS_CB_LOCATION_RAD + BMS_IMPROPER_TOLERANCE_RAD + excess
    )
    expected = 0.5 * BMS_CB_FORCE_CONSTANT_KJ_MOL * excess**2
    np.testing.assert_allclose(restraint.energy(outside), expected, rtol=2.0e-6)


def test_cb_improper_gradient_matches_finite_difference() -> None:
    restraint = BMSCBImproperRestraint()
    coordinates = _core_beta_with_torsion(-0.8, dtype=jnp.float64)
    _energy, automatic = restraint.energy_and_grad(coordinates)
    step = 1.0e-4 if jax.config.read("jax_enable_x64") else 1.0e-3
    numerical = np.zeros((6, 3), dtype=np.float64)
    coordinates_np = np.asarray(coordinates)
    for particle in range(6):
        for component in range(3):
            plus = coordinates_np.copy()
            minus = coordinates_np.copy()
            plus[particle, component] += step
            minus[particle, component] -= step
            numerical[particle, component] = (
                float(restraint.energy(jnp.asarray(plus)))
                - float(restraint.energy(jnp.asarray(minus)))
            ) / (2.0 * step)
    np.testing.assert_allclose(automatic, numerical, rtol=3.0e-3, atol=8.0e-2)
    np.testing.assert_allclose(jnp.sum(automatic, axis=0), 0.0, atol=2.0e-4)


def test_reflection_is_selected_by_positive_cb_improper() -> None:
    restraint = BMSCBImproperRestraint()
    selected = _core_beta_with_torsion(BMS_CB_LOCATION_RAD)
    reflected = selected.at[:, 2].set(-selected[:, 2])
    np.testing.assert_allclose(restraint.torsion(selected), BMS_CB_LOCATION_RAD, atol=1.0e-7)
    np.testing.assert_allclose(restraint.torsion(reflected), -BMS_CB_LOCATION_RAD, atol=1.0e-7)
    assert float(restraint.energy(selected)) == 0.0
    assert float(restraint.energy(reflected)) > 0.0


def test_ambient_target_and_formal_reweight_include_same_cb_improper() -> None:
    bundle = CGBGAla2Bundle(
        data_path="unused.npz",
        checkpoint_path="unused.pkl",
        box_nm=((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0)),
        species=(0, 0, 0, 0, 0, 0),
        mask=(True,) * 6,
        reference_nm=((0.0, 0.0, 0.0),) * 6,
        standardization_std_nm=0.2,
    )

    def fractional_energy(fractional: jax.Array) -> jax.Array:
        relative = fractional[1:] - fractional[:1]
        return jnp.sum(relative * relative)

    pmf = CGBGAla2PMF(bundle, fractional_energy)
    restraint = BMSCBImproperRestraint()
    ambient = AmbientAla2Potential(
        pmf=pmf,
        physical_std_nm=0.2,
        cb_improper=restraint,
    )
    physical = _core_beta_with_torsion(-BMS_CB_LOCATION_RAD)
    standardized = physical / 0.2
    expected_target = pmf.energy(physical) + restraint.energy(physical)
    np.testing.assert_allclose(ambient.energy(standardized), expected_target)
    np.testing.assert_allclose(ambient.formal_reweight_energy(standardized), expected_target)

    evaluated = ambient.evaluate(standardized, include_training_wall=False)
    expected_reduced_gradient = jax.grad(
        lambda value: ambient.energy(value) / ambient.kT
    )(standardized)
    # The physical target is translation invariant, so lifting only adds the
    # analytically defined auxiliary COM gradient.
    com_gradient = jnp.broadcast_to(
        jnp.mean(standardized, axis=0), standardized.shape
    )
    np.testing.assert_allclose(
        evaluated.reduced_gradient,
        expected_reduced_gradient + com_gradient,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(evaluated.score, -evaluated.reduced_gradient)
    np.testing.assert_allclose(evaluated.energy, expected_target)
