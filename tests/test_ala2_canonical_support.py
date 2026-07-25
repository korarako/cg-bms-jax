from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.potential import (
    Ala2CanonicalSupport,
    AmbientAla2Potential,
    BMSCBImproperRestraint,
    CGBGAla2Bundle,
    CGBGAla2PMF,
    SmoothEnergyWindow,
    SmoothLowerEnergyBound,
)
from cg_bms_jax.runtime import compose_config


def _support() -> Ala2CanonicalSupport:
    return Ala2CanonicalSupport(
        box_lengths_nm=(3.70465, 3.70465, 3.70465),
        bond_indices=((0, 1), (1, 2), (2, 3), (2, 4), (4, 5)),
        bond_lower_nm=(0.110, 0.120, 0.130, 0.130, 0.110),
        bond_upper_nm=(0.155, 0.175, 0.180, 0.180, 0.155),
        bond_force_constant_kj_mol_nm2=10_000.0,
        angle_indices=((0, 1, 2), (1, 2, 3), (1, 2, 4), (3, 2, 4), (2, 4, 5)),
        angle_lower_rad=tuple(
            math.radians(value) for value in (95.0, 80.0, 80.0, 80.0, 90.0)
        ),
        angle_upper_rad=tuple(
            math.radians(value) for value in (155.0, 140.0, 145.0, 145.0, 145.0)
        ),
        angle_force_constant_kj_mol_rad2=250.0,
        repulsion_indices=(
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 5),
            (3, 4),
            (3, 5),
        ),
        repulsion_min_nm=0.180,
        repulsion_force_constant_kj_mol_nm2=10_000.0,
    )


def _canonical_geometry(dtype=jnp.float64) -> jax.Array:
    # Published core-beta topology fixture (PDB Angstrom coordinates / 10).
    return jnp.asarray(
        [
            [1.4303, 1.6649, 1.6521],
            [1.4543, 1.5434, 1.7069],
            [1.5854, 1.5006, 1.7625],
            [1.5606, 1.4144, 1.8832],
            [1.6746, 1.4237, 1.6655],
            [1.8068, 1.4548, 1.6583],
        ],
        dtype=dtype,
    )


def test_canonical_support_is_exactly_flat_on_molecular_fixture() -> None:
    support = _support()
    components = support.energy_components(_canonical_geometry())
    for value in components.values():
        np.testing.assert_allclose(value, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(support.energy(_canonical_geometry()), 0.0, atol=1.0e-12)


def test_canonical_support_restores_stretched_bond_and_repels_collision() -> None:
    support = _support()
    canonical = _canonical_geometry()

    # Move ACE-C directly away from ALA-N.  The derivative of the active bond
    # energy along that displacement must point back toward the flat region.
    direction = canonical[0] - canonical[1]
    direction = direction / jnp.linalg.norm(direction)
    stretched = canonical.at[0].add(0.10 * direction)
    components = support.energy_components(stretched)
    assert float(components["U_bond"]) > 0.0
    gradient = jax.grad(
        lambda value: support.energy_components(value)["U_bond"]
    )(stretched)
    force = -gradient
    assert float(jnp.dot(force[0] - force[1], direction)) < 0.0

    # Pair 0--5 is nonbonded.  A near collision must have positive energy and
    # a force that increases their separation.
    collided = canonical.at[5].set(canonical[0] + jnp.asarray((0.03, 0.0, 0.0)))
    components = support.energy_components(collided)
    assert float(components["U_repulsion"]) > 0.0
    gradient = jax.grad(
        lambda value: support.energy_components(value)["U_repulsion"]
    )(collided)
    separation = collided[5] - collided[0]
    assert float(jnp.dot((-gradient[5]) - (-gradient[0]), separation)) > 0.0


def test_canonical_support_is_translation_rotation_and_pbc_invariant() -> None:
    support = _support()
    coordinates = _canonical_geometry()
    angle = 0.73
    rotation = jnp.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=coordinates.dtype,
    )
    transformed = coordinates @ rotation.T + jnp.asarray((0.4, -0.7, 0.2))
    np.testing.assert_allclose(
        support.energy(transformed), support.energy(coordinates), rtol=1.0e-8, atol=1.0e-8
    )

    shifted_image = coordinates.at[0, 0].add(support.box_lengths_nm[0])
    np.testing.assert_allclose(
        support.energy(shifted_image), support.energy(coordinates), rtol=1.0e-8, atol=1.0e-8
    )

    distorted = coordinates.at[0, 0].add(0.2)
    _energy, gradient = support.energy_and_grad(distorted)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_allclose(jnp.sum(gradient, axis=0), 0.0, atol=2.0e-4)


def test_smooth_lower_bound_transforms_energy_and_gradient_together() -> None:
    floor = SmoothLowerEnergyBound(minimum_kj_mol=-330.0, softness_kj_mol=2.5)
    raw = jnp.asarray((-1000.0, -330.0, -313.6, -280.0))
    effective = floor.energy(raw)
    scale = floor.gradient_scale(raw)
    assert bool(jnp.all(effective >= -330.0))
    np.testing.assert_allclose(scale[0], 0.0, atol=1.0e-7)
    np.testing.assert_allclose(scale[1], 0.5, atol=1.0e-7)
    assert float(scale[2]) > 0.998
    np.testing.assert_allclose(effective[-1], raw[-1], atol=1.0e-6)
    automatic = jax.grad(lambda value: jnp.sum(floor.energy(value)))(raw)
    np.testing.assert_allclose(automatic, scale, rtol=1.0e-6, atol=1.0e-6)


def test_smooth_energy_window_saturates_both_ood_tails() -> None:
    window = SmoothEnergyWindow(-330.0, -230.0, 2.5, 2.5)
    raw = jnp.asarray((-10_000.0, -313.6, -280.0, -248.2, 10_000.0))
    effective = window.energy(raw)
    scale = window.gradient_scale(raw)
    assert bool(jnp.all(effective >= -330.0))
    assert bool(jnp.all(effective <= -230.0))
    assert float(scale[0]) < 1.0e-7
    assert float(scale[-1]) < 1.0e-7
    assert float(scale[1]) > 0.998
    assert float(scale[3]) > 0.999
    automatic = jax.grad(lambda value: jnp.sum(window.energy(value)))(raw)
    np.testing.assert_allclose(automatic, scale, rtol=1.0e-6, atol=1.0e-6)


def test_ambient_global_target_component_sum_and_standardized_gradient() -> None:
    box = ((3.70465, 0.0, 0.0), (0.0, 3.70465, 0.0), (0.0, 0.0, 3.70465))
    bundle = CGBGAla2Bundle(
        data_path="unused.npz",
        checkpoint_path="unused.pkl",
        box_nm=box,
        species=(1, 3, 1, 1, 1, 3),
        mask=(True,) * 6,
        reference_nm=tuple(tuple(float(value) for value in row) for row in _canonical_geometry()),
        standardization_std_nm=0.1,
    )

    def raw_pmf(fractional: jax.Array) -> jax.Array:
        physical = fractional * jnp.asarray((3.70465, 3.70465, 3.70465))
        centred = physical - jnp.mean(physical, axis=0, keepdims=True)
        return -350.0 + 4.0 * jnp.sum(centred**2)

    ambient = AmbientAla2Potential(
        pmf=CGBGAla2PMF(bundle, raw_pmf),
        physical_std_nm=0.1,
        pmf_lower_bound=SmoothEnergyWindow(-330.0, -230.0, 2.5, 2.5),
        canonical_support=_support(),
        pmf_gate_energy_scale_kj_mol=25.0,
    )
    standardized = _canonical_geometry() / 0.1
    components = ambient.formal_energy_components(standardized)
    np.testing.assert_allclose(components["U_pmf"], components["U_pmf_raw"])
    np.testing.assert_allclose(
        components["U_target"],
        components["U_pmf_gated"]
        + components["U_bond"]
        + components["U_angle"]
        + components["U_repulsion"]
        + components["U_cb"],
    )
    np.testing.assert_allclose(ambient.energy(standardized), components["U_target"])
    np.testing.assert_allclose(
        ambient.formal_reweight_energy(standardized), components["U_target"]
    )
    evaluated = ambient.evaluate(standardized, include_training_wall=False)
    np.testing.assert_allclose(evaluated.components["U_target"], evaluated.energy)
    automatic = jax.grad(lambda value: ambient.energy(value))(standardized)
    # The molecular target is translation invariant; evaluate adds only the
    # exact auxiliary Cartesian-COM reduced gradient.
    molecular_reduced = automatic / ambient.kT
    expected_com = jnp.broadcast_to(jnp.mean(standardized, axis=0), standardized.shape)
    np.testing.assert_allclose(
        evaluated.reduced_gradient,
        molecular_reduced + expected_com,
        rtol=5.0e-5,
        atol=5.0e-5,
    )

    # Outside the flat topology region, finite-difference/autodiff must also
    # include the derivative of the topology gate itself.
    distorted = standardized.at[0, 0].add(2.0)
    _energy, analytic = ambient.energy_and_grad(distorted)
    automatic = jax.grad(lambda value: ambient.energy(value))(distorted)
    np.testing.assert_allclose(analytic, automatic, rtol=1.0e-4, atol=2.0e-4)
    distorted_components = ambient.formal_energy_components(distorted)
    assert float(distorted_components["pmf_topology_gate"]) < 1.0


def test_additive_v3_scales_pmf_adds_ucb_and_has_autodiff_gradient() -> None:
    box_lengths = (3.70465, 3.70465, 3.70465)
    bundle = CGBGAla2Bundle(
        data_path="unused.npz",
        checkpoint_path="unused.pkl",
        box_nm=(
            (box_lengths[0], 0.0, 0.0),
            (0.0, box_lengths[1], 0.0),
            (0.0, 0.0, box_lengths[2]),
        ),
        species=(1, 3, 1, 1, 1, 3),
        mask=(True,) * 6,
        reference_nm=tuple(
            tuple(float(value) for value in row) for row in _canonical_geometry()
        ),
        standardization_std_nm=0.1,
    )

    def raw_pmf(fractional: jax.Array) -> jax.Array:
        physical = fractional * jnp.asarray(box_lengths)
        centred = physical - jnp.mean(physical, axis=0, keepdims=True)
        return -300.0 + 3.0 * jnp.sum(centred**2)

    pmf_scale = 0.25
    ambient = AmbientAla2Potential(
        pmf=CGBGAla2PMF(bundle, raw_pmf),
        physical_std_nm=0.1,
        pmf_lower_bound=SmoothEnergyWindow(-330.0, -230.0, 2.5, 2.5),
        pmf_scale=pmf_scale,
        canonical_support=_support(),
        cb_improper=BMSCBImproperRestraint(),
        # Additive-v3 deliberately has no coordinate-dependent PMF gate.
        pmf_gate_energy_scale_kj_mol=None,
    )
    # Reflection preserves every support distance/angle while selecting the
    # opposite CB improper, so this fixture isolates a non-zero U_CB term.
    reflected = _canonical_geometry().at[:, 2].multiply(-1.0)
    standardized = reflected / ambient.physical_std_nm
    components = ambient.formal_energy_components(standardized)

    assert ambient.pmf_gate_energy_scale_kj_mol is None
    assert float(components["U_cb"]) > 0.0
    np.testing.assert_allclose(
        components["U_pmf_scaled"],
        pmf_scale * components["U_pmf_effective"],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        components["U_target"],
        components["U_pmf_scaled"]
        + components["U_support"]
        + components["U_cb"],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        ambient.formal_reweight_energy(standardized),
        components["U_target"],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    _energy, analytic = ambient.energy_and_grad(standardized)
    automatic = jax.grad(lambda value: ambient.energy(value))(standardized)
    np.testing.assert_allclose(analytic, automatic, rtol=1.0e-4, atol=2.0e-4)


def test_canonical_hydra_config_pins_complete_target_identity() -> None:
    config = compose_config(
        "train_forward",
        ["experiment=ala2_ambient18_300k_bms_eta10_s1_canonical_gate2k"],
    )
    target = config.experiment.target
    assert target.mode == "pmf_canonical_support"
    assert target.implementation_abi == "ala2_canonical_support_v2"
    assert target.pmf_energy_window.minimum_kj_mol == -330.0
    assert target.pmf_energy_window.maximum_kj_mol == -230.0
    assert target.pmf_topology_gate.energy_scale_kj_mol == 25.0
    assert len(target.canonical_support.bond_indices) == 5
    assert len(target.canonical_support.angle_indices) == 5
    assert len(target.canonical_support.repulsion_indices) == 10
    # The target change must not silently alter the BMS dynamics/optimizer.
    assert config.experiment.sde.sigma_max == 1.0
    assert config.experiment.sde.steps == 50
    assert config.experiment.training.damping == 10.0
    assert config.experiment.training.terminal_score_clip_norm == 100.0


def test_additive_v3_hydra_identity_has_scale_ucb_and_no_gate() -> None:
    config = compose_config(
        "train_forward",
        ["experiment=ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3"],
    )
    target = config.experiment.target
    assert target.mode == "pmf_canonical_support"
    assert target.implementation_abi == "ala2_canonical_additive_v3"
    assert target.pmf_scale == 1.0
    assert "pmf_topology_gate" not in target
    assert target.pmf_energy_window.minimum_kj_mol == -330.0
    assert target.pmf_energy_window.maximum_kj_mol == 1000.0
    assert list(target.cb_improper.indices) == [2, 1, 4, 3]
    assert target.cb_improper.location_rad == 0.6154797086703873
    assert target.cb_improper.tolerance_rad == 0.4363323129985824
    assert target.cb_improper.force_constant_kj_mol == 2412.1333030827507
    assert len(target.canonical_support.bond_indices) == 5
    assert len(target.canonical_support.angle_indices) == 5
    assert len(target.canonical_support.repulsion_indices) == 10
    # Changing only U must leave the established BMS dynamics untouched.
    assert config.experiment.sde.sigma_max == 1.0
    assert config.experiment.sde.steps == 50
    assert config.experiment.training.damping == 10.0
    assert config.experiment.training.terminal_score_clip_norm == 100.0


def test_official_bms_cadence_additive_v3_aligns_requested_fields() -> None:
    config = compose_config(
        "train_forward",
        [
            "experiment="
            "ala2_ambient18_300k_bms_eta10_official_cadence_additive_v3"
        ],
    )
    experiment = config.experiment

    assert experiment.model.num_features == 128
    assert experiment.model.num_layers == 4
    assert experiment.model.num_radial_basis == 64
    assert experiment.model.r_max_nm == 0.8
    assert experiment.model.r_offset_nm == 0.0

    assert experiment.training.rollout_samples == 256
    assert experiment.training.buffer_capacity == 2048
    assert experiment.training.batch_size == 64
    assert experiment.training.gradient_steps == 100
    assert experiment.training.previous_model_interval_outer == 25

    assert experiment.sde.sigma_min == pytest.approx(0.0010165566603295086)
    assert experiment.sde.sigma_max == pytest.approx(6.0993399619770505)
    assert experiment.sde.rho == 3.0
    assert experiment.sde.steps == 50

    # This arm aligns selected official settings while retaining the paper's
    # damped eta=10 variant and the established CG additive target.
    assert experiment.training.damping == 10.0
    assert experiment.target.implementation_abi == "ala2_canonical_additive_v3"
