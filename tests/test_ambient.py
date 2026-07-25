from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.coordinates import (
    AmbientTransform,
    RelativeDomainSpec,
    RelativeSoftWall,
    cgbg_com_log_radial_density,
    cgbg_correct_log_density,
    com_aux_reduced_energy,
    com_aux_reduced_gradient,
    com_aux_score,
    combine_orthogonal_shape_com,
    compute_centered_std,
    domain_support_mask,
    helmert_basis,
    split_orthogonal_shape_com,
    split_shape_com,
    standard_com_log_prob,
)
from cg_bms_jax.data import GaussianSource
from cg_bms_jax.potential import AmbientAla2Potential, CGBGAla2Bundle, CGBGAla2PMF


def test_shape_com_round_trip_and_standardization() -> None:
    coordinates = jnp.asarray(
        [
            [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]],
            [[-2.0, 0.0, 1.0], [0.0, 2.0, 3.0], [2.0, 4.0, 5.0]],
        ]
    )
    shape, centre = split_shape_com(coordinates)
    np.testing.assert_allclose(shape + centre, coordinates)
    np.testing.assert_allclose(jnp.mean(shape, axis=-2), 0.0, atol=1e-7)
    np.testing.assert_allclose(compute_centered_std(coordinates), jnp.std(shape))

    transform = AmbientTransform(n_beads=3, physical_std=float(jnp.std(shape)))
    standardized = transform.centre_and_standardize(coordinates)
    np.testing.assert_allclose(jnp.std(standardized), 1.0, atol=1e-6)
    lifted = transform.add_com_noise(standardized, jax.random.PRNGKey(1))
    np.testing.assert_allclose(
        lifted - jnp.mean(lifted, axis=-2, keepdims=True),
        standardized,
        atol=1e-6,
    )


def test_auxiliary_com_energy_gradient_is_exact() -> None:
    x = jnp.arange(18, dtype=jnp.float64).reshape(6, 3) / 10.0
    automatic = jax.grad(com_aux_reduced_energy)(x)
    analytic = com_aux_reduced_gradient(x)
    np.testing.assert_allclose(analytic, automatic, rtol=1e-10, atol=1e-10)
    # With CG-BG's sigma=1/sqrt(N), every bead receives the COM gradient.
    np.testing.assert_allclose(analytic, jnp.broadcast_to(jnp.mean(x, axis=0), x.shape))


def test_all_atom_ambient_is_an_exact_63_plus_3_orthogonal_transform() -> None:
    n_atoms = 22
    basis = helmert_basis(n_atoms, dtype=jnp.float64)
    np.testing.assert_allclose(
        basis.T @ basis,
        jnp.eye(n_atoms - 1, dtype=basis.dtype),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        basis.T @ jnp.ones((n_atoms,), dtype=basis.dtype),
        0.0,
        atol=1e-12,
    )
    orthogonal = jnp.concatenate(
        (
            basis,
            jnp.ones((n_atoms, 1), dtype=basis.dtype) / jnp.sqrt(n_atoms),
        ),
        axis=1,
    )
    np.testing.assert_allclose(
        jnp.abs(jnp.linalg.det(orthogonal)),
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )

    ambient = jax.random.normal(
        jax.random.PRNGKey(21),
        (3, n_atoms, 3),
        dtype=jnp.float64,
    )
    shape, normalized_com = split_orthogonal_shape_com(ambient)
    assert shape.shape == (3, 21, 3)
    assert normalized_com.shape == (3, 3)
    reconstructed = combine_orthogonal_shape_com(shape, normalized_com)
    np.testing.assert_allclose(reconstructed, ambient, rtol=1e-12, atol=1e-12)


def test_all_atom_standard_source_density_decomposes_as_63_plus_3() -> None:
    transform = AmbientTransform(n_beads=22, physical_std=1.0)
    assert transform.ambient_dim == 66
    assert transform.shape_dim == 63
    assert transform.com_dim == 3
    assert transform.source_event_shape == (22, 3)

    source = GaussianSource(event_shape=transform.source_event_shape)
    samples = source.sample(jax.random.PRNGKey(22), 5, dtype=jnp.float64)
    components = transform.source_density(samples)
    np.testing.assert_allclose(
        components.total,
        source.log_prob(samples),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        components.total,
        components.shape + components.com,
        rtol=0.0,
        atol=0.0,
    )

    shape, normalized_com = transform.split_density_coordinates(samples)
    np.testing.assert_allclose(
        transform.combine_density_coordinates(shape, normalized_com),
        samples,
        rtol=1e-12,
        atol=1e-12,
    )


def test_auxiliary_com_is_standard_normal_in_orthogonal_coordinates() -> None:
    x = jax.random.normal(jax.random.PRNGKey(23), (4, 22, 3), dtype=jnp.float64)
    normalized_com = split_orthogonal_shape_com(x)[1]
    expected = -0.5 * (
        jnp.sum(normalized_com**2, axis=-1) + 3.0 * jnp.log(2.0 * jnp.pi)
    )
    np.testing.assert_allclose(standard_com_log_prob(x), expected, atol=1e-12)
    np.testing.assert_allclose(
        jax.grad(standard_com_log_prob)(x[0]),
        com_aux_score(x[0]),
        rtol=1e-12,
        atol=1e-12,
    )

    arbitrary_shape_logq = jnp.asarray([-1.0, -2.0, -3.0, -4.0])
    density = AmbientTransform(n_beads=22, physical_std=1.0).exact_density(
        arbitrary_shape_logq, x
    )
    np.testing.assert_allclose(density.shape, arbitrary_shape_logq)
    np.testing.assert_allclose(density.com, expected)
    np.testing.assert_allclose(density.total, arbitrary_shape_logq + expected)


def test_relative_domain_and_flat_bottom_wall() -> None:
    domain = RelativeDomainSpec(box=(3.0, 3.0, 3.0), anchor=0, margin=0.2)
    wall = RelativeSoftWall(domain=domain, strength=5.0)
    inside = jnp.zeros((6, 3))
    margin_region = inside.at[1, 0].set(1.4)
    outside = inside.at[1, 0].set(1.6)

    assert bool(domain.support_mask(inside))
    assert bool(domain.support_mask(margin_region))
    assert not bool(domain.support_mask(outside))
    assert float(wall.reduced_energy(inside)) == 0.0
    assert float(wall.reduced_energy(margin_region)) > 0.0

    gradient = wall.reduced_gradient(margin_region)
    np.testing.assert_allclose(jnp.sum(gradient, axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(gradient, jax.grad(wall.reduced_energy)(margin_region))
    assert domain.metadata() == {
        "box": (3.0, 3.0, 3.0),
        "anchor": 0,
        "margin": 0.2,
        "support_mode": "relative_fundamental",
    }

    # This path must remain JIT compatible (no Python bool conversion of a tracer).
    support_jit = jax.jit(
        lambda value: domain_support_mask(
            value,
            box=jnp.asarray(domain.box),
            mode="relative_fundamental",
            anchor=domain.anchor,
        )
    )
    assert bool(support_jit(inside))


def test_cgbg_compat_density_correction_formula() -> None:
    x = jnp.zeros((2, 6, 3)).at[:, :, 0].set(0.25)
    logq = jnp.asarray([-4.0, -5.0])
    std = 0.2
    radial = cgbg_com_log_radial_density(x)
    corrected = cgbg_correct_log_density(logq, x, std)
    np.testing.assert_allclose(corrected, logq - radial - 18.0 * jnp.log(std))


def test_ala2_training_wall_is_excluded_from_formal_target() -> None:
    box = ((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0))
    bundle = CGBGAla2Bundle(
        data_path="unused.npz",
        checkpoint_path="unused.pkl",
        box_nm=box,
        species=(0, 0, 0, 0, 0, 0),
        mask=(True,) * 6,
        reference_nm=((0.0, 0.0, 0.0),) * 6,
        standardization_std_nm=1.0,
    )

    def fractional_energy(frac: jax.Array) -> jax.Array:
        relative = frac[1:] - frac[:1]
        return jnp.sum(relative * relative)

    pmf = CGBGAla2PMF(bundle, fractional_energy)
    domain = RelativeDomainSpec(box=(3.0, 3.0, 3.0), margin=0.2)
    ambient = AmbientAla2Potential(
        pmf=pmf,
        physical_std_nm=1.0,
        training_wall=RelativeSoftWall(domain, strength=4.0),
    )
    x = jnp.zeros((6, 3)).at[1, 0].set(1.4)
    with_wall = ambient.evaluate(x, include_training_wall=True)
    without_wall = ambient.evaluate(x, include_training_wall=False)
    assert float(with_wall.reduced_energy) > float(without_wall.reduced_energy)
    np.testing.assert_allclose(ambient.formal_reweight_energy(x), pmf.energy(x))
