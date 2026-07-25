from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.model.painn import (
    PaiNN,
    RadialBasis,
    TimeEmbedding,
    TorchLinear,
)

jax.config.update("jax_enable_x64", True)


def _tiny_painn(*, parity_breaking: bool = True) -> PaiNN:
    return PaiNN(
        num_features=8,
        num_radial_basis=4,
        num_layers=2,
        num_elements=6,
        r_max=0.8,
        r_offset=0.0,
        time_init_mode="node",
        parity_breaking=parity_breaking,
        conservative=False,
        unique_atom_indices=True,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )


def _positions() -> jax.Array:
    return jnp.asarray(
        [
            [
                [-0.20, 0.01, 0.03],
                [-0.08, 0.12, -0.02],
                [0.02, 0.02, 0.09],
                [0.06, -0.11, 0.04],
                [0.15, 0.04, -0.08],
                [0.24, -0.05, 0.01],
            ],
            [
                [-0.19, -0.03, 0.06],
                [-0.10, 0.10, 0.01],
                [0.01, 0.04, 0.11],
                [0.07, -0.09, 0.02],
                [0.16, 0.01, -0.07],
                [0.23, -0.03, -0.01],
            ],
        ],
        dtype=jnp.float64,
    )


def _init(model: PaiNN, positions: jax.Array):
    return model.init(
        {
            "params": jax.random.PRNGKey(1),
            "constants": jax.random.PRNGKey(2),
        },
        jnp.asarray([0.2, 0.7], dtype=positions.dtype),
        positions,
    )


def _proper_rotation(dtype=jnp.float64) -> jax.Array:
    axis = jnp.asarray([1.0, -2.0, 0.5], dtype=dtype)
    axis = axis / jnp.linalg.norm(axis)
    angle = jnp.asarray(0.73, dtype=dtype)
    x, y, z = axis
    skew = jnp.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=dtype
    )
    eye = jnp.eye(3, dtype=dtype)
    return eye + jnp.sin(angle) * skew + (1.0 - jnp.cos(angle)) * (skew @ skew)


def test_torch_linear_uses_symmetric_torch_default_bounds():
    layer = TorchLinear(17, dtype=jnp.float64, param_dtype=jnp.float64)
    inputs = jnp.ones((3, 11), dtype=jnp.float64)
    variables = layer.init(jax.random.PRNGKey(0), inputs)
    params = variables["params"]
    bound = 1.0 / math.sqrt(11.0)
    assert params["kernel"].shape == (11, 17)
    assert params["bias"].shape == (17,)
    assert bool(jnp.all(jnp.abs(params["kernel"]) <= bound))
    assert bool(jnp.all(jnp.abs(params["bias"]) <= bound))
    # Unlike the Flax default, the Torch bias initializer is not identically zero.
    assert bool(jnp.any(params["bias"] != 0.0))


def test_time_embedding_is_checkpointed_as_non_trainable_constant():
    module = TimeEmbedding(8, dtype=jnp.float64, param_dtype=jnp.float64)
    values = jnp.asarray([0.0, 0.25], dtype=jnp.float64)
    variables = module.init(
        {"params": jax.random.PRNGKey(0), "constants": jax.random.PRNGKey(3)},
        values,
    )
    assert "constants" in variables
    assert "freqs" in variables["constants"]
    assert "params" not in variables or not variables["params"]
    frequencies = variables["constants"]["freqs"]
    expected = math.sqrt(2.0) * jnp.concatenate(
        (
            jnp.sin(values[..., None] * frequencies),
            jnp.cos(values[..., None] * frequencies),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(module.apply(variables, values), expected, atol=1e-12)


def test_radial_basis_matches_bms_gaussian_and_polynomial_formula():
    module = RadialBasis(num_radial=4, cutoff=0.8, envelope_exponent=5)
    distance = jnp.asarray([0.0, 0.2, 0.8, 1.0], dtype=jnp.float64)
    variables = module.init(jax.random.PRNGKey(0), distance)
    radial, envelope = module.apply(variables, distance)
    scaled = np.asarray(distance / 0.8)
    centers = np.linspace(0.0, 1.0, 4)
    coefficient = -0.5 / (centers[1] - centers[0]) ** 2
    expected_radial = np.exp(coefficient * (scaled[:, None] - centers) ** 2)
    expected_envelope = np.where(
        scaled < 1.0,
        1.0 - 21.0 * scaled**5 + 35.0 * scaled**6 - 15.0 * scaled**7,
        0.0,
    )
    np.testing.assert_allclose(radial, expected_radial, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(envelope, expected_envelope, rtol=1e-12, atol=1e-12)


def test_painn_output_shape_is_finite_and_jittable():
    positions = _positions()
    times = jnp.asarray([0.2, 0.7], dtype=positions.dtype)
    model = _tiny_painn()
    variables = _init(model, positions)
    output = jax.jit(model.apply)(variables, times, positions)
    assert output.shape == positions.shape
    assert bool(jnp.all(jnp.isfinite(output)))


def test_painn_broadcasts_scalar_time_for_probability_flow_batch():
    positions = _positions()[:1]
    model = _tiny_painn()
    variables = model.init(
        {
            "params": jax.random.PRNGKey(12),
            "constants": jax.random.PRNGKey(13),
        },
        0.35,
        positions,
    )
    output = model.apply(variables, 0.35, positions)
    assert output.shape == positions.shape
    assert bool(jnp.all(jnp.isfinite(output)))


def test_painn_is_translation_invariant():
    positions = _positions()
    times = jnp.asarray([0.2, 0.7], dtype=positions.dtype)
    model = _tiny_painn()
    variables = _init(model, positions)
    shift = jnp.asarray([0.91, -0.43, 1.27], dtype=positions.dtype)
    original = model.apply(variables, times, positions)
    translated = model.apply(variables, times, positions + shift)
    np.testing.assert_allclose(translated, original, rtol=1e-10, atol=1e-10)


def test_chiral_painn_is_equivariant_under_proper_rotations():
    positions = _positions()
    times = jnp.asarray([0.2, 0.7], dtype=positions.dtype)
    model = _tiny_painn(parity_breaking=True)
    variables = _init(model, positions)
    rotation = _proper_rotation(positions.dtype)
    rotated_positions = jnp.einsum("ij,bnj->bni", rotation, positions)
    output = model.apply(variables, times, positions)
    rotated_output = model.apply(variables, times, rotated_positions)
    expected = jnp.einsum("ij,bnj->bni", rotation, output)
    np.testing.assert_allclose(rotated_output, expected, rtol=2e-9, atol=2e-9)


def test_coincident_beads_remain_finite_due_to_bms_epsilons():
    positions = jnp.zeros((2, 6, 3), dtype=jnp.float64)
    times = jnp.asarray([0.1, 0.9], dtype=positions.dtype)
    model = _tiny_painn()
    variables = _init(model, positions)
    output = model.apply(variables, times, positions)
    assert output.shape == positions.shape
    assert bool(jnp.all(jnp.isfinite(output)))
