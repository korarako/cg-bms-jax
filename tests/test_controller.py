from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.model.controller import (
    MLPController,
    RadialCOMHead,
    ShapeCOMController,
)
from cg_bms_jax.model.painn import PaiNN
from cg_bms_jax.process import exact_divergence

jax.config.update("jax_enable_x64", True)


class _ShapeField(nn.Module):
    @nn.compact
    def __call__(self, time, positions):
        del time
        # Include a common vector to verify that the wrapper projects it away.
        common = jnp.asarray([1.0, -2.0, 0.5], dtype=positions.dtype)
        return 2.0 * positions + common


class _COMField(nn.Module):
    @nn.compact
    def __call__(self, time, center):
        del time
        return -center


def _rotation(dtype=jnp.float64):
    angle = jnp.asarray(0.41, dtype=dtype)
    return jnp.asarray(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )


def test_mlp_controller_preserves_low_dimensional_state_shape():
    model = MLPController(
        hidden_features=(16, 16),
        time_embedding_dim=8,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )
    state = jnp.asarray([[-1.0], [0.2], [1.7]], dtype=jnp.float64)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "constants": jax.random.PRNGKey(1)},
        0.3,
        state,
    )
    output = model.apply(variables, 0.3, state)
    assert output.shape == state.shape
    assert bool(jnp.all(jnp.isfinite(output)))


def test_radial_com_head_is_rotation_equivariant_and_zero_at_origin():
    model = RadialCOMHead(
        hidden_features=(16, 16),
        time_embedding_dim=8,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )
    center = jnp.asarray(
        [[0.2, -0.1, 0.3], [-0.4, 0.05, 0.1]], dtype=jnp.float64
    )
    time = jnp.asarray([0.2, 0.8], dtype=center.dtype)
    variables = model.init(
        {"params": jax.random.PRNGKey(2), "constants": jax.random.PRNGKey(3)},
        time,
        center,
    )
    rotation = _rotation(center.dtype)
    rotated_center = jnp.einsum("ij,bj->bi", rotation, center)
    output = model.apply(variables, time, center)
    rotated_output = model.apply(variables, time, rotated_center)
    expected = jnp.einsum("ij,bj->bi", rotation, output)
    np.testing.assert_allclose(rotated_output, expected, rtol=1e-10, atol=1e-10)
    zero_output = model.apply(variables, time, jnp.zeros_like(center))
    np.testing.assert_allclose(zero_output, 0.0, atol=0.0)


def test_shape_com_wrapper_keeps_branches_orthogonal_and_translation_sensitive_only_in_com():
    model = ShapeCOMController(shape_model=_ShapeField(), com_head=_COMField())
    positions = jnp.asarray(
        [
            [
                [-0.2, 0.0, 0.1],
                [-0.1, 0.1, 0.0],
                [0.0, 0.0, -0.1],
                [0.1, -0.1, 0.0],
                [0.2, 0.0, 0.1],
                [0.3, 0.1, -0.1],
            ]
        ],
        dtype=jnp.float64,
    )
    variables = model.init(
        jax.random.PRNGKey(0), 0.4, positions, return_components=True
    )
    control, parts = model.apply(
        variables, 0.4, positions, return_components=True
    )
    np.testing.assert_allclose(parts["shape"].mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(control.mean(axis=1), parts["com"], atol=1e-12)

    shift = jnp.asarray([0.7, -0.2, 1.1], dtype=positions.dtype)
    shifted_control, shifted_parts = model.apply(
        variables, 0.4, positions + shift, return_components=True
    )
    np.testing.assert_allclose(shifted_parts["shape"], parts["shape"], atol=1e-12)
    np.testing.assert_allclose(shifted_parts["com"], parts["com"] - shift, atol=1e-12)
    expected_shift = jnp.broadcast_to(-shift, shifted_control.shape)
    np.testing.assert_allclose(shifted_control - control, expected_shift, atol=1e-12)


def test_shape_com_position_scale_only_changes_shape_geometry():
    positions = jnp.asarray(
        [[[-0.2, 0.1, 0.0], [0.0, -0.1, 0.2], [0.2, 0.0, -0.2]]],
        dtype=jnp.float64,
    )
    unscaled = ShapeCOMController(
        shape_model=_ShapeField(), com_head=_COMField(), position_scale=1.0
    )
    scaled = ShapeCOMController(
        shape_model=_ShapeField(), com_head=_COMField(), position_scale=2.5
    )
    variables_unscaled = unscaled.init(
        jax.random.PRNGKey(10), 0.5, positions, return_components=True
    )
    variables_scaled = scaled.init(
        jax.random.PRNGKey(11), 0.5, positions, return_components=True
    )
    _, parts_unscaled = unscaled.apply(
        variables_unscaled, 0.5, positions, return_components=True
    )
    _, parts_scaled = scaled.apply(
        variables_scaled, 0.5, positions, return_components=True
    )
    np.testing.assert_allclose(parts_scaled["center"], parts_unscaled["center"], atol=0.0)
    np.testing.assert_allclose(parts_scaled["com"], parts_unscaled["com"], atol=0.0)
    np.testing.assert_allclose(
        parts_scaled["shape_positions"], 2.5 * parts_unscaled["shape_positions"], atol=1e-12
    )
    np.testing.assert_allclose(parts_scaled["shape"], 2.5 * parts_unscaled["shape"], atol=1e-12)


def test_shape_com_controller_with_faithful_painn_is_finite():
    painn = PaiNN(
        num_features=8,
        num_radial_basis=4,
        num_layers=2,
        num_elements=6,
        r_max=0.8,
        time_init_mode="node",
        parity_breaking=True,
        unique_atom_indices=True,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )
    com = RadialCOMHead(
        hidden_features=(8,),
        time_embedding_dim=8,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )
    model = ShapeCOMController(shape_model=painn, com_head=com)
    positions = jax.random.normal(
        jax.random.PRNGKey(7), (2, 6, 3), dtype=jnp.float64
    ) * 0.1
    variables = model.init(
        {"params": jax.random.PRNGKey(8), "constants": jax.random.PRNGKey(9)},
        jnp.asarray([0.2, 0.8], dtype=positions.dtype),
        positions,
    )
    output = model.apply(
        variables,
        jnp.asarray([0.2, 0.8], dtype=positions.dtype),
        positions,
    )
    assert output.shape == positions.shape
    assert bool(jnp.all(jnp.isfinite(output)))


def test_all_atom_orthogonal_com_controller_is_full_rank_and_pf_compatible():
    model = ShapeCOMController(
        shape_model=_ShapeField(),
        com_head=_COMField(),
        com_coordinates="orthogonal",
        num_particles=22,
    )
    positions = jax.random.normal(
        jax.random.PRNGKey(31),
        (1, 22, 3),
        dtype=jnp.float64,
    )
    variables = model.init(
        jax.random.PRNGKey(32),
        0.4,
        positions,
        return_components=True,
    )
    control, parts = model.apply(
        variables,
        0.4,
        positions,
        return_components=True,
    )
    center = jnp.mean(positions, axis=1)
    np.testing.assert_allclose(parts["centered"].mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        parts["shape_positions"].mean(axis=1), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(parts["shape"].mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        parts["orthogonal_com"],
        jnp.sqrt(jnp.asarray(22.0)) * center,
        rtol=1e-12,
        atol=1e-12,
    )
    # _COMField is the exact score of the standard N(0,I_3) orthogonal COM.
    np.testing.assert_allclose(parts["orthogonal_com_control"], -parts["orthogonal_com"])
    np.testing.assert_allclose(parts["com"], -center, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(control.mean(axis=1), -center, rtol=1e-12, atol=1e-12)

    # The existing PF divergence operates directly on the unchanged (B,22,3)
    # ambient state.  This linear test has trace 2*63 - 3 = 123.
    divergence = exact_divergence(
        lambda time, value: model.apply(variables, time, value),
        0.4,
        positions,
    )
    np.testing.assert_allclose(divergence, 123.0, rtol=1e-12, atol=1e-12)


def test_all_atom_shape_branch_accepts_centered_22_atom_painn_geometry():
    painn = PaiNN(
        num_features=8,
        num_radial_basis=4,
        num_layers=1,
        num_elements=22,
        r_max=8.0,
        time_init_mode="node",
        parity_breaking=True,
        unique_atom_indices=True,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )
    com = RadialCOMHead(
        hidden_features=(8,),
        time_embedding_dim=8,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )
    model = ShapeCOMController(
        shape_model=painn,
        com_head=com,
        com_coordinates="orthogonal",
        num_particles=22,
    )
    positions = jax.random.normal(
        jax.random.PRNGKey(33),
        (2, 22, 3),
        dtype=jnp.float64,
    )
    variables = model.init(
        {"params": jax.random.PRNGKey(34), "constants": jax.random.PRNGKey(35)},
        jnp.asarray([0.2, 0.8], dtype=positions.dtype),
        positions,
        return_components=True,
    )
    output, parts = model.apply(
        variables,
        jnp.asarray([0.2, 0.8], dtype=positions.dtype),
        positions,
        return_components=True,
    )
    assert output.shape == positions.shape
    assert bool(jnp.all(jnp.isfinite(output)))
    np.testing.assert_allclose(parts["shape_positions"].mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(parts["shape"].mean(axis=1), 0.0, atol=1e-12)
