from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.evaluation.pf_validation import (
    compare_integrated_logq,
    direct_flow_map_logq,
    terminal_distribution_alignment,
)


def test_direct_flow_map_logq_for_affine_1d_flow() -> None:
    scale = 2.5
    offset = -0.7
    initial = jnp.asarray([[-2.0], [0.0], [1.5]], dtype=jnp.float64)
    initial_logq = jnp.asarray([-2.3, -0.9, -1.7], dtype=jnp.float64)

    result = direct_flow_map_logq(
        lambda state: scale * state + offset,
        initial,
        initial_logq,
    )

    np.testing.assert_allclose(result.terminal_states, scale * initial + offset)
    np.testing.assert_allclose(result.logq, initial_logq - np.log(abs(scale)))
    np.testing.assert_allclose(result.jacobians[:, 0, 0], scale)
    np.testing.assert_array_equal(result.jacobian_signs, np.ones(initial.shape[0]))


def test_direct_flow_map_logq_accepts_orientation_reversing_affine_2d_flow() -> None:
    matrix = jnp.asarray([[2.0, 0.5], [0.0, -3.0]], dtype=jnp.float64)
    offset = jnp.asarray([0.25, -1.0], dtype=jnp.float64)
    initial = jnp.asarray(
        [[-1.0, 0.2], [0.0, 1.0], [1.5, -0.5]],
        dtype=jnp.float64,
    )
    initial_logq = jnp.asarray([-1.0, -2.0, -3.0], dtype=jnp.float64)

    result = direct_flow_map_logq(
        lambda state: matrix @ state + offset,
        initial,
        initial_logq,
    )

    expected_terminal = initial @ np.asarray(matrix).T + np.asarray(offset)
    np.testing.assert_allclose(result.terminal_states, expected_terminal)
    np.testing.assert_allclose(result.log_abs_det_jacobians, np.log(6.0))
    np.testing.assert_allclose(result.logq, initial_logq - np.log(6.0))
    np.testing.assert_array_equal(result.jacobian_signs, -np.ones(initial.shape[0]))


def test_direct_flow_map_logq_rejects_non_square_and_singular_maps() -> None:
    initial = jnp.ones((2, 2), dtype=jnp.float64)
    initial_logq = jnp.zeros((2,), dtype=jnp.float64)

    with pytest.raises(ValueError, match="same event shape"):
        direct_flow_map_logq(lambda state: state[:1], initial, initial_logq)

    singular = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)
    with pytest.raises(ValueError, match="singular"):
        direct_flow_map_logq(lambda state: singular @ state, initial, initial_logq)


def test_compare_integrated_logq_reports_finite_and_error_summaries() -> None:
    direct = np.asarray([1.0, 2.0, 3.0, 4.0])
    integrated = np.asarray([1.1, np.nan, 2.9, 4.0])

    summary = compare_integrated_logq(integrated, direct, atol=0.11, rtol=0.0)

    assert summary["num_samples"] == 4
    assert summary["finite_pair_count"] == 3
    assert summary["finite_pair_fraction"] == 0.75
    assert not summary["all_finite"]
    np.testing.assert_allclose(summary["mean_error"], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(summary["rmse"], np.sqrt(0.02 / 3.0))
    assert summary["within_tolerance_count"] == 3
    assert not summary["all_within_tolerance"]


def test_terminal_distribution_alignment_for_affine_1d_and_2d_clouds() -> None:
    base = np.linspace(-2.0, 2.0, 256)
    sde_1d = 1.7 * base - 0.4
    energy = np.square(sde_1d)
    one_dimensional = terminal_distribution_alignment(
        sde_1d,
        sde_1d.copy(),
        bins=32,
        sde_energy=energy,
        pf_energy=energy.copy(),
        energy_bins=32,
    )

    assert one_dimensional["dimension"] == 1
    np.testing.assert_allclose(one_dimensional["mean_l2_error"], 0.0)
    np.testing.assert_allclose(one_dimensional["covariance_frobenius_error"], 0.0)
    np.testing.assert_allclose(one_dimensional["histogram_js"], 0.0, atol=1.0e-15)
    assert one_dimensional["energy"] is not None
    np.testing.assert_allclose(
        one_dimensional["energy"]["histogram_js"],
        0.0,
        atol=1.0e-15,
    )

    source_2d = np.stack((base, np.sin(base)), axis=1)
    matrix = np.asarray([[1.2, 0.3], [-0.4, 0.8]])
    sde_2d = source_2d @ matrix.T + np.asarray([0.2, -0.5])
    shift = np.asarray([0.15, -0.1])
    pf_2d = sde_2d + shift
    two_dimensional = terminal_distribution_alignment(sde_2d, pf_2d, bins=24)

    assert two_dimensional["dimension"] == 2
    np.testing.assert_allclose(two_dimensional["mean_difference"], shift)
    np.testing.assert_allclose(two_dimensional["covariance_difference"], 0.0, atol=1.0e-14)
    assert two_dimensional["histogram_js"] > 0.0
