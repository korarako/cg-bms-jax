from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.data import GaussianSource
from cg_bms_jax.potential import ScoreDecomposition
from cg_bms_jax.process import EDMSDE
from cg_bms_jax.training.forward import (
    ForwardTrainingConfig,
    attach_forward_target,
    clip_target_score_by_bead,
    train_forward_fixed_point,
)


def test_clip_target_score_is_per_bead_and_preserves_direction() -> None:
    score = jnp.asarray(
        [
            [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 1.0],
                [-6.0, 0.0, 8.0],
            ]
        ],
        dtype=jnp.float32,
    )
    clipped = clip_target_score_by_bead(score, 2.0)

    np.testing.assert_allclose(clipped[0, 0], [1.2, 1.6, 0.0], rtol=1.0e-6)
    np.testing.assert_allclose(clipped[0, 1], score[0, 1], rtol=1.0e-6)
    np.testing.assert_allclose(clipped[0, 2], [-1.2, 0.0, 1.6], rtol=1.0e-6)
    norms = np.linalg.norm(np.asarray(clipped), axis=-1)
    assert np.all(norms <= 2.0 + 1.0e-6)
    cosine = np.sum(np.asarray(score * clipped), axis=-1) / (
        np.linalg.norm(np.asarray(score), axis=-1)
        * np.linalg.norm(np.asarray(clipped), axis=-1)
    )
    np.testing.assert_allclose(cosine, 1.0, atol=1.0e-6)


def test_terminal_score_clip_is_jittable_and_enforces_threshold() -> None:
    compiled = jax.jit(lambda value: clip_target_score_by_bead(value, 5.0))
    score = jnp.asarray([[[6.0, 8.0, 0.0], [0.0, -3.0, 4.0]]])
    clipped = compiled(score)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(clipped), axis=-1), 5.0, atol=1.0e-6)

    endpoint = jnp.zeros_like(score)
    attach_compiled = jax.jit(
        lambda value: attach_forward_target(
            endpoint,
            endpoint,
            lambda _: value,
            terminal_score_clip_norm=5.0,
        )["target_score"]
    )
    attached = attach_compiled(score)
    np.testing.assert_allclose(np.asarray(attached), np.asarray(clipped), atol=1.0e-6)


def test_attach_forward_target_clips_before_replay_and_default_is_disabled() -> None:
    endpoint = jnp.zeros((2, 2, 3), dtype=jnp.float32)
    raw_score = jnp.asarray(
        [
            [[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 10.0], [1.0, 2.0, 2.0]],
        ],
        dtype=jnp.float32,
    )
    disabled = attach_forward_target(endpoint, endpoint, lambda _: raw_score)
    np.testing.assert_array_equal(disabled["target_score"], raw_score)

    enabled = attach_forward_target(
        endpoint,
        endpoint,
        lambda _: raw_score,
        terminal_score_clip_norm=2.0,
    )
    replay_norms = np.linalg.norm(np.asarray(enabled["target_score"]), axis=-1)
    assert np.all(replay_norms <= 2.0 + 1.0e-6)
    assert np.all(np.asarray(enabled["valid"]))


def test_terminal_score_clip_rejects_non_molecular_shape_and_invalid_limit() -> None:
    with pytest.raises(ValueError, match="batch, beads, 3"):
        clip_target_score_by_bead(jnp.ones((4, 2)), 1.0)
    with pytest.raises(ValueError, match="positive"):
        ForwardTrainingConfig(terminal_score_clip_norm=0.0)
    with pytest.raises(ValueError, match="terminal_score_clip_mean_mode"):
        ForwardTrainingConfig(terminal_score_clip_mean_mode="unknown")
    with pytest.raises(ValueError, match="only applies"):
        ForwardTrainingConfig(
            terminal_score_clip_mean_mode="analytic_standard_normal"
        )


def test_ambient_clip_preserves_com_score_and_reprojects_shape() -> None:
    endpoint = jnp.zeros((1, 3, 3), dtype=jnp.float32)
    shape = jnp.asarray(
        [[[20.0, 0.0, 0.0], [-10.0, 3.0, 0.0], [-10.0, -3.0, 0.0]]]
    )
    com = jnp.asarray([[[1.5, -2.0, 0.5]]])
    raw_score = shape + com
    attached = attach_forward_target(
        endpoint,
        endpoint,
        lambda _: raw_score,
        terminal_score_clip_norm=5.0,
        terminal_score_clip_preserve_mean=True,
    )["target_score"]

    np.testing.assert_allclose(
        np.mean(np.asarray(attached), axis=1, keepdims=True),
        np.asarray(com),
        atol=1.0e-6,
    )
    clipped_shape = np.asarray(attached - com)
    np.testing.assert_allclose(np.mean(clipped_shape, axis=1), 0.0, atol=1.0e-6)


def test_ambient_clip_uses_analytic_standard_normal_com_score() -> None:
    endpoint = jnp.asarray(
        [
            [
                [2.0, -1.0, 0.5],
                [4.0, 1.0, -0.5],
                [0.0, -3.0, 1.5],
            ]
        ],
        dtype=jnp.float32,
    )
    analytic_com_score = -jnp.mean(endpoint, axis=1, keepdims=True)
    shape = jnp.asarray(
        [[[20.0, 0.0, 0.0], [-10.0, 3.0, 0.0], [-10.0, -3.0, 0.0]]],
        dtype=jnp.float32,
    )
    # Simulate a large translation-like residual left by cancellation of
    # off-manifold molecular forces.  It must not be mistaken for COM score.
    numerical_residual = jnp.asarray([[[1.0e7, -2.0e7, 3.0e7]]])
    attached = attach_forward_target(
        endpoint,
        endpoint,
        lambda _: ScoreDecomposition(
            clippable=shape + numerical_residual,
            preserved=jnp.broadcast_to(analytic_com_score, endpoint.shape),
        ),
        terminal_score_clip_norm=5.0,
        terminal_score_clip_preserve_mean=True,
        terminal_score_clip_mean_mode="analytic_standard_normal",
    )["target_score"]

    attached_np = np.asarray(attached)
    analytic_np = np.asarray(analytic_com_score)
    assert np.isfinite(attached_np).all()
    np.testing.assert_allclose(
        np.mean(attached_np, axis=1, keepdims=True),
        analytic_np,
        atol=1.0e-5,
    )
    # Recentring after per-particle clipping can at most double the clip bound.
    shape_norm = np.linalg.norm(attached_np - analytic_np, axis=-1)
    assert np.max(shape_norm) <= 10.0 + 1.0e-5


def test_ambient_clip_stays_finite_for_large_finite_openmm_scores() -> None:
    endpoint = jnp.linspace(-1.0, 1.0, 66, dtype=jnp.float32).reshape(1, 22, 3)
    shape_score = jnp.tile(
        jnp.asarray([1.0e25, -2.0e25, 1.5e25], dtype=jnp.float32),
        (1, 22, 1),
    )
    shape_score = shape_score.at[:, 1::2].multiply(-1.0)
    com_score = -jnp.mean(endpoint, axis=-2, keepdims=True)
    attached = attach_forward_target(
        endpoint,
        endpoint,
        lambda _value: ScoreDecomposition(
            clippable=shape_score,
            preserved=jnp.broadcast_to(com_score, endpoint.shape),
        ),
        terminal_score_clip_norm=100.0,
        terminal_score_clip_preserve_mean=True,
        terminal_score_clip_mean_mode="analytic_standard_normal",
    )["target_score"]

    attached_np = np.asarray(attached)
    com_np = np.asarray(com_score)
    assert np.isfinite(attached_np).all()
    np.testing.assert_allclose(
        np.mean(attached_np, axis=1, keepdims=True),
        com_np,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    clipped_shape = attached_np - com_np
    assert np.max(np.linalg.norm(clipped_shape, axis=-1)) <= 200.0 + 1.0e-3


def test_invalid_terminal_score_is_sanitized_before_clipping() -> None:
    endpoint = jnp.zeros((1, 22, 3), dtype=jnp.float32)
    invalid_score = jnp.full_like(endpoint, jnp.inf)

    pairs = attach_forward_target(
        endpoint,
        endpoint,
        lambda _value: ScoreDecomposition(
            clippable=invalid_score,
            preserved=jnp.zeros_like(endpoint),
        ),
        terminal_score_clip_norm=100.0,
        terminal_score_clip_preserve_mean=True,
        terminal_score_clip_mean_mode="analytic_standard_normal",
    )

    assert not bool(pairs["valid"][0])
    assert np.isfinite(np.asarray(pairs["target_score"])).all()
    np.testing.assert_array_equal(np.asarray(pairs["target_score"]), 0.0)


def test_forward_loop_stores_clipped_targets_and_records_raw_diagnostics() -> None:
    source = GaussianSource(event_shape=(2, 3), scale=1.0)
    sde = EDMSDE(sigma_min=0.05, sigma_max=0.2, rho=3.0)

    def control_apply(params, constants, time, state):
        del constants, time
        return params["gain"] * state

    result = train_forward_fixed_point(
        initial_params={"gain": jnp.asarray(0.0, dtype=jnp.float32)},
        constants={},
        control_apply=control_apply,
        source=source,
        sde=sde,
        target_score_fn=lambda value: jnp.full_like(value, 10.0),
        config=ForwardTrainingConfig(
            outer_iterations=1,
            rollout_batches_per_outer=1,
            rollout_batch_size=4,
            rollout_steps=1,
            updates_per_outer=1,
            minibatch_size=4,
            replay_capacity=4,
            terminal_score_clip_norm=5.0,
        ),
    )

    size = int(result.replay_buffer.size)
    replay_score = np.asarray(result.replay_buffer.storage["target_score"][:size])
    assert np.max(np.linalg.norm(replay_score, axis=-1)) <= 5.0 + 1.0e-6
    record = result.metrics[0]
    np.testing.assert_allclose(record["target_score_raw_norm_max"], np.sqrt(300.0), rtol=1.0e-6)
    np.testing.assert_allclose(record["target_score_clipped_norm_max"], 5.0, atol=1.0e-6)
    assert record["target_score_clip_fraction"] == 1.0
