from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.data import GaussianSource
from cg_bms_jax.process import EDMSDE
from cg_bms_jax.training import (
    BackwardTrainingConfig,
    ForwardTrainingConfig,
    flax_apply,
    split_flax_variables,
    train_backward_score,
    train_forward_fixed_point,
)


class TinyController(nn.Module):
    hidden: int = 8

    @nn.compact
    def __call__(self, time, state):
        state = jnp.asarray(state)
        batch = state.shape[0]
        flat = state.reshape(batch, -1)
        time = jnp.asarray(time, dtype=state.dtype)
        if time.ndim == 0:
            time = jnp.broadcast_to(time, (batch,))
        features = jnp.concatenate((flat, time[:, None]), axis=-1)
        hidden = nn.tanh(nn.Dense(self.hidden)(features))
        return nn.Dense(flat.shape[-1])(hidden).reshape(state.shape)


def _initialize(model, event_shape, seed):
    example = jnp.zeros((4, *event_shape), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(seed), 0.5, example)
    return split_flax_variables(variables)


def test_tiny_forward_then_backward_closed_loop_is_finite_and_shape_generic(capsys):
    source = GaussianSource(event_shape=(2,), scale=1.0)
    sde = EDMSDE(sigma_min=0.05, sigma_max=0.4, rho=3.0)
    forward_model = TinyController()
    forward_params, forward_constants = _initialize(forward_model, source.event_shape, 2)
    forward = train_forward_fixed_point(
        initial_params=forward_params,
        constants=forward_constants,
        control_apply=flax_apply(forward_model.apply),
        source=source,
        sde=sde,
        target_score_fn=lambda value: -value,
        config=ForwardTrainingConfig(
            outer_iterations=2,
            rollout_batches_per_outer=1,
            rollout_batch_size=12,
            rollout_steps=3,
            updates_per_outer=2,
            minibatch_size=8,
            replay_capacity=24,
            learning_rate=2.0e-3,
            damping=0.2,
            progress_every=2,
            seed=10,
        ),
    )
    forward_output = capsys.readouterr().out
    assert "[forward]" in forward_output
    assert "step=4/4" in forward_output
    assert "loss=" in forward_output
    assert forward_output.count("[forward]") == 3
    assert int(forward.state.step) == 4
    assert len(forward.metrics) == 4
    assert all(bool(item["finite"]) for item in forward.metrics)
    assert forward.replay_buffer.storage["endpoint_0"].shape == (24, 2)

    backward_model = TinyController()
    backward_params, backward_constants = _initialize(backward_model, source.event_shape, 3)
    backward = train_backward_score(
        initial_params=backward_params,
        constants=backward_constants,
        backward_apply=flax_apply(backward_model.apply),
        forward_params=forward.state.params,
        forward_constants=forward.state.constants,
        forward_apply=flax_apply(forward_model.apply),
        source=source,
        sde=sde,
        config=BackwardTrainingConfig(
            rollout_batches=2,
            rollout_batch_size=12,
            rollout_steps=3,
            updates=3,
            minibatch_size=8,
            replay_capacity=24,
            learning_rate=2.0e-3,
            progress_every=2,
            seed=11,
        ),
    )
    backward_output = capsys.readouterr().out
    assert "[backward]" in backward_output
    assert "step=3/3" in backward_output
    assert backward_output.count("[backward]") == 3
    assert int(backward.state.step) == 3
    assert len(backward.metrics) == 3
    assert all(bool(item["finite"]) for item in backward.metrics)
    leaves = jax.tree_util.tree_leaves(backward.state.params)
    assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def test_forward_fixed_point_uses_one_frozen_snapshot_per_outer_iteration():
    source = GaussianSource(event_shape=(1,))
    sde = EDMSDE(sigma_min=0.05, sigma_max=0.3, rho=2.0)
    model = TinyController(hidden=4)
    params, constants = _initialize(model, source.event_shape, 4)
    result = train_forward_fixed_point(
        initial_params=params,
        constants=constants,
        control_apply=flax_apply(model.apply),
        source=source,
        sde=sde,
        target_score_fn=lambda value: -value,
        config=ForwardTrainingConfig(
            outer_iterations=1,
            rollout_batches_per_outer=1,
            rollout_batch_size=10,
            rollout_steps=2,
            updates_per_outer=2,
            minibatch_size=10,
            replay_capacity=10,
            learning_rate=1.0e-2,
            damping=1.0,
            seed=22,
        ),
    )
    # On the first update current == frozen previous.  The second update still
    # compares against the outer-start snapshot and therefore has nonzero
    # damping (up to an extraordinarily unlikely exactly-zero gradient).
    assert result.metrics[0]["damping"] < 1.0e-12
    assert result.metrics[1]["damping"] > result.metrics[0]["damping"]


def test_forward_history_thinning_and_outer_snapshots_observe_completed_outers():
    source = GaussianSource(event_shape=(1,))
    sde = EDMSDE(sigma_min=0.05, sigma_max=0.3, rho=2.0)
    model = TinyController(hidden=4)
    params, constants = _initialize(model, source.event_shape, 41)
    callbacks: list[tuple[int, int, int]] = []

    result = train_forward_fixed_point(
        initial_params=params,
        constants=constants,
        control_apply=flax_apply(model.apply),
        source=source,
        sde=sde,
        target_score_fn=lambda value: -value,
        outer_callback=lambda completed_outer, snapshot: callbacks.append(
            (
                completed_outer,
                int(snapshot.state.step),
                int(snapshot.replay_buffer.size),
            )
        ),
        config=ForwardTrainingConfig(
            outer_iterations=2,
            rollout_batches_per_outer=1,
            rollout_batch_size=10,
            rollout_steps=2,
            updates_per_outer=3,
            minibatch_size=10,
            replay_capacity=10,
            learning_rate=1.0e-2,
            damping=1.0,
            history_every=2,
            seed=42,
        ),
    )

    assert int(result.state.step) == 6
    assert tuple(int(item["step"]) for item in result.metrics) == (1, 2, 3, 4, 6)
    assert callbacks == [(1, 3, 10), (2, 6, 10)]


def test_forward_without_outer_callback_remains_supported():
    source = GaussianSource(event_shape=(1,))
    sde = EDMSDE(sigma_min=0.05, sigma_max=0.3, rho=2.0)
    model = TinyController(hidden=4)
    params, constants = _initialize(model, source.event_shape, 43)

    result = train_forward_fixed_point(
        initial_params=params,
        constants=constants,
        control_apply=flax_apply(model.apply),
        source=source,
        sde=sde,
        target_score_fn=lambda value: -value,
        config=ForwardTrainingConfig(
            outer_iterations=1,
            rollout_batches_per_outer=1,
            rollout_batch_size=4,
            rollout_steps=2,
            updates_per_outer=1,
            minibatch_size=4,
            replay_capacity=4,
        ),
    )
    assert int(result.state.step) == 1
