"""Backward score matching for likelihood-carrying BMS probability flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from cg_bms_jax.data.buffer import (
    ReplayBufferState,
    extend_replay_buffer,
    init_replay_buffer,
    sample_replay_buffer,
)
from cg_bms_jax.data.source import GaussianSource
from cg_bms_jax.process.backward_matching import backward_score_target
from cg_bms_jax.process.bridge import sample_bridge
from cg_bms_jax.process.rollout import euler_maruyama_rollout
from cg_bms_jax.process.sde import EDMSDE
from cg_bms_jax.training.state import (
    ControlApply,
    ControllerTrainState,
    initialize_train_state,
    make_optimizer,
    tree_all_finite,
)

Array = jax.Array


@dataclass(frozen=True)
class BackwardTrainingConfig:
    """Loop sizes for the separately trained reverse-score controller."""

    rollout_batches: int = 10
    rollout_batch_size: int = 256
    rollout_steps: int = 100
    updates: int = 1_000
    minibatch_size: int = 256
    replay_capacity: int = 65_536
    learning_rate: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    time_epsilon: float = 1.0e-3
    zero_last_noise: bool = False
    progress_every: int = 100
    seed: int = 1

    def __post_init__(self) -> None:
        integer_fields = (
            self.rollout_batches,
            self.rollout_batch_size,
            self.rollout_steps,
            self.updates,
            self.minibatch_size,
            self.replay_capacity,
            self.progress_every,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("all backward loop sizes and replay_capacity must be positive")
        if self.replay_capacity < self.rollout_batch_size:
            raise ValueError("replay_capacity must be at least rollout_batch_size")
        if not 0.0 < self.time_epsilon < 0.5:
            raise ValueError("time_epsilon must be in (0, 0.5)")


class BackwardStepMetrics(NamedTuple):
    loss: Array
    gradient_norm: Array
    finite: Array


@dataclass(frozen=True)
class BackwardTrainingResult:
    state: ControllerTrainState
    replay_buffer: ReplayBufferState
    metrics: tuple[dict[str, float | bool | int], ...]


def _event_axes(array: Array) -> tuple[int, ...]:
    return tuple(range(1, array.ndim))


def collect_backward_pairs(
    key: Array,
    *,
    forward_params: Any,
    forward_constants: Any,
    forward_apply: ControlApply,
    source: GaussianSource,
    sde: EDMSDE,
    batch_size: int,
    rollout_steps: int,
    zero_last_noise: bool = False,
) -> dict[str, Array]:
    """Regenerate endpoint pairs using the frozen final forward controller."""

    rollout_source_key, rollout_key, pair_source_key = jax.random.split(key, 3)
    rollout_initial = source.sample(rollout_source_key, batch_size)

    def apply_for_rollout(params: Any, time: Array, state: Array) -> Array:
        return forward_apply(params, forward_constants, time, state)

    endpoint_1 = euler_maruyama_rollout(
        rollout_key,
        rollout_initial,
        sde=sde,
        control_apply=apply_for_rollout,
        control_params=forward_params,
        steps=rollout_steps,
        zero_last_noise=zero_last_noise,
    ).final
    endpoint_0 = source.sample(pair_source_key, batch_size, dtype=endpoint_1.dtype)
    finite_0 = jnp.all(jnp.isfinite(endpoint_0), axis=_event_axes(endpoint_0))
    finite_1 = jnp.all(jnp.isfinite(endpoint_1), axis=_event_axes(endpoint_1))
    valid = finite_0 & finite_1
    valid_broadcast = valid.reshape((batch_size,) + (1,) * (endpoint_1.ndim - 1))
    endpoint_0 = jnp.where(valid_broadcast, endpoint_0, 0.0)
    endpoint_1 = jnp.where(valid_broadcast, endpoint_1, 0.0)
    return {"endpoint_0": endpoint_0, "endpoint_1": endpoint_1, "valid": valid}


def make_backward_update_step(
    *,
    backward_apply: ControlApply,
    optimizer: optax.GradientTransformation,
    sde: EDMSDE,
    time_epsilon: float,
):
    """Build a JIT update for ``v = grad log p(X_t | X_0)``.

    The regression target is score-like.  It contains no ``g(t)^2`` factor;
    that factor is introduced only when constructing the SDE/PF-ODE drift.
    """

    def update(
        key: Array,
        state: ControllerTrainState,
        batch: dict[str, Array],
    ) -> tuple[ControllerTrainState, BackwardStepMetrics]:
        time_key, bridge_key = jax.random.split(key)
        batch_size = batch["endpoint_0"].shape[0]
        time = jax.random.uniform(
            time_key,
            (batch_size,),
            minval=time_epsilon,
            maxval=1.0 - time_epsilon,
            dtype=batch["endpoint_0"].dtype,
        )
        state_t = sample_bridge(
            bridge_key,
            sde,
            time,
            batch["endpoint_0"],
            batch["endpoint_1"],
        )
        target = backward_score_target(sde, time, batch["endpoint_0"], state_t)
        finite_target = jnp.all(jnp.isfinite(target), axis=_event_axes(target))
        valid = batch["valid"] & finite_target

        def loss_fn(params: Any) -> Array:
            prediction = backward_apply(params, state.constants, time, state_t)
            residual = prediction - jax.lax.stop_gradient(target)
            per_sample = jnp.mean(jnp.square(residual), axis=_event_axes(residual))
            weights = valid.astype(per_sample.dtype)
            return jnp.sum(jnp.where(valid, per_sample, 0.0)) / jnp.maximum(jnp.sum(weights), 1.0)

        loss, gradients = jax.value_and_grad(loss_fn)(state.params)
        gradient_norm = optax.global_norm(gradients)
        updates, optimizer_state = optimizer.update(gradients, state.optimizer_state, state.params)
        params = optax.apply_updates(state.params, updates)
        next_state = ControllerTrainState(
            params=params,
            constants=state.constants,
            optimizer_state=optimizer_state,
            step=state.step + jnp.asarray(1, dtype=state.step.dtype),
        )
        finite = tree_all_finite(next_state.params) & jnp.isfinite(loss)
        return next_state, BackwardStepMetrics(loss=loss, gradient_norm=gradient_norm, finite=finite)

    return jax.jit(update)


def train_backward_score(
    *,
    initial_params: Any,
    constants: Any,
    backward_apply: ControlApply,
    forward_params: Any,
    forward_constants: Any,
    forward_apply: ControlApply,
    source: GaussianSource,
    sde: EDMSDE,
    config: BackwardTrainingConfig,
) -> BackwardTrainingResult:
    """Train an independent backward controller from fresh forward rollouts."""

    optimizer = make_optimizer(
        learning_rate=config.learning_rate,
        gradient_clip_norm=config.gradient_clip_norm,
        weight_decay=config.weight_decay,
    )
    state = initialize_train_state(initial_params, constants, optimizer)
    example = source.sample(jax.random.PRNGKey(config.seed), config.rollout_batch_size)
    replay = init_replay_buffer(
        config.replay_capacity,
        {
            "endpoint_0": example,
            "endpoint_1": example,
            "valid": jnp.ones((config.rollout_batch_size,), dtype=jnp.bool_),
        },
    )
    collect = jax.jit(
        lambda key: collect_backward_pairs(
            key,
            forward_params=forward_params,
            forward_constants=forward_constants,
            forward_apply=forward_apply,
            source=source,
            sde=sde,
            batch_size=config.rollout_batch_size,
            rollout_steps=config.rollout_steps,
            zero_last_noise=config.zero_last_noise,
        )
    )
    update = make_backward_update_step(
        backward_apply=backward_apply,
        optimizer=optimizer,
        sde=sde,
        time_epsilon=config.time_epsilon,
    )

    key = jax.random.PRNGKey(config.seed)
    for _ in range(config.rollout_batches):
        key, collect_key = jax.random.split(key)
        pairs = collect(collect_key)
        valid_count = int(jax.device_get(jnp.sum(pairs["valid"])))
        if valid_count == 0:
            raise FloatingPointError("backward rollout produced no valid endpoint pairs")
        replay = extend_replay_buffer(replay, pairs)

    replay_valid_fraction = float(
        jax.device_get(
            jnp.sum(
                replay.storage["valid"]
                & (jnp.arange(replay.capacity, dtype=jnp.int32) < replay.size)
            )
            / jnp.maximum(replay.size, 1)
        )
    )

    history: list[dict[str, float | bool | int]] = []
    for _ in range(config.updates):
        key, sample_key, update_key = jax.random.split(key, 3)
        batch = sample_replay_buffer(sample_key, replay, config.minibatch_size)
        state, step_metrics = update(update_key, state, batch)
        host_metrics = jax.device_get(step_metrics)
        record: dict[str, float | bool | int] = {
            "step": int(jax.device_get(state.step)),
            "loss": float(host_metrics.loss),
            "gradient_norm": float(host_metrics.gradient_norm),
            "replay_valid_fraction": replay_valid_fraction,
            "finite": bool(host_metrics.finite),
        }
        history.append(record)
        step = int(record["step"])
        if (
            step == 1
            or step % config.progress_every == 0
            or step == config.updates
            or not record["finite"]
        ):
            progress = 100.0 * step / config.updates
            print(
                f"[backward] step={step}/{config.updates} ({progress:6.2f}%) "
                f"loss={record['loss']:.6e} "
                f"grad_norm={record['gradient_norm']:.6e} "
                f"valid={record['replay_valid_fraction']:.4f} "
                f"finite={record['finite']}",
                flush=True,
            )
        if not record["finite"]:
            raise FloatingPointError("non-finite backward BMS update")
    return BackwardTrainingResult(state=state, replay_buffer=replay, metrics=tuple(history))
