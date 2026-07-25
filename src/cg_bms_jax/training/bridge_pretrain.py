"""Supervised bridge-matching warm start for a BMS controller.

This module implements the data pretraining stage used for molecular WT-ASBS
(Eq. 66 in that work), adapted to the dense arrays used by this repository.
It is deliberately separate from fixed-point BMS training: no potential,
terminal score, damping snapshot, or replay buffer is involved.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from cg_bms_jax.data.source import GaussianSource
from cg_bms_jax.process.bridge import conditional_score_1t, sample_bridge
from cg_bms_jax.process.sde import EDMSDE
from cg_bms_jax.training.state import (
    ControlApply,
    ControllerTrainState,
    initialize_train_state,
    make_learning_rate,
    make_optimizer,
    tree_all_finite,
)

Array = jax.Array
EndpointBatchProvider = Callable[[Array, int], Array]
BridgeCheckpointCallback = Callable[
    [ControllerTrainState, tuple[dict[str, float | bool | int], ...]],
    None,
]


@dataclass(frozen=True)
class BridgePretrainConfig:
    """Loop and optimizer settings for WT-ASBS-style bridge pretraining.

    The defaults reproduce the single-controller Ala2 pretraining schedule in
    the released WT-ASBS configuration: 100k updates, AdamW at ``3e-4``, a 5k
    update warmup, cosine decay to ``1e-4``, and global gradient clipping at
    100.  ``max_time=0.99`` avoids the singular conditional score at time one.
    """

    updates: int = 100_000
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    final_learning_rate: float = 1.0e-4
    learning_rate_warmup_updates: int = 5_000
    learning_rate_decay_updates: int | None = None
    learning_rate_warmup_start_factor: float = 1.0e-3
    gradient_clip_norm: float = 100.0
    weight_decay: float = 1.0e-3
    max_time: float = 0.99
    progress_every: int = 100
    checkpoint_every: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.updates <= 0:
            raise ValueError("updates must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.progress_every <= 0:
            raise ValueError("progress_every must be positive")
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive when enabled")
        if not 0.0 < self.max_time < 1.0:
            raise ValueError("max_time must lie strictly between zero and one")
        if self.learning_rate_warmup_updates < 0:
            raise ValueError("learning_rate_warmup_updates must be non-negative")
        horizon = (
            self.updates
            if self.learning_rate_decay_updates is None
            else self.learning_rate_decay_updates
        )
        if horizon <= 0:
            raise ValueError("learning_rate_decay_updates must be positive when enabled")
        if horizon < self.learning_rate_warmup_updates:
            raise ValueError("learning-rate decay horizon cannot precede warmup")


class BridgePretrainStepMetrics(NamedTuple):
    loss: Array
    gradient_norm: Array
    remaining_variance_mean: Array
    finite: Array


@dataclass(frozen=True)
class BridgePretrainResult:
    """Final controller state and host-side metrics for every update."""

    state: ControllerTrainState
    metrics: tuple[dict[str, float | bool | int], ...]


def make_array_endpoint_provider(endpoints: Array) -> EndpointBatchProvider:
    """Return a deterministic-key sampler over an in-memory endpoint array.

    Rows are sampled independently with replacement, matching
    ``PreGeneratedAtomsSource`` in WT-ASBS.  The returned provider is useful for
    tests and moderate in-memory datasets; a caller may instead supply any
    provider with the same ``(key, batch_size)`` signature.
    """

    values = jnp.asarray(endpoints)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("endpoints must have shape (samples, *event_shape) with samples > 0")
    if not bool(jax.device_get(jnp.all(jnp.isfinite(values)))):
        raise ValueError("endpoints must be finite")

    def provide(key: Array, batch_size: int) -> Array:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = jax.random.randint(key, (batch_size,), 0, values.shape[0])
        return values[indices]

    return provide


def remaining_bridge_variance(
    sde: EDMSDE,
    time: Array | float,
    reference: Array,
) -> Array:
    """Return ``kappa(1)-kappa(t)`` broadcast over an event batch."""

    reference = jnp.asarray(reference)
    if reference.ndim < 2:
        raise ValueError("reference must have a batch-plus-event shape")
    time_array = jnp.asarray(time, dtype=reference.dtype)
    if time_array.ndim == 0:
        time_array = jnp.broadcast_to(time_array, (reference.shape[0],))
    if time_array.shape != (reference.shape[0],):
        raise ValueError(f"time must be scalar or have shape ({reference.shape[0]},)")
    time_b = time_array.reshape((reference.shape[0],) + (1,) * (reference.ndim - 1))
    return sde.remaining_variance(time_b)


def variance_weighted_bridge_loss(
    prediction: Array,
    target: Array,
    remaining_variance: Array | float,
) -> Array:
    r"""Return the WT-ASBS Eq. 66 regression objective.

    ``remaining_variance`` is the conditional variance
    ``kappa(1)-kappa(t)``.  Weighting the squared score residual by this value
    removes one power of the endpoint singularity while preserving the
    bridge-matching optimum.
    """

    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have an identical batch-plus-event shape")
    variance = jnp.asarray(remaining_variance, dtype=prediction.dtype)
    if variance.ndim == 0:
        variance = jnp.broadcast_to(
            variance,
            (prediction.shape[0],) + (1,) * (prediction.ndim - 1),
        )
    elif variance.shape == (prediction.shape[0],):
        variance = variance.reshape(
            (prediction.shape[0],) + (1,) * (prediction.ndim - 1)
        )
    try:
        variance = jnp.broadcast_to(variance, prediction.shape)
    except ValueError as error:
        raise ValueError("remaining_variance is not broadcastable to the prediction shape") from error
    return jnp.mean(variance * jnp.square(prediction - target))


def make_bridge_pretrain_update_step(
    *,
    control_apply: ControlApply,
    optimizer: optax.GradientTransformation,
    source: GaussianSource,
    sde: EDMSDE,
    max_time: float,
) -> Callable[
    [Array, ControllerTrainState, Array],
    tuple[ControllerTrainState, BridgePretrainStepMetrics],
]:
    """Build one JIT-compiled supervised bridge-matching update."""

    if not 0.0 < max_time < 1.0:
        raise ValueError("max_time must lie strictly between zero and one")

    def update(
        key: Array,
        state: ControllerTrainState,
        endpoint_1: Array,
    ) -> tuple[ControllerTrainState, BridgePretrainStepMetrics]:
        endpoint_1 = jnp.asarray(endpoint_1)
        if endpoint_1.ndim < 2 or tuple(endpoint_1.shape[1:]) != tuple(source.event_shape):
            raise ValueError(
                f"endpoint batch must have shape (batch, {source.event_shape}), got {endpoint_1.shape}"
            )
        source_key, time_key, bridge_key = jax.random.split(key, 3)
        endpoint_0 = source.sample(
            source_key,
            endpoint_1.shape[0],
            dtype=endpoint_1.dtype,
        )
        time = jax.random.uniform(
            time_key,
            (endpoint_1.shape[0],),
            minval=0.0,
            maxval=max_time,
            dtype=endpoint_1.dtype,
        )
        state_t = sample_bridge(bridge_key, sde, time, endpoint_0, endpoint_1)
        target = conditional_score_1t(sde, time, endpoint_1, state_t)
        variance = remaining_bridge_variance(sde, time, endpoint_1)

        def loss_fn(params: Any) -> Array:
            prediction = control_apply(params, state.constants, time, state_t)
            if prediction.shape != target.shape:
                raise ValueError("controller output must match the endpoint batch shape")
            return variance_weighted_bridge_loss(
                prediction,
                jax.lax.stop_gradient(target),
                variance,
            )

        loss, gradients = jax.value_and_grad(loss_fn)(state.params)
        gradient_norm = optax.global_norm(gradients)
        updates, optimizer_state = optimizer.update(
            gradients,
            state.optimizer_state,
            state.params,
        )
        params = optax.apply_updates(state.params, updates)
        next_state = ControllerTrainState(
            params=params,
            constants=state.constants,
            optimizer_state=optimizer_state,
            step=state.step + jnp.asarray(1, dtype=state.step.dtype),
        )
        finite = (
            jnp.all(jnp.isfinite(endpoint_0))
            & jnp.all(jnp.isfinite(endpoint_1))
            & jnp.all(jnp.isfinite(state_t))
            & jnp.all(jnp.isfinite(target))
            & jnp.isfinite(loss)
            & jnp.isfinite(gradient_norm)
            & tree_all_finite(gradients)
            & tree_all_finite(next_state.params)
        )
        return next_state, BridgePretrainStepMetrics(
            loss=loss,
            gradient_norm=gradient_norm,
            remaining_variance_mean=jnp.mean(variance),
            finite=finite,
        )

    return jax.jit(update)


def train_bridge_pretrain(
    *,
    initial_params: Any,
    constants: Any,
    control_apply: ControlApply,
    source: GaussianSource,
    sde: EDMSDE,
    config: BridgePretrainConfig,
    target_endpoints: Array | None = None,
    endpoint_provider: EndpointBatchProvider | None = None,
    checkpoint_callback: BridgeCheckpointCallback | None = None,
) -> BridgePretrainResult:
    """Train a controller against independent source/data endpoint pairs.

    Exactly one of ``target_endpoints`` and ``endpoint_provider`` must be
    supplied.  Endpoint sampling and source sampling receive independent PRNG
    keys, so the coupling is the product coupling used by WT-ASBS pretraining.
    """

    if (target_endpoints is None) == (endpoint_provider is None):
        raise ValueError("provide exactly one of target_endpoints or endpoint_provider")
    if config.checkpoint_every is not None and checkpoint_callback is None:
        raise ValueError(
            "checkpoint_callback is required when checkpoint_every is enabled"
        )
    provider = (
        make_array_endpoint_provider(target_endpoints)
        if endpoint_provider is None
        else endpoint_provider
    )

    learning_rate = make_learning_rate(
        peak=config.learning_rate,
        final=config.final_learning_rate,
        warmup_updates=config.learning_rate_warmup_updates,
        decay_updates=config.learning_rate_decay_updates,
        total_updates=config.updates,
        warmup_start_factor=config.learning_rate_warmup_start_factor,
    )
    optimizer = make_optimizer(
        learning_rate=learning_rate,
        gradient_clip_norm=config.gradient_clip_norm,
        weight_decay=config.weight_decay,
    )
    state = initialize_train_state(initial_params, constants, optimizer)
    if not bool(jax.device_get(tree_all_finite(state.params))):
        raise FloatingPointError("initial bridge-pretrain controller is non-finite")
    update = make_bridge_pretrain_update_step(
        control_apply=control_apply,
        optimizer=optimizer,
        source=source,
        sde=sde,
        max_time=config.max_time,
    )

    key = jax.random.PRNGKey(config.seed)
    history: list[dict[str, float | bool | int]] = []
    for update_index in range(config.updates):
        key, endpoint_key, update_key = jax.random.split(key, 3)
        endpoint_1 = jnp.asarray(provider(endpoint_key, config.batch_size))
        expected_shape = (config.batch_size, *source.event_shape)
        if endpoint_1.shape != expected_shape:
            raise ValueError(
                f"endpoint provider must return shape {expected_shape}, got {endpoint_1.shape}"
            )
        state, step_metrics = update(update_key, state, endpoint_1)
        host_metrics = jax.device_get(step_metrics)
        step = int(jax.device_get(state.step))
        record: dict[str, float | bool | int] = {
            "step": step,
            "loss": float(host_metrics.loss),
            "gradient_norm": float(host_metrics.gradient_norm),
            "remaining_variance_mean": float(host_metrics.remaining_variance_mean),
            "finite": bool(host_metrics.finite),
            "learning_rate": float(
                config.learning_rate
                if isinstance(learning_rate, (int, float))
                else jax.device_get(learning_rate(state.step - 1))
            ),
        }
        history.append(record)
        should_report = (
            step == 1
            or step % config.progress_every == 0
            or update_index + 1 == config.updates
            or not record["finite"]
        )
        if should_report:
            progress = 100.0 * step / config.updates
            print(
                f"[bridge-pretrain] step={step}/{config.updates} ({progress:6.2f}%) "
                f"loss={record['loss']:.6e} grad_norm={record['gradient_norm']:.6e} "
                f"remaining_var={record['remaining_variance_mean']:.6e} "
                f"lr={record['learning_rate']:.6e} finite={record['finite']}",
                flush=True,
            )
        if not record["finite"]:
            raise FloatingPointError(
                f"non-finite bridge-pretrain update at step {step}"
            )
        if (
            checkpoint_callback is not None
            and config.checkpoint_every is not None
            and step % config.checkpoint_every == 0
            and step < config.updates
        ):
            checkpoint_callback(state, tuple(history))

    return BridgePretrainResult(state=state, metrics=tuple(history))


__all__ = [
    "BridgePretrainConfig",
    "BridgePretrainResult",
    "BridgePretrainStepMetrics",
    "BridgeCheckpointCallback",
    "EndpointBatchProvider",
    "make_array_endpoint_provider",
    "make_bridge_pretrain_update_step",
    "remaining_bridge_variance",
    "train_bridge_pretrain",
    "variance_weighted_bridge_loss",
]
