"""Faithful fixed-point forward training for Bridge Matching Samplers."""

from __future__ import annotations

from collections.abc import Callable
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
from cg_bms_jax.process.bridge import sample_bridge
from cg_bms_jax.process.forward_matching import nelson_target
from cg_bms_jax.process.rollout import euler_maruyama_rollout
from cg_bms_jax.process.sde import EDMSDE
from cg_bms_jax.potential.base import ScoreDecomposition
from cg_bms_jax.training.state import (
    ControlApply,
    ControllerTrainState,
    initialize_train_state,
    make_learning_rate,
    make_optimizer,
    tree_all_finite,
)

Array = jax.Array
TargetScoreFn = Callable[[Array], Array | ScoreDecomposition]


@dataclass(frozen=True)
class ForwardTrainingConfig:
    """Static loop sizes and optimizer settings for forward fixed points."""

    outer_iterations: int = 10
    rollout_batches_per_outer: int = 1
    rollout_batch_size: int = 256
    rollout_steps: int = 100
    updates_per_outer: int = 100
    minibatch_size: int = 256
    replay_capacity: int = 65_536
    learning_rate: float = 1.0e-4
    final_learning_rate: float | None = None
    learning_rate_warmup_updates: int = 0
    learning_rate_decay_updates: int | None = None
    learning_rate_warmup_start_factor: float = 1.0e-3
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    damping: float = 0.1
    previous_model_interval_outer: int = 1
    time_epsilon: float = 1.0e-3
    zero_last_noise: bool = False
    progress_every: int = 100
    history_every: int = 1
    seed: int = 0
    terminal_score_clip_norm: float | None = None
    terminal_score_clip_preserve_mean: bool = False
    terminal_score_clip_mean_mode: str = "target_mean"

    def __post_init__(self) -> None:
        integer_fields = (
            self.outer_iterations,
            self.rollout_batches_per_outer,
            self.rollout_batch_size,
            self.rollout_steps,
            self.updates_per_outer,
            self.minibatch_size,
            self.replay_capacity,
            self.progress_every,
            self.history_every,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("all forward loop sizes and replay_capacity must be positive")
        if self.replay_capacity < self.rollout_batch_size:
            raise ValueError("replay_capacity must be at least rollout_batch_size")
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")
        if self.previous_model_interval_outer <= 0:
            raise ValueError("previous_model_interval_outer must be positive")
        if self.learning_rate_warmup_updates < 0:
            raise ValueError("learning_rate_warmup_updates must be non-negative")
        if (
            self.learning_rate_decay_updates is not None
            and self.learning_rate_decay_updates <= 0
        ):
            raise ValueError("learning_rate_decay_updates must be positive when enabled")
        if (
            self.learning_rate_decay_updates is not None
            and self.learning_rate_decay_updates < self.learning_rate_warmup_updates
        ):
            raise ValueError("learning-rate decay horizon cannot precede warmup")
        if not 0.0 < self.learning_rate_warmup_start_factor <= 1.0:
            raise ValueError("learning_rate_warmup_start_factor must be in (0, 1]")
        if self.final_learning_rate is not None:
            if self.final_learning_rate < 0.0:
                raise ValueError("final_learning_rate must be non-negative")
            if self.final_learning_rate > self.learning_rate:
                raise ValueError("final_learning_rate cannot exceed learning_rate")
        if self.terminal_score_clip_norm is not None and self.terminal_score_clip_norm <= 0.0:
            raise ValueError("terminal_score_clip_norm must be positive when enabled")
        if self.terminal_score_clip_mean_mode not in {
            "target_mean",
            "analytic_standard_normal",
        }:
            raise ValueError(
                "terminal_score_clip_mean_mode must be 'target_mean' or "
                "'analytic_standard_normal'"
            )
        if (
            not self.terminal_score_clip_preserve_mean
            and self.terminal_score_clip_mean_mode != "target_mean"
        ):
            raise ValueError(
                "terminal_score_clip_mean_mode only applies when "
                "terminal_score_clip_preserve_mean=true"
            )
        if not 0.0 < self.time_epsilon < 0.5:
            raise ValueError("time_epsilon must be in (0, 0.5)")


class ForwardStepMetrics(NamedTuple):
    loss: Array
    matching: Array
    damping: Array
    gradient_norm: Array
    finite: Array


@dataclass(frozen=True)
class ForwardTrainingResult:
    """Final state, replay data and host-side per-update metrics."""

    state: ControllerTrainState
    replay_buffer: ReplayBufferState
    metrics: tuple[dict[str, float | bool | int], ...]


@dataclass(frozen=True)
class ForwardOuterSnapshot:
    """Controller and replay state at a completed fixed-point outer loop."""

    state: ControllerTrainState
    replay_buffer: ReplayBufferState


ForwardOuterCallback = Callable[[int, ForwardOuterSnapshot], None]


def _event_axes(array: Array) -> tuple[int, ...]:
    return tuple(range(1, array.ndim))


def _finite_per_sample(array: Array) -> Array:
    axes = _event_axes(array)
    return jnp.all(jnp.isfinite(array), axis=axes) if axes else jnp.isfinite(array)


def _masked_square_mean(residual: Array, valid: Array) -> Array:
    per_sample = jnp.mean(jnp.square(residual), axis=_event_axes(residual))
    weights = jnp.asarray(valid, dtype=per_sample.dtype)
    return jnp.sum(jnp.where(valid, per_sample, 0.0)) / jnp.maximum(jnp.sum(weights), 1.0)


def _per_bead_norm(array: Array, *, eps: float = 1.0e-10) -> Array:
    """Return a stable norm for each three-dimensional bead vector.

    OpenMM can return finite collision forces whose squared float32 magnitude
    overflows.  Scaling by the largest component before squaring preserves the
    upstream per-bead clipping rule without turning a finite direction into an
    infinite norm.
    """

    value = jnp.asarray(array)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("terminal score clipping requires shape (batch, beads, 3)")
    scale = jnp.max(jnp.abs(value), axis=-1)
    safe_scale = jnp.where(scale > 0.0, scale, jnp.asarray(1.0, value.dtype))
    scaled = value / safe_scale[..., None]
    norm = scale * jnp.sqrt(jnp.sum(jnp.square(scaled), axis=-1))
    return jnp.maximum(norm, jnp.sqrt(jnp.asarray(eps, dtype=value.dtype)))


def clip_target_score_by_bead(
    target_score: Array,
    max_norm: float,
    *,
    eps: float = 1.0e-10,
) -> Array:
    """Clip each bead's 3D terminal-score norm while preserving direction.

    This is the JAX equivalent of the original BMS ``grad_clip_val`` rule:
    ``score *= min(max_norm / sqrt(sum(score**2) + eps), 1)``.  It is distinct
    from optimizer gradient clipping and is applied before targets enter replay.
    """

    if isinstance(max_norm, (int, float)) and max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    score = jnp.asarray(target_score)
    if score.ndim != 3 or score.shape[-1] != 3:
        raise ValueError("terminal score clipping requires shape (batch, beads, 3)")
    limit = jnp.asarray(max_norm, dtype=score.dtype)
    scale = jnp.max(jnp.abs(score), axis=-1, keepdims=True)
    safe_scale = jnp.where(scale > 0.0, scale, jnp.asarray(1.0, score.dtype))
    unit0 = score / safe_scale
    unit_norm = jnp.sqrt(
        jnp.sum(jnp.square(unit0), axis=-1, keepdims=True)
        + jnp.asarray(eps, dtype=score.dtype)
    )
    direction = unit0 / unit_norm
    # Compare in scaled space so neither ``score**2`` nor ``scale*unit_norm``
    # needs to be represented for a finite but extreme OpenMM force.
    needs_clip = scale > limit / unit_norm
    clipped = limit * direction
    return jnp.where(needs_clip, clipped, score)


class TargetScoreClipDiagnostics(NamedTuple):
    """Per-bead diagnostics computed from the same target sent to replay."""

    raw_norm: Array
    clipped_norm: Array
    was_clipped: Array


def collect_forward_pairs(
    key: Array,
    *,
    params: Any,
    constants: Any,
    source: GaussianSource,
    sde: EDMSDE,
    control_apply: ControlApply,
    target_score_fn: TargetScoreFn,
    batch_size: int,
    rollout_steps: int,
    zero_last_noise: bool = False,
    terminal_score_clip_norm: float | None = None,
    terminal_score_clip_preserve_mean: bool = False,
    terminal_score_clip_mean_mode: str = "target_mean",
) -> dict[str, Array]:
    """Generate one independent endpoint coupling for a fixed-point update.

    The source sample used to start the controlled rollout is intentionally
    discarded.  A fresh independent source endpoint is paired with the
    generated terminal sample, exactly as in the BMS fixed-point algorithm.
    """

    endpoint_0, endpoint_1 = collect_forward_endpoints(
        key,
        params=params,
        constants=constants,
        source=source,
        sde=sde,
        control_apply=control_apply,
        batch_size=batch_size,
        rollout_steps=rollout_steps,
        zero_last_noise=zero_last_noise,
    )
    return attach_forward_target(
        endpoint_0,
        endpoint_1,
        target_score_fn,
        terminal_score_clip_norm=terminal_score_clip_norm,
        terminal_score_clip_preserve_mean=terminal_score_clip_preserve_mean,
        terminal_score_clip_mean_mode=terminal_score_clip_mean_mode,
    )


def collect_forward_endpoints(
    key: Array,
    *,
    params: Any,
    constants: Any,
    source: GaussianSource,
    sde: EDMSDE,
    control_apply: ControlApply,
    batch_size: int,
    rollout_steps: int,
    zero_last_noise: bool = False,
) -> tuple[Array, Array]:
    """Roll out and independently pair endpoints without evaluating the PMF."""

    rollout_source_key, rollout_key, pair_source_key = jax.random.split(key, 3)
    rollout_initial = source.sample(rollout_source_key, batch_size)

    def apply_for_rollout(current_params: Any, time: Array, state: Array) -> Array:
        return control_apply(current_params, constants, time, state)

    endpoint_1 = euler_maruyama_rollout(
        rollout_key,
        rollout_initial,
        sde=sde,
        control_apply=apply_for_rollout,
        control_params=params,
        steps=rollout_steps,
        zero_last_noise=zero_last_noise,
    ).final
    endpoint_0 = source.sample(pair_source_key, batch_size, dtype=endpoint_1.dtype)
    return endpoint_0, endpoint_1


def attach_forward_target(
    endpoint_0: Array,
    endpoint_1: Array,
    target_score_fn: TargetScoreFn,
    *,
    terminal_score_clip_norm: float | None = None,
    terminal_score_clip_preserve_mean: bool = False,
    terminal_score_clip_mean_mode: str = "target_mean",
) -> dict[str, Array]:
    """Evaluate/sanitize terminal scores outside the enclosing rollout JIT.

    Keeping this boundary explicit is required for exact CG-BG MACE parity:
    nesting the standalone PMF executable inside the rollout JIT lets XLA
    constant-fold the checkpoint and changes float32 scatter reductions.
    """

    pairs, _ = _attach_forward_target_with_diagnostics(
        endpoint_0,
        endpoint_1,
        target_score_fn,
        terminal_score_clip_norm=terminal_score_clip_norm,
        terminal_score_clip_preserve_mean=terminal_score_clip_preserve_mean,
        terminal_score_clip_mean_mode=terminal_score_clip_mean_mode,
    )
    return pairs


def _attach_forward_target_with_diagnostics(
    endpoint_0: Array,
    endpoint_1: Array,
    target_score_fn: TargetScoreFn,
    *,
    terminal_score_clip_norm: float | None,
    terminal_score_clip_preserve_mean: bool = False,
    terminal_score_clip_mean_mode: str = "target_mean",
) -> tuple[dict[str, Array], TargetScoreClipDiagnostics | None]:
    """Attach a replay target and optionally retain host-side clip diagnostics."""

    batch_size = endpoint_1.shape[0]
    target = target_score_fn(endpoint_1)
    score_decomposition = (
        target if isinstance(target, ScoreDecomposition) else None
    )
    if score_decomposition is None:
        target_score = jnp.asarray(target, dtype=endpoint_1.dtype)
    else:
        clippable_score = jnp.asarray(
            score_decomposition.clippable, dtype=endpoint_1.dtype
        )
        preserved_score = jnp.asarray(
            score_decomposition.preserved, dtype=endpoint_1.dtype
        )
        if clippable_score.shape != endpoint_1.shape:
            raise ValueError("clippable score must return the endpoint batch shape")
        if preserved_score.shape != endpoint_1.shape:
            raise ValueError("preserved score must return the endpoint batch shape")
        target_score = clippable_score + preserved_score
    if target_score.shape != endpoint_1.shape:
        raise ValueError("target_score_fn must return the endpoint batch shape")
    valid = _finite_per_sample(endpoint_0) & _finite_per_sample(endpoint_1)
    valid = valid & _finite_per_sample(target_score)
    valid_broadcast = valid.reshape((batch_size,) + (1,) * (endpoint_1.ndim - 1))
    diagnostics = None
    if terminal_score_clip_norm is not None:
        # Do not let a rejected endpoint take part in clipping arithmetic:
        # operations such as Inf - Inf would otherwise create NaNs before the
        # invalid replay row is masked below.
        score_to_clip = jnp.where(valid_broadcast, target_score, 0.0)
        mean_score = None
        if terminal_score_clip_preserve_mean:
            # Ambient molecular targets augment a translation-invariant shape
            # density with a full-rank COM density.  Per-bead clipping the total
            # score would manufacture a spurious COM component.  Clip only the
            # mean-free shape score, reproject it, then restore the COM score.
            #
            # ``target_mean`` retains the historical CG path.  For the exact
            # standard-normal orthogonal COM used by all-atom Ala2, infer the
            # per-particle COM score analytically as ``-mean(endpoint)``.  This
            # avoids recovering it by cancelling enormous, nearly zero-sum
            # OpenMM forces in float32.
            if terminal_score_clip_mean_mode == "target_mean":
                mean_score = jnp.mean(score_to_clip, axis=-2, keepdims=True)
                score_to_clip = score_to_clip - mean_score
                score_to_clip = score_to_clip - jnp.mean(
                    score_to_clip, axis=-2, keepdims=True
                )
            elif terminal_score_clip_mean_mode == "analytic_standard_normal":
                if score_decomposition is None:
                    raise ValueError(
                        "analytic_standard_normal clipping requires an explicit "
                        "ScoreDecomposition"
                    )
                score_to_clip = jnp.where(
                    valid_broadcast, clippable_score, 0.0
                )
                mean_score = jnp.where(
                    valid_broadcast, preserved_score, 0.0
                )
            else:
                raise ValueError(
                    "terminal_score_clip_mean_mode must be 'target_mean' or "
                    "'analytic_standard_normal'"
                )
        raw_norm = _per_bead_norm(score_to_clip)
        clipped_score = clip_target_score_by_bead(
            score_to_clip, terminal_score_clip_norm
        )
        if mean_score is not None:
            clipped_score = clipped_score - jnp.mean(
                clipped_score, axis=-2, keepdims=True
            )
            clipped_score = clipped_score + mean_score
        diagnostics = TargetScoreClipDiagnostics(
            raw_norm=raw_norm,
            clipped_norm=_per_bead_norm(
                clipped_score if mean_score is None else clipped_score - mean_score
            ),
            was_clipped=raw_norm > terminal_score_clip_norm,
        )
        target_score = clipped_score
    # Invalid values remain marked in the replay buffer but are sanitized so
    # that a masked batch cannot poison a compiled gradient with NaNs.
    endpoint_0 = jnp.where(valid_broadcast, endpoint_0, 0.0)
    endpoint_1 = jnp.where(valid_broadcast, endpoint_1, 0.0)
    target_score = jnp.where(valid_broadcast, target_score, 0.0)
    pairs = {
        "endpoint_0": endpoint_0,
        "endpoint_1": endpoint_1,
        "target_score": target_score,
        "valid": valid,
    }
    return pairs, diagnostics


def make_forward_update_step(
    *,
    control_apply: ControlApply,
    optimizer: optax.GradientTransformation,
    source: GaussianSource,
    sde: EDMSDE,
    damping: float,
    time_epsilon: float,
) -> Callable[[Array, ControllerTrainState, Any, dict[str, Array]], tuple[ControllerTrainState, ForwardStepMetrics]]:
    """Build a JIT-compatible update with a frozen previous controller."""

    def update(
        key: Array,
        state: ControllerTrainState,
        previous_params: Any,
        batch: dict[str, Array],
    ) -> tuple[ControllerTrainState, ForwardStepMetrics]:
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
        target = nelson_target(
            sde,
            time,
            batch["endpoint_0"],
            batch["endpoint_1"],
            source.score(batch["endpoint_0"]),
            batch["target_score"],
        )
        valid = batch["valid"] & _finite_per_sample(state_t) & _finite_per_sample(target)
        previous_prediction = jax.lax.stop_gradient(
            control_apply(previous_params, state.constants, time, state_t)
        )

        def loss_fn(params: Any) -> tuple[Array, tuple[Array, Array]]:
            prediction = control_apply(params, state.constants, time, state_t)
            matching_loss = _masked_square_mean(prediction - jax.lax.stop_gradient(target), valid)
            damping_loss = _masked_square_mean(
                prediction - jax.lax.stop_gradient(previous_prediction), valid
            )
            return matching_loss + damping * damping_loss, (matching_loss, damping_loss)

        (loss, (matching_loss, damping_loss)), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
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
        return next_state, ForwardStepMetrics(
            loss=loss,
            matching=matching_loss,
            damping=damping_loss,
            gradient_norm=gradient_norm,
            finite=finite,
        )

    return jax.jit(update)


def train_forward_fixed_point(
    *,
    initial_params: Any,
    constants: Any,
    control_apply: ControlApply,
    source: GaussianSource,
    sde: EDMSDE,
    target_score_fn: TargetScoreFn,
    config: ForwardTrainingConfig,
    outer_callback: ForwardOuterCallback | None = None,
) -> ForwardTrainingResult:
    """Run the complete forward BMS fixed-point loop.

    The previous controller is snapshotted once per outer iteration.  All
    inner optimizer updates in that iteration use the same immutable snapshot
    in the damping term.
    """

    learning_rate = make_learning_rate(
        peak=config.learning_rate,
        final=config.final_learning_rate,
        warmup_updates=config.learning_rate_warmup_updates,
        decay_updates=config.learning_rate_decay_updates,
        total_updates=config.outer_iterations * config.updates_per_outer,
        warmup_start_factor=config.learning_rate_warmup_start_factor,
    )

    optimizer = make_optimizer(
        learning_rate=learning_rate,
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
            "target_score": example,
            "valid": jnp.ones((config.rollout_batch_size,), dtype=jnp.bool_),
        },
    )
    update = make_forward_update_step(
        control_apply=control_apply,
        optimizer=optimizer,
        source=source,
        sde=sde,
        damping=config.damping,
        time_epsilon=config.time_epsilon,
    )
    collect_endpoints = jax.jit(
        lambda key, params: collect_forward_endpoints(
            key,
            params=params,
            constants=state.constants,
            source=source,
            sde=sde,
            control_apply=control_apply,
            batch_size=config.rollout_batch_size,
            rollout_steps=config.rollout_steps,
            zero_last_noise=config.zero_last_noise,
        )
    )

    key = jax.random.PRNGKey(config.seed)
    history: list[dict[str, float | bool | int]] = []
    total_updates = config.outer_iterations * config.updates_per_outer
    previous_params = state.params
    for outer in range(config.outer_iterations):
        if outer % config.previous_model_interval_outer == 0:
            previous_params = state.params
        score_raw_norm_sum = 0.0
        score_raw_norm_max = 0.0
        score_clipped_norm_max = 0.0
        score_clipped_count = 0
        score_bead_count = 0
        for _ in range(config.rollout_batches_per_outer):
            key, collect_key = jax.random.split(key)
            endpoint_0, endpoint_1 = collect_endpoints(collect_key, state.params)
            pairs, clip_diagnostics = _attach_forward_target_with_diagnostics(
                endpoint_0,
                endpoint_1,
                target_score_fn,
                terminal_score_clip_norm=config.terminal_score_clip_norm,
                terminal_score_clip_preserve_mean=(
                    config.terminal_score_clip_preserve_mean
                ),
                terminal_score_clip_mean_mode=(
                    config.terminal_score_clip_mean_mode
                ),
            )
            valid_count = int(jax.device_get(jnp.sum(pairs["valid"])))
            if valid_count == 0:
                raise FloatingPointError("forward rollout produced no valid endpoint pairs")
            if clip_diagnostics is not None:
                valid_beads = jnp.broadcast_to(
                    pairs["valid"][:, None], clip_diagnostics.raw_norm.shape
                )
                raw_norm = jnp.where(valid_beads, clip_diagnostics.raw_norm, 0.0)
                clipped_norm = jnp.where(valid_beads, clip_diagnostics.clipped_norm, 0.0)
                score_raw_norm_sum += float(jax.device_get(jnp.sum(raw_norm)))
                score_raw_norm_max = max(
                    score_raw_norm_max,
                    float(jax.device_get(jnp.max(raw_norm))),
                )
                score_clipped_norm_max = max(
                    score_clipped_norm_max,
                    float(jax.device_get(jnp.max(clipped_norm))),
                )
                score_clipped_count += int(
                    jax.device_get(jnp.sum(clip_diagnostics.was_clipped & valid_beads))
                )
                score_bead_count += int(jax.device_get(jnp.sum(valid_beads)))
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
        for update_in_outer in range(config.updates_per_outer):
            key, sample_key, update_key = jax.random.split(key, 3)
            batch = sample_replay_buffer(sample_key, replay, config.minibatch_size)
            state, step_metrics = update(update_key, state, previous_params, batch)
            host_metrics = jax.device_get(step_metrics)
            record: dict[str, float | bool | int] = {
                "outer_iteration": outer,
                "step": int(jax.device_get(state.step)),
                "loss": float(host_metrics.loss),
                "matching": float(host_metrics.matching),
                "damping": float(host_metrics.damping),
                "gradient_norm": float(host_metrics.gradient_norm),
                "replay_valid_fraction": replay_valid_fraction,
                "finite": bool(host_metrics.finite),
                "learning_rate": float(
                    config.learning_rate
                    if isinstance(learning_rate, (int, float))
                    else jax.device_get(learning_rate(state.step - 1))
                ),
            }
            if config.terminal_score_clip_norm is not None:
                record.update(
                    {
                        "target_score_raw_norm_mean": score_raw_norm_sum
                        / max(score_bead_count, 1),
                        "target_score_raw_norm_max": score_raw_norm_max,
                        "target_score_clipped_norm_max": score_clipped_norm_max,
                        "target_score_clip_fraction": score_clipped_count
                        / max(score_bead_count, 1),
                    }
                )
            step = int(record["step"])
            should_retain = (
                step == 1
                or step % config.history_every == 0
                or update_in_outer + 1 == config.updates_per_outer
                or step == total_updates
                or not record["finite"]
            )
            if should_retain:
                history.append(record)
            should_report = (
                step == 1
                or step % config.progress_every == 0
                or update_in_outer + 1 == config.updates_per_outer
                or step == total_updates
                or not record["finite"]
            )
            if should_report:
                progress = 100.0 * step / total_updates
                clip_message = ""
                if config.terminal_score_clip_norm is not None:
                    clip_message = (
                        f" score_raw_max={record['target_score_raw_norm_max']:.6e} "
                        f"score_clip_frac={record['target_score_clip_fraction']:.4f}"
                    )
                print(
                    f"[forward] outer={outer + 1}/{config.outer_iterations} "
                    f"step={step}/{total_updates} ({progress:6.2f}%) "
                    f"loss={record['loss']:.6e} matching={record['matching']:.6e} "
                    f"damping={record['damping']:.6e} "
                    f"grad_norm={record['gradient_norm']:.6e} "
                    f"lr={record['learning_rate']:.6e} "
                    f"valid={record['replay_valid_fraction']:.4f} "
                    f"finite={record['finite']}"
                    f"{clip_message}",
                    flush=True,
                )
            if not record["finite"]:
                raise FloatingPointError("non-finite forward BMS update")
        if outer_callback is not None:
            outer_callback(
                outer + 1,
                ForwardOuterSnapshot(
                    state=state,
                    replay_buffer=replay,
                ),
            )
    return ForwardTrainingResult(state=state, replay_buffer=replay, metrics=tuple(history))
