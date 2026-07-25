"""Backward-controller matching command."""

from __future__ import annotations

from typing import Any

import jax
from omegaconf import DictConfig

from cg_bms_jax.experiment.common import (
    load_forward_controller_for_kind,
    run_hydra_entry,
)
from cg_bms_jax.experiment.training_support import save_final_training_result, training_sizes
from cg_bms_jax.runtime import build_runtime_system, resolve_project_path
from cg_bms_jax.training import (
    BackwardTrainingConfig,
    flax_apply,
    split_flax_variables,
    train_backward_score,
)


def run(config: DictConfig) -> Any:
    if config.resume is not None:
        raise NotImplementedError(
            "Backward resume must restore optimizer state; use a fresh output or add resume support"
        )
    # Reverse-score matching needs forward rollouts but never evaluates the PMF.
    # The factory still reads pinned Ala2 geometry metadata, while avoiding an
    # unnecessary MACE compile in this stage.
    system = build_runtime_system(
        config,
        key=jax.random.PRNGKey(int(config.seed)),
        load_potential=False,
    )
    forward = load_forward_controller_for_kind(
        resolve_project_path(str(config.forward_checkpoint)),
        controller_kind=str(config.get("forward_controller_kind", "forward")),
        system=system,
    )
    initial_params, initial_constants = split_flax_variables(system.initial_variables)
    forward_params, forward_constants = split_flax_variables(forward.variables)
    training = config.experiment.training
    rollout_batch_size, rollout_batches = training_sizes(dict(training))
    configured_updates = config.get("updates")
    updates = (
        int(training.outer_iterations) * int(training.gradient_steps)
        if configured_updates is None
        else int(configured_updates)
    )
    if updates <= 0:
        raise ValueError("updates must be positive when explicitly configured")
    loop = BackwardTrainingConfig(
        rollout_batches=rollout_batches,
        rollout_batch_size=rollout_batch_size,
        rollout_steps=int(config.experiment.sde.steps),
        updates=updates,
        minibatch_size=int(training.batch_size),
        replay_capacity=int(training.buffer_capacity),
        learning_rate=float(training.learning_rate),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        time_epsilon=float(training.get("time_epsilon", 1.0e-3)),
        zero_last_noise=bool(training.get("zero_last_noise", False)),
        progress_every=int(config.get("progress_every", 100)),
        seed=int(config.seed),
    )
    control_apply = flax_apply(system.controller.apply)
    result = train_backward_score(
        initial_params=initial_params,
        constants=initial_constants,
        backward_apply=control_apply,
        forward_params=forward_params,
        forward_constants=forward_constants,
        forward_apply=control_apply,
        source=system.source,
        sde=system.sde,
        config=loop,
    )
    return save_final_training_result(
        config=config,
        system=system,
        result=result,
        role="backward",
        parent_forward_sha256=forward.digest,
        warmstart_data_sha256=forward.metadata.warmstart_data_sha256,
    )


def main() -> None:
    result = run_hydra_entry("train_backward", run)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
