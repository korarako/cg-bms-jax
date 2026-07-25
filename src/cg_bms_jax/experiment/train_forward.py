"""Forward fixed-point BMS training command."""

from __future__ import annotations

from typing import Any

import jax
from omegaconf import DictConfig

from cg_bms_jax.evaluation import (
    load_evaluation_archive,
    write_ala2_replay_visualization,
)
from cg_bms_jax.experiment.common import (
    load_controller_initializer_checkpoint,
    run_hydra_entry,
)
from cg_bms_jax.experiment.training_support import (
    save_final_training_result,
    save_intermediate_training_checkpoint,
    training_sizes,
)
from cg_bms_jax.runtime import build_runtime_system, resolve_project_path
from cg_bms_jax.training import (
    ForwardTrainingConfig,
    flax_apply,
    split_flax_variables,
    train_forward_fixed_point,
)


def run(config: DictConfig) -> Any:
    initialize_controller_from = config.get("initialize_controller_from")
    if config.resume is not None and initialize_controller_from is not None:
        raise ValueError("resume and initialize_controller_from are mutually exclusive")
    if config.resume is not None:
        raise NotImplementedError(
            "Forward resume must restore optimizer state; use a fresh output or add resume support"
        )
    system = build_runtime_system(config, key=jax.random.PRNGKey(int(config.seed)))
    initializer = None
    if initialize_controller_from is None:
        params, constants = split_flax_variables(system.initial_variables)
    else:
        initializer = load_controller_initializer_checkpoint(
            resolve_project_path(str(initialize_controller_from)),
            system=system,
        )
        params, constants = split_flax_variables(initializer.variables)
    training = config.experiment.training
    rollout_batch_size, rollout_batches = training_sizes(dict(training))
    terminal_score_clip_norm = training.get("terminal_score_clip_norm")
    loop = ForwardTrainingConfig(
        outer_iterations=int(training.outer_iterations),
        rollout_batches_per_outer=rollout_batches,
        rollout_batch_size=rollout_batch_size,
        rollout_steps=int(config.experiment.sde.steps),
        updates_per_outer=int(training.gradient_steps),
        minibatch_size=int(training.batch_size),
        replay_capacity=int(training.buffer_capacity),
        learning_rate=float(training.learning_rate),
        final_learning_rate=(
            None
            if training.get("final_learning_rate") is None
            else float(training.final_learning_rate)
        ),
        learning_rate_warmup_updates=int(
            training.get("learning_rate_warmup_updates", 0)
        ),
        learning_rate_decay_updates=(
            None
            if training.get("learning_rate_decay_updates") is None
            else int(training.learning_rate_decay_updates)
        ),
        learning_rate_warmup_start_factor=float(
            training.get("learning_rate_warmup_start_factor", 1.0e-3)
        ),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
        terminal_score_clip_norm=(
            None if terminal_score_clip_norm is None else float(terminal_score_clip_norm)
        ),
        terminal_score_clip_preserve_mean=bool(
            training.get("terminal_score_clip_preserve_mean", False)
        ),
        terminal_score_clip_mean_mode=str(
            training.get("terminal_score_clip_mean_mode", "target_mean")
        ),
        weight_decay=float(training.get("weight_decay", 0.0)),
        damping=float(training.damping),
        previous_model_interval_outer=int(
            training.get("previous_model_interval_outer", 1)
        ),
        time_epsilon=float(training.get("time_epsilon", 1.0e-3)),
        zero_last_noise=bool(training.get("zero_last_noise", False)),
        progress_every=int(config.get("progress_every", 100)),
        history_every=int(config.get("history_every", 1)),
        seed=int(config.seed),
    )
    control_apply = flax_apply(system.controller.apply)

    def target_score(state):
        result = system.evaluate_target(state, include_training_wall=True)
        if loop.terminal_score_clip_mean_mode == "analytic_standard_normal":
            if result.score_decomposition is None:
                raise ValueError(
                    "analytic_standard_normal clipping requires the potential "
                    "to provide an explicit score decomposition"
                )
            return result.score_decomposition
        return result.score

    def optional_interval(name: str) -> int | None:
        raw = config.get(name)
        if raw is None:
            return None
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{name} must be positive when enabled")
        return value

    checkpoint_interval = optional_interval("save_checkpoint_interval_outer")
    save_vis_interval = optional_interval("save_vis_interval_outer")
    output_root = resolve_project_path(str(config.output_dir))
    target_reference = None
    implicit_reference = None
    if save_vis_interval is not None:
        if str(system.experiment["coordinate_mode"]) != "cgbg_ambient18":
            raise ValueError("save_vis_interval_outer currently supports Ala2 only")
        assets = system.experiment["assets"]
        target_reference = load_evaluation_archive(
            resolve_project_path(str(assets["reference"]))
        )
        implicit_value = assets.get("implicit_reference")
        if (
            bool(config.get("save_vis_include_implicit_reference", False))
            and implicit_value is not None
        ):
            implicit_path = resolve_project_path(str(implicit_value))
            if implicit_path.is_file():
                implicit_reference = load_evaluation_archive(implicit_path)

    checkpoint_kwargs = {
        "warmstart_data_sha256": (
            None if initializer is None else initializer.metadata.warmstart_data_sha256
        ),
        "initial_controller_path": None if initializer is None else str(initializer.path),
        "initial_controller_sha256": None if initializer is None else initializer.digest,
        "initial_controller_role": (
            None if initializer is None else initializer.metadata.role
        ),
    }

    def save_intermediate(state: Any) -> Any:
        path = save_intermediate_training_checkpoint(
            config=config,
            system=system,
            state=state,
            role="forward",
            **checkpoint_kwargs,
        )
        print(f"[forward] checkpoint={path}", flush=True)
        return path

    def save_visualization(completed_outer: int, snapshot: Any) -> None:
        step = int(jax.device_get(snapshot.state.step))
        destination = (
            output_root
            / "training_replay_eval"
            / f"outer_{completed_outer:06d}_step_{step:08d}"
        )
        path = write_ala2_replay_visualization(
            system=system,
            replay=snapshot.replay_buffer,
            target_reference=target_reference,
            implicit_reference=implicit_reference,
            output_dir=destination,
            completed_outer=completed_outer,
            step=step,
            max_samples=int(config.get("save_vis_max_samples", 2048)),
            target_batch_size=int(config.get("save_vis_target_batch_size", 64)),
            n_bootstraps=int(config.get("save_vis_n_bootstraps", 1)),
            seed=int(config.get("save_vis_seed", 0)),
        )
        print(f"[forward] replay_visualization={path}", flush=True)

    def on_outer(completed_outer: int, snapshot: Any) -> None:
        # The final checkpoint is written once by save_final_training_result.
        if completed_outer == loop.outer_iterations:
            return
        if (
            checkpoint_interval is not None
            and completed_outer % checkpoint_interval == 0
        ):
            save_intermediate(snapshot.state)
        if save_vis_interval is not None and completed_outer % save_vis_interval == 0:
            save_visualization(completed_outer, snapshot)

    result = train_forward_fixed_point(
        initial_params=params,
        constants=constants,
        control_apply=control_apply,
        source=system.source,
        sde=system.sde,
        target_score_fn=target_score,
        config=loop,
        outer_callback=(
            on_outer
            if checkpoint_interval is not None or save_vis_interval is not None
            else None
        ),
    )
    final_path = save_final_training_result(
        config=config,
        system=system,
        result=result,
        role="forward",
        **checkpoint_kwargs,
    )
    if (
        save_vis_interval is not None
        and loop.outer_iterations % save_vis_interval == 0
    ):
        save_visualization(loop.outer_iterations, result)
    return final_path


def main() -> None:
    result = run_hydra_entry("train_forward", run)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
