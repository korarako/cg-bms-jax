"""Validate integrated PF likelihood against a direct flow-map Jacobian."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from cg_bms_jax.evaluation import compare_integrated_logq, direct_flow_map_logq
from cg_bms_jax.experiment.common import (
    load_controller_checkpoint,
    load_forward_controller_for_kind,
    require_forward_backward_compatible,
    run_hydra_entry,
)
from cg_bms_jax.process import (
    ProbabilityFlowConfig,
    bms_probability_flow_velocity,
    integrate_flow_map,
    integrate_probability_flow,
)
from cg_bms_jax.runtime import build_runtime_system, resolve_project_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(config: DictConfig) -> Path:
    seed = int(config.seed)
    system = build_runtime_system(
        config,
        key=jax.random.PRNGKey(seed),
        load_potential=False,
    )
    if int(np.prod(system.event_shape)) > 4:
        raise ValueError(
            "Direct dense-Jacobian validation is restricted to at most four "
            "event dimensions"
        )
    forward_kind = str(config.get("forward_controller_kind", "forward"))
    forward = load_forward_controller_for_kind(
        resolve_project_path(str(config.forward_checkpoint)),
        controller_kind=forward_kind,
        system=system,
    )
    backward = load_controller_checkpoint(
        resolve_project_path(str(config.backward_checkpoint)),
        expected_role="backward",
        system=system,
    )
    require_forward_backward_compatible(
        forward,
        backward,
        forward_controller_kind=forward_kind,
    )
    likelihood = config.likelihood
    if str(likelihood.solver).lower() != "dopri5":
        raise ValueError("PF validation requires solver=dopri5")
    flow_config = ProbabilityFlowConfig(
        dt0=float(likelihood.dt0),
        rtol=float(likelihood.rtol),
        atol=float(likelihood.atol),
        max_steps=int(likelihood.max_steps),
    )
    num_samples = int(config.num_samples)
    if not 0 < num_samples <= 64:
        raise ValueError("num_samples must lie in [1,64] for direct Jacobians")

    def velocity(time: jax.Array, state: jax.Array) -> jax.Array:
        def forward_apply(
            _params: None,
            current_time: jax.Array,
            current_state: jax.Array,
        ) -> jax.Array:
            return system.apply(forward.variables, current_time, current_state)

        def backward_apply(
            _params: None,
            current_time: jax.Array,
            current_state: jax.Array,
        ) -> jax.Array:
            return system.apply(backward.variables, current_time, current_state)

        return bms_probability_flow_velocity(
            system.sde,
            forward_apply,
            None,
            backward_apply,
            None,
            time,
            state,
        )

    initial = system.source.sample(
        jax.random.PRNGKey(seed + 1),
        num_samples,
        dtype=jnp.float32,
    )
    initial_logq = system.source.log_prob(initial)
    coupled = integrate_probability_flow(
        velocity,
        initial,
        initial_logq,
        config=flow_config,
    )

    def single_flow_map(single_state: jax.Array) -> jax.Array:
        solved = integrate_flow_map(
            velocity,
            single_state[None, ...],
            config=flow_config,
        )
        return solved.samples[0]

    direct = direct_flow_map_logq(single_flow_map, initial, initial_logq)
    coordinate_error = np.asarray(coupled.samples) - np.asarray(
        direct.terminal_states
    )
    comparison = compare_integrated_logq(
        coupled.log_prob,
        direct.logq,
        atol=float(config.comparison.logq_atol),
        rtol=float(config.comparison.logq_rtol),
    )
    metrics = {
        "schema_version": 1,
        "experiment_family": system.identity["experiment_family"],
        "forward_checkpoint_sha256": forward.digest,
        "backward_checkpoint_sha256": backward.digest,
        "seed": seed,
        "num_samples": num_samples,
        "event_shape": list(system.event_shape),
        "likelihood": {
            "solver": "dopri5",
            "dt0": flow_config.dt0,
            "rtol": flow_config.rtol,
            "atol": flow_config.atol,
            "max_steps": flow_config.max_steps,
        },
        "coordinate_path": {
            "rmse": float(np.sqrt(np.mean(coordinate_error**2))),
            "max_abs": float(np.max(np.abs(coordinate_error))),
        },
        "logq": comparison,
        "jacobian": {
            "minimum_log_abs_det": float(
                np.min(np.asarray(direct.log_abs_det_jacobians))
            ),
            "maximum_log_abs_det": float(
                np.max(np.asarray(direct.log_abs_det_jacobians))
            ),
            "orientation_signs": sorted(
                {
                    int(value)
                    for value in np.asarray(direct.jacobian_signs).reshape(-1)
                }
            ),
        },
    }
    output = resolve_project_path(str(config.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "pf_jacobian_validation.npz",
        initial_X=np.asarray(initial),
        initial_logq=np.asarray(initial_logq),
        integrated_X=np.asarray(coupled.samples),
        integrated_logq=np.asarray(coupled.log_prob),
        direct_X=np.asarray(direct.terminal_states),
        direct_logq=np.asarray(direct.logq),
        jacobians=np.asarray(direct.jacobians),
        jacobian_signs=np.asarray(direct.jacobian_signs),
        log_abs_det_jacobians=np.asarray(direct.log_abs_det_jacobians),
    )
    metrics_path = output / "pf_jacobian_validation.json"
    metrics_path.write_text(
        json.dumps(
            _json_safe(metrics),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return metrics_path


def main() -> None:
    result = run_hydra_entry("validate_pf", run)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()

