"""Generate proposal-only samples from the stochastic forward BMS SDE."""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import monotonic
from typing import Any

import jax
import numpy as np
from omegaconf import DictConfig

from cg_bms_jax.experiment.common import (
    load_controller_checkpoint,
    load_controller_initializer_checkpoint,
    run_hydra_entry,
)
from cg_bms_jax.process import euler_maruyama_rollout
from cg_bms_jax.runtime import build_runtime_system, resolve_project_path


def run(config: DictConfig) -> Path:
    seed = int(config.seed)
    system = build_runtime_system(config, key=jax.random.PRNGKey(seed))
    controller_kind = str(config.get("controller_kind", "forward")).lower()
    checkpoint_path = resolve_project_path(str(config.forward_checkpoint))
    if controller_kind == "forward":
        forward = load_controller_checkpoint(
            checkpoint_path,
            expected_role="forward",
            system=system,
        )
        sampler_kind = "sde"
    elif controller_kind == "forward_pretrain":
        forward = load_controller_initializer_checkpoint(
            checkpoint_path,
            system=system,
        )
        if forward.metadata.role != "forward_pretrain":
            raise ValueError(
                "controller_kind=forward_pretrain requires a forward_pretrain checkpoint"
            )
        sampler_kind = "sde_pretrain"
    else:
        raise ValueError(
            "controller_kind must be 'forward' or 'forward_pretrain'"
        )
    num_samples = int(config.num_samples)
    batch_size = int(config.batch_size)
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive")
    progress_every_batches = int(config.get("progress_every_batches", 10))
    if progress_every_batches <= 0:
        raise ValueError("progress_every_batches must be positive")
    steps = int(config.experiment.sde.steps)

    def rollout_batch(key: jax.Array) -> jax.Array:
        source_key, rollout_key = jax.random.split(key)
        initial = system.source.sample(source_key, batch_size)

        def control_apply(_unused: None, time: jax.Array, state: jax.Array) -> jax.Array:
            return system.apply(forward.variables, time, state)

        return euler_maruyama_rollout(
            rollout_key,
            initial,
            sde=system.sde,
            control_apply=control_apply,
            control_params=None,
            steps=steps,
        ).final

    compiled_rollout = jax.jit(rollout_batch)
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), math.ceil(num_samples / batch_size))
    state_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    component_parts: dict[str, list[np.ndarray]] = {}
    valid_parts: list[np.ndarray] = []
    started_at = monotonic()
    for index, key in enumerate(keys):
        take = min(batch_size, num_samples - index * batch_size)
        state = compiled_rollout(key)[:take]
        target = system.evaluate_target(state, include_training_wall=False)
        state_parts.append(np.asarray(jax.device_get(state)))
        energy_parts.append(np.asarray(jax.device_get(target.energy)))
        for name, value in dict(getattr(target, "components", {})).items():
            component_parts.setdefault(name, []).append(
                np.asarray(jax.device_get(value))
            )
        valid = target.valid_mask & system.support_mask(state)
        valid_parts.append(np.asarray(jax.device_get(valid), dtype=bool))
        batch_number = index + 1
        processed = min(batch_number * batch_size, num_samples)
        if (
            batch_number == 1
            or batch_number % progress_every_batches == 0
            or batch_number == len(keys)
        ):
            elapsed = monotonic() - started_at
            rate = processed / max(elapsed, 1.0e-9)
            eta = (num_samples - processed) / max(rate, 1.0e-9)
            print(
                f"[sde] batch={batch_number}/{len(keys)} "
                f"samples={processed}/{num_samples} "
                f"({100.0 * processed / num_samples:6.2f}%) "
                f"elapsed={elapsed:.1f}s rate={rate:.1f} samples/s eta={eta:.1f}s",
                flush=True,
            )

    standardized = np.concatenate(state_parts, axis=0)
    physical = np.asarray(jax.device_get(system.to_physical(standardized)))
    metadata = {
        **dict(system.identity),
        "sampler_kind": sampler_kind,
        "controller_kind": controller_kind,
        "target_terms": list(system.identity.get("target_terms", ["pmf"])),
        "forward_checkpoint_sha256": forward.digest,
        "forward_metadata": forward.metadata.to_dict(),
        "rollout_steps": steps,
        "seed": seed,
    }
    payload: dict[str, Any] = {
        "R": physical,
        "U": np.concatenate(energy_parts, axis=0),
        "U_target": np.concatenate(energy_parts, axis=0),
        "valid_mask": np.concatenate(valid_parts, axis=0),
        "sampler_kind": np.asarray(sampler_kind),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    }
    if (
        system.transform is not None
        or getattr(system, "physical_map", None) is not None
    ):
        payload["X_standardized"] = standardized
    for name, parts in component_parts.items():
        payload[name] = np.concatenate(parts, axis=0)
    # Intentionally no logp/logq/logw field: an SDE path has no attached
    # deterministic change-of-variables likelihood in this workflow.
    output = resolve_project_path(str(config.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


def main() -> None:
    result = run_hydra_entry("sample_sde", run)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
