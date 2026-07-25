"""Coupled PF-ODE likelihood sampling followed by PMF reweighting."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from cg_bms_jax.experiment.common import (
    load_controller_checkpoint,
    load_forward_controller_for_kind,
    require_forward_backward_compatible,
    run_hydra_entry,
)
from cg_bms_jax.process import (
    ProbabilityFlowConfig,
    bms_probability_flow_velocity,
    sample_probability_flow,
)
from cg_bms_jax.reweight import (
    build_reweight_payload,
    compute_cgbg_compat_weights,
    compute_exact_ambient_weights,
    save_reweight_archive,
)
from cg_bms_jax.runtime import build_runtime_system, resolve_project_path


@dataclass(frozen=True)
class _BatchedTargetEvaluation:
    energy: np.ndarray
    reduced_energy: np.ndarray
    valid_mask: np.ndarray
    support_mask: np.ndarray
    physical_coordinates: np.ndarray
    components: dict[str, np.ndarray]


def _evaluate_target_in_batches(
    system: Any,
    state: np.ndarray,
    *,
    batch_size: int,
    progress_every_batches: int,
) -> _BatchedTargetEvaluation:
    """Evaluate the potentially large MACE target without one giant XLA batch."""

    values = np.asarray(state)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("state must contain a non-empty sample dimension")
    if batch_size <= 0 or progress_every_batches <= 0:
        raise ValueError("target batch and progress sizes must be positive")

    total_samples = int(values.shape[0])
    total_batches = math.ceil(total_samples / batch_size)
    energies: list[np.ndarray] = []
    reduced_energies: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    support_masks: list[np.ndarray] = []
    physical_coordinates: list[np.ndarray] = []
    component_batches: dict[str, list[np.ndarray]] = {}
    component_names: tuple[str, ...] | None = None
    started_at = monotonic()

    for index, start in enumerate(range(0, total_samples, batch_size)):
        stop = min(start + batch_size, total_samples)
        chunk = jnp.asarray(values[start:stop])
        target = system.evaluate_target(chunk, include_training_wall=False)
        support = system.support_mask(chunk)
        physical = system.to_physical(chunk)

        energy = np.asarray(jax.device_get(target.energy))
        reduced = np.asarray(jax.device_get(target.reduced_energy))
        valid = np.asarray(jax.device_get(target.valid_mask), dtype=bool)
        support_host = np.asarray(jax.device_get(support), dtype=bool)
        physical_host = np.asarray(jax.device_get(physical))
        expected_shape = (stop - start,)
        for name, array in (
            ("energy", energy),
            ("reduced_energy", reduced),
            ("valid_mask", valid),
            ("support_mask", support_host),
        ):
            if array.shape != expected_shape:
                raise ValueError(
                    f"Batched target {name} shape {array.shape} does not match {expected_shape}"
                )
        if physical_host.shape[0] != stop - start:
            raise ValueError("Batched physical coordinates lost the sample dimension")

        components = {
            str(name): np.asarray(jax.device_get(value))
            for name, value in dict(getattr(target, "components", {})).items()
        }
        names = tuple(sorted(components))
        if component_names is None:
            component_names = names
            component_batches = {name: [] for name in names}
        elif names != component_names:
            raise ValueError("Target component fields changed between evaluation batches")
        for name in names:
            if components[name].shape != expected_shape:
                raise ValueError(
                    f"Target component {name!r} shape {components[name].shape} "
                    f"does not match {expected_shape}"
                )
            component_batches[name].append(components[name])

        energies.append(energy)
        reduced_energies.append(reduced)
        valid_masks.append(valid)
        support_masks.append(support_host)
        physical_coordinates.append(physical_host)

        batch_number = index + 1
        if (
            batch_number == 1
            or batch_number % progress_every_batches == 0
            or batch_number == total_batches
        ):
            elapsed = max(monotonic() - started_at, 1.0e-9)
            processed = stop
            rate = processed / elapsed
            eta = (total_samples - processed) / max(rate, 1.0e-9)
            print(
                f"[target] batch={batch_number}/{total_batches} "
                f"samples={processed}/{total_samples} "
                f"({100.0 * processed / total_samples:6.2f}%) "
                f"elapsed={elapsed:.1f}s rate={rate:.1f} samples/s eta={eta:.1f}s",
                flush=True,
            )

    return _BatchedTargetEvaluation(
        energy=np.concatenate(energies, axis=0),
        reduced_energy=np.concatenate(reduced_energies, axis=0),
        valid_mask=np.concatenate(valid_masks, axis=0),
        support_mask=np.concatenate(support_masks, axis=0),
        physical_coordinates=np.concatenate(physical_coordinates, axis=0),
        components={
            name: np.concatenate(batches, axis=0)
            for name, batches in component_batches.items()
        },
    )


def _pf_stage_metadata(
    *,
    forward_sha256: str,
    backward_sha256: str,
    seed: int,
    num_samples: int,
    batch_size: int,
    flow_config: ProbabilityFlowConfig,
    sampling_total_samples: int | None = None,
    sampling_batch_offset: int = 0,
) -> dict[str, Any]:
    metadata = {
        "schema_version": 1,
        "forward_sha256": forward_sha256,
        "backward_sha256": backward_sha256,
        "seed": seed,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "t0": flow_config.t0,
        "t1": flow_config.t1,
        "dt0": flow_config.dt0,
        "rtol": flow_config.rtol,
        "atol": flow_config.atol,
        "max_steps": flow_config.max_steps,
        "divergence": "exact",
    }
    if sampling_total_samples is not None:
        metadata["deterministic_shard"] = {
            "global_num_samples": int(sampling_total_samples),
            "batch_offset": int(sampling_batch_offset),
        }
    return metadata


def _save_pf_stage(
    path: Path,
    *,
    state: np.ndarray,
    logq: np.ndarray,
    initial_state: np.ndarray,
    initial_logq: np.ndarray,
    step_counts: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Atomically preserve completed PF trajectories before target evaluation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X_standardized=np.asarray(state),
            logq_ambient=np.asarray(logq),
            initial_X=np.asarray(initial_state),
            initial_logq=np.asarray(initial_logq),
            ode_steps_per_batch=np.asarray(step_counts, dtype=np.int64),
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
        )
    temporary.replace(path)


def _load_pf_stage(
    path: Path,
    *,
    expected_metadata: dict[str, Any],
    event_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).reshape(()).item()))
        if metadata != expected_metadata:
            raise ValueError("PF stage metadata does not match the requested likelihood run")
        state = np.asarray(archive["X_standardized"])
        logq = np.asarray(archive["logq_ambient"])
        initial_state = np.asarray(archive["initial_X"])
        initial_logq = np.asarray(archive["initial_logq"])
        step_counts = np.asarray(archive["ode_steps_per_batch"], dtype=np.int64)
    expected_state_shape = (int(expected_metadata["num_samples"]), *event_shape)
    if state.shape != expected_state_shape or initial_state.shape != expected_state_shape:
        raise ValueError("PF stage coordinate shape does not match the requested run")
    expected_density_shape = (int(expected_metadata["num_samples"]),)
    if logq.shape != expected_density_shape or initial_logq.shape != expected_density_shape:
        raise ValueError("PF stage log-density shape does not match the requested run")
    if not all(
        np.isfinite(value).all()
        for value in (state, logq, initial_state, initial_logq)
    ):
        raise FloatingPointError("PF stage contains non-finite coordinates or densities")
    return state, logq, initial_state, initial_logq, step_counts


def _progress_message(
    *,
    batch_number: int,
    total_batches: int,
    processed_samples: int,
    total_samples: int,
    ode_steps: int,
    elapsed_seconds: float,
) -> str:
    """Format one host-side PF-ODE progress update."""

    elapsed = max(float(elapsed_seconds), 1.0e-9)
    rate = processed_samples / elapsed
    remaining = max(total_samples - processed_samples, 0)
    eta = remaining / max(rate, 1.0e-9)
    percent = 100.0 * processed_samples / total_samples
    return (
        f"[pf-ode] batch={batch_number}/{total_batches} "
        f"samples={processed_samples}/{total_samples} ({percent:6.2f}%) "
        f"ode_steps={ode_steps} elapsed={elapsed:.1f}s "
        f"rate={rate:.1f} samples/s eta={eta:.1f}s"
    )


def run(config: DictConfig) -> Path:
    seed = int(config.seed)
    system = build_runtime_system(config, key=jax.random.PRNGKey(seed))
    forward_controller_kind = str(
        config.get("forward_controller_kind", "forward")
    ).lower()
    forward = load_forward_controller_for_kind(
        resolve_project_path(str(config.forward_checkpoint)),
        controller_kind=forward_controller_kind,
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
        forward_controller_kind=forward_controller_kind,
    )

    num_samples = int(config.num_samples)
    batch_size = int(config.batch_size)
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive")
    configured_sampling_total = config.get("sampling_total_samples")
    sampling_total_samples = (
        num_samples
        if configured_sampling_total is None
        else int(configured_sampling_total)
    )
    sampling_batch_offset = int(config.get("sampling_batch_offset", 0))
    if sampling_total_samples <= 0:
        raise ValueError("sampling_total_samples must be positive")
    if sampling_batch_offset < 0:
        raise ValueError("sampling_batch_offset must be non-negative")
    global_total_batches = math.ceil(sampling_total_samples / batch_size)
    total_batches = math.ceil(num_samples / batch_size)
    if sampling_batch_offset + total_batches > global_total_batches:
        raise ValueError(
            "The requested deterministic shard exceeds the global PF key schedule"
        )
    global_start = sampling_batch_offset * batch_size
    expected_local_samples = min(
        total_batches * batch_size,
        sampling_total_samples - global_start,
    )
    if expected_local_samples != num_samples:
        raise ValueError(
            "num_samples does not match the contiguous deterministic shard: "
            f"expected {expected_local_samples}, found {num_samples}"
        )
    progress_every_batches = int(config.get("progress_every_batches", 10))
    if progress_every_batches <= 0:
        raise ValueError("progress_every_batches must be positive")
    likelihood = config.likelihood
    if str(likelihood.solver).lower() != "dopri5":
        raise ValueError("The exact-likelihood workflow currently requires solver=dopri5")
    if str(likelihood.divergence).lower() != "exact":
        raise ValueError("Formal reweighting requires the exact full-rank divergence")
    flow_config = ProbabilityFlowConfig(
        dt0=float(likelihood.get("dt0", 1.0e-3)),
        rtol=float(likelihood.rtol),
        atol=float(likelihood.atol),
        max_steps=int(likelihood.max_steps),
    )
    output = resolve_project_path(str(config.output))
    stage_path = output.with_name(f"{output.stem}.pf_stage.npz")
    stage_metadata = _pf_stage_metadata(
        forward_sha256=forward.digest,
        backward_sha256=backward.digest,
        seed=seed,
        num_samples=num_samples,
        batch_size=batch_size,
        flow_config=flow_config,
        sampling_total_samples=(
            None
            if configured_sampling_total is None
            else sampling_total_samples
        ),
        sampling_batch_offset=sampling_batch_offset,
    )

    def velocity(time: jax.Array, state: jax.Array) -> jax.Array:
        def forward_apply(_params: None, current_time: jax.Array, current_state: jax.Array) -> jax.Array:
            return system.apply(forward.variables, current_time, current_state)

        def backward_apply(_params: None, current_time: jax.Array, current_state: jax.Array) -> jax.Array:
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

    # A single Diffrax state contains both X and log(q).  No later pass can
    # accidentally attach a likelihood to a different numerical trajectory.
    def solve_batch(key: jax.Array):
        return sample_probability_flow(
            key,
            system.source,
            batch_size,
            velocity,
            config=flow_config,
        )

    if bool(config.get("reuse_pf_stage", True)) and stage_path.is_file():
        state, logq, initial_state, initial_logq, step_counts_array = _load_pf_stage(
            stage_path,
            expected_metadata=stage_metadata,
            event_shape=tuple(system.event_shape),
        )
        print(f"[pf-ode] reusing verified stage {stage_path}", flush=True)
    else:
        compiled_solve = jax.jit(solve_batch)
        global_keys = jax.random.split(
            jax.random.PRNGKey(seed + 1),
            global_total_batches,
        )
        keys = global_keys[
            sampling_batch_offset : sampling_batch_offset + total_batches
        ]
        states: list[np.ndarray] = []
        logqs: list[np.ndarray] = []
        initials: list[np.ndarray] = []
        initial_logqs: list[np.ndarray] = []
        step_counts: list[int] = []
        started_at = monotonic()
        for index, key in enumerate(keys):
            take = min(batch_size, num_samples - index * batch_size)
            result = compiled_solve(key)
            states.append(np.asarray(jax.device_get(result.samples[:take])))
            logqs.append(np.asarray(jax.device_get(result.log_prob[:take])))
            initials.append(np.asarray(jax.device_get(result.initial_samples[:take])))
            initial_logqs.append(
                np.asarray(jax.device_get(result.initial_log_prob[:take]))
            )
            ode_steps = int(jax.device_get(result.num_steps))
            step_counts.append(ode_steps)
            batch_number = index + 1
            processed_samples = min(batch_number * batch_size, num_samples)
            if (
                batch_number == 1
                or batch_number % progress_every_batches == 0
                or batch_number == total_batches
            ):
                print(
                    _progress_message(
                        batch_number=batch_number,
                        total_batches=total_batches,
                        processed_samples=processed_samples,
                        total_samples=num_samples,
                        ode_steps=ode_steps,
                        elapsed_seconds=monotonic() - started_at,
                    ),
                    flush=True,
                )
        state = np.concatenate(states, axis=0)
        logq = np.concatenate(logqs, axis=0)
        initial_state = np.concatenate(initials, axis=0)
        initial_logq = np.concatenate(initial_logqs, axis=0)
        step_counts_array = np.asarray(step_counts, dtype=np.int64)
        _save_pf_stage(
            stage_path,
            state=state,
            logq=logq,
            initial_state=initial_state,
            initial_logq=initial_logq,
            step_counts=step_counts_array,
            metadata=stage_metadata,
        )
        print(f"[pf-ode] saved verified stage {stage_path}", flush=True)

    if not bool(np.isfinite(state).all()):
        raise FloatingPointError("PF-ODE produced non-finite terminal coordinates")
    if not bool(np.isfinite(logq).all()):
        raise FloatingPointError("PF-ODE produced non-finite terminal logq")
    if step_counts_array.shape != (total_batches,):
        raise ValueError("PF-ODE stage has an invalid number of batch step counts")
    if np.any(step_counts_array <= 0) or np.any(
        step_counts_array > flow_config.max_steps
    ):
        raise RuntimeError(
            "PF-ODE reported invalid step counts "
            f"{step_counts_array.tolist()} for max_steps={flow_config.max_steps}"
        )
    configured_target_batch_size = config.get("target_batch_size")
    target_batch_size = (
        batch_size
        if configured_target_batch_size is None
        else int(configured_target_batch_size)
    )
    target = _evaluate_target_in_batches(
        system,
        state,
        batch_size=target_batch_size,
        progress_every_batches=progress_every_batches,
    )
    energy_components = dict(getattr(target, "components", {}))
    configured_target_terms = list(system.identity.get("target_terms", ["pmf"]))
    support = target.support_mask
    valid = target.valid_mask & support
    density_mode = str(config.reweight.density_mode).lower()
    if density_mode in {"exact_ambient", "ambient_exact"}:
        weights = compute_exact_ambient_weights(
            target.reduced_energy,
            logq,
            valid_mask=valid,
        )
        if (
            str(system.identity.get("density_mode"))
            == "ambient_66d_aux_com_exact"
        ):
            # The AA identity already names its exact Gaussian COM target.
            target_terms = configured_target_terms
        else:
            target_terms = configured_target_terms + (
                ["auxiliary_cartesian_com"]
                if system.transform is not None
                else []
            )
    elif density_mode == "cgbg_compat":
        if str(system.identity.get("density_mode")) == "ambient_66d_aux_com_exact":
            raise ValueError(
                "All-atom Ala2 is defined on exact Cartesian 66D density; "
                "the CG-BG radial COM correction is not applicable"
            )
        if system.transform is None:
            raise ValueError("cgbg_compat density is only defined for ambient molecular coordinates")
        weights = compute_cgbg_compat_weights(
            target.energy,
            system.potential.kT,
            logq,
            state,
            system.transform.physical_std,
            valid_mask=valid,
        )
        target_terms = configured_target_terms + ["cgbg_radial_com_correction"]
    else:
        raise ValueError(f"Unknown reweight.density_mode: {density_mode!r}")

    metadata = {
        **dict(system.identity),
        "sampler_kind": "pf_ode",
        "forward_controller_kind": forward_controller_kind,
        "state_density_mode": system.identity.get("density_mode"),
        "density_mode": weights.density_mode,
        "target_terms": target_terms,
        "training_wall_in_formal_target": False,
        "hard_support_uses_wall_margin": False,
        "domain": system.domain_metadata(),
        "forward_checkpoint_sha256": forward.digest,
        "backward_checkpoint_sha256": backward.digest,
        "forward_metadata": forward.metadata.to_dict(),
        "backward_metadata": backward.metadata.to_dict(),
        "likelihood": {
            "solver": "dopri5",
            "t0": flow_config.t0,
            "t1": flow_config.t1,
            "dt0": flow_config.dt0,
            "rtol": flow_config.rtol,
            "atol": flow_config.atol,
            "max_steps": flow_config.max_steps,
            "divergence": "exact",
            "dtype": str(state.dtype),
            "target_batch_size": target_batch_size,
            "pf_stage": str(stage_path),
        },
        "seed": seed,
        "sampling": {
            "global_num_samples": sampling_total_samples,
            "batch_size": batch_size,
            "global_num_batches": global_total_batches,
            "batch_offset": sampling_batch_offset,
            "local_num_batches": total_batches,
            "local_num_samples": num_samples,
            "key_schedule": "jax.random.split(PRNGKey(seed+1), global_num_batches)",
        },
    }
    extra = {
        "initial_X": initial_state,
        "initial_logq": initial_logq,
        "ode_steps_per_batch": step_counts_array,
        # U is the complete configured dimensional molecular target.  Every
        # formal component is recorded below.  The exact ambient COM term is
        # reduced/dimensionless and is already included in logw.
        "target_reduced_energy": target.reduced_energy,
        "U_target": target.energy,
    }
    extra.update(energy_components)
    payload = build_reweight_payload(
        physical_coordinates=target.physical_coordinates,
        energy=target.energy,
        logq_ambient=logq,
        result=weights,
        metadata=metadata,
        standardized_ambient=(
            state
            if system.transform is not None
            or getattr(system, "physical_map", None) is not None
            else None
        ),
        support_mask=support,
        extra=extra,
    )
    return save_reweight_archive(output, payload)


def main() -> None:
    result = run_hydra_entry("sample_reweight", run)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
