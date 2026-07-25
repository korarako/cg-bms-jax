"""Shared command-line and checkpoint compatibility helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from cg_bms_jax.checkpoint import CheckpointMetadata, canonical_config_sha256
from cg_bms_jax.runtime import RuntimeSystem, compose_config
from cg_bms_jax.training import (
    initialize_train_state,
    make_learning_rate,
    make_optimizer,
    split_flax_variables,
)


@dataclass(frozen=True)
class ControllerCheckpoint:
    variables: Any
    metadata: CheckpointMetadata
    digest: str
    path: Path


def _experiment_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    experiment = config.get("experiment", config)
    if not isinstance(experiment, Mapping):
        raise TypeError("checkpoint config must contain an experiment mapping")
    return experiment


def _require_controller_tree_compatible(
    restored: Any,
    expected: Any,
    *,
    label: str,
    compare_discrete_values: bool = False,
) -> None:
    restored_with_paths, restored_structure = jax.tree_util.tree_flatten_with_path(restored)
    expected_with_paths, expected_structure = jax.tree_util.tree_flatten_with_path(expected)
    if restored_structure != expected_structure:
        raise ValueError(f"initializer {label} PyTree structure is incompatible")
    for (restored_path, restored_leaf), (expected_path, expected_leaf) in zip(
        restored_with_paths, expected_with_paths, strict=True
    ):
        if restored_path != expected_path:
            raise ValueError(f"initializer {label} PyTree paths are incompatible")
        restored_array = np.asarray(jax.device_get(restored_leaf))
        expected_array = np.asarray(jax.device_get(expected_leaf))
        path = jax.tree_util.keystr(restored_path)
        if restored_array.shape != expected_array.shape:
            raise ValueError(
                f"initializer {label}{path} shape mismatch: "
                f"{restored_array.shape} != {expected_array.shape}"
            )
        if restored_array.dtype != expected_array.dtype:
            raise ValueError(
                f"initializer {label}{path} dtype mismatch: "
                f"{restored_array.dtype} != {expected_array.dtype}"
            )
        if restored_array.dtype.kind in "fc" and not np.all(np.isfinite(restored_array)):
            raise ValueError(f"initializer {label}{path} contains non-finite values")
        if (
            compare_discrete_values
            and restored_array.dtype.kind in "biu"
            and not np.array_equal(restored_array, expected_array)
        ):
            raise ValueError(
                f"initializer {label}{path} changes a discrete model constant"
            )


def run_hydra_entry(config_name: str, callback: Callable[[DictConfig], Any]) -> Any:
    """Compose Hydra overrides from ``sys.argv`` and run one command.

    We intentionally do not let Hydra change the current directory.  Asset and
    output paths are resolved explicitly by the runtime layer, which makes the
    console scripts and direct Python calls behave identically.
    """

    arguments = sys.argv[1:]
    if any(argument in {"-h", "--help"} for argument in arguments):
        config = compose_config(config_name)
        print(OmegaConf.to_yaml(config, resolve=False))
        print("\nOverride values with Hydra syntax, e.g. experiment=mb_cg1d_smoke seed=7")
        return None
    config = compose_config(config_name, arguments)
    return callback(config)


def _path_digest(path: Path) -> str:
    """Content digest used only to validate checkpoint parentage."""

    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _restore_metadata(value: Any) -> CheckpointMetadata:
    if isinstance(value, CheckpointMetadata):
        return value
    if isinstance(value, Mapping):
        return CheckpointMetadata.from_dict(dict(value))
    raise TypeError("checkpoint.io did not return CheckpointMetadata-compatible metadata")


def _restore_variables(payload: Any) -> Any:
    variables = _field(payload, "variables")
    if variables is not None:
        return variables
    params = _field(payload, "params")
    constants = _field(payload, "constants")
    if params is None:
        raise TypeError("checkpoint payload has neither variables nor params")
    # A persistence backend may use the legacy field name `params` for the
    # complete Flax variables mapping.  Preserve it when collections are clear.
    if isinstance(params, Mapping) and "params" in params:
        return params
    if constants is None:
        raise ValueError(
            "Checkpoint contains bare trainable params but no Flax constants. "
            "The BMS Fourier frequencies are part of the learned controller identity."
        )
    variables = {"params": params}
    if isinstance(constants, Mapping):
        variables.update(dict(constants))
    else:
        raise TypeError("checkpoint constants must be a mapping of Flax collections")
    return variables


def load_controller_checkpoint(
    path: str | Path,
    *,
    expected_role: str,
    system: RuntimeSystem,
) -> ControllerCheckpoint:
    """Load through checkpoint.io and enforce role/system identity."""

    checkpoint_io = importlib.import_module("cg_bms_jax.checkpoint.io")
    loader = getattr(checkpoint_io, "restore_checkpoint", None)
    if loader is None:
        raise AttributeError("cg_bms_jax.checkpoint.io must define restore_checkpoint(path)")
    resolved = Path(path).expanduser().resolve()
    # Restore into a freshly initialized tree for the current device topology.
    # Target-free Orbax restore can otherwise reuse sharding metadata from the
    # machine that wrote the checkpoint, which is unsafe across GPU layouts.
    initial_params, initial_constants = split_flax_variables(system.initial_variables)
    training = system.experiment["training"]
    learning_rate: float | Any = float(training["learning_rate"])
    if expected_role == "forward":
        learning_rate = make_learning_rate(
            peak=float(training["learning_rate"]),
            final=(
                None
                if training.get("final_learning_rate") is None
                else float(training["final_learning_rate"])
            ),
            warmup_updates=int(training.get("learning_rate_warmup_updates", 0)),
            decay_updates=(
                None
                if training.get("learning_rate_decay_updates") is None
                else int(training["learning_rate_decay_updates"])
            ),
            total_updates=(
                int(training["outer_iterations"])
                * int(training["gradient_steps"])
            ),
            warmup_start_factor=float(
                training.get("learning_rate_warmup_start_factor", 1.0e-3)
            ),
        )
    optimizer = make_optimizer(
        learning_rate=learning_rate,
        gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    target_state = initialize_train_state(initial_params, initial_constants, optimizer)
    payload = loader(resolved, target_state=target_state)
    metadata = _restore_metadata(_field(payload, "metadata"))
    if metadata.role != expected_role:
        raise ValueError(f"Expected a {expected_role} checkpoint, found {metadata.role}")
    expected_model = system.identity.get("model_signature")
    expected_sde = system.identity.get("sde_signature")
    expected_pmf = system.identity.get("pmf_sha256")
    expected_pmf_revision = system.identity.get("pmf_revision")
    requires_target_identity = system.identity.get("target_mode") in {
        "pmf_canonical_support",
        "analytic_mb2d",
        "openmm_bms_chirality",
    }
    expected_config = canonical_config_sha256(system.experiment)
    mismatches: list[str] = []
    if expected_model is not None and metadata.model_signature != expected_model:
        mismatches.append("model_signature")
    if expected_sde is not None and metadata.sde_signature != expected_sde:
        mismatches.append("sde_signature")
    if expected_pmf is not None and metadata.assets.pmf_sha256 != expected_pmf:
        mismatches.append("pmf_sha256")
    if expected_pmf_revision is not None and metadata.assets.pmf_revision != expected_pmf_revision:
        mismatches.append("pmf_revision")
    if metadata.config_sha256 != expected_config:
        mismatches.append("experiment_config_sha256")
    if requires_target_identity:
        if metadata.schema_version < 2:
            mismatches.append("checkpoint_schema_version")
        for field in (
            "formal_target_signature",
            "training_target_signature",
            "topology_signature",
        ):
            if getattr(metadata, field) != system.identity.get(field):
                mismatches.append(field)
    if mismatches:
        raise ValueError("Checkpoint does not match the configured runtime: " + ", ".join(mismatches))
    digest_fn = getattr(checkpoint_io, "checkpoint_sha256", None)
    recorded_digest = _field(payload, "sha256")
    digest = (
        str(recorded_digest)
        if recorded_digest is not None
        else str(digest_fn(resolved))
        if digest_fn is not None
        else _path_digest(resolved)
    )
    return ControllerCheckpoint(_restore_variables(payload), metadata, digest, resolved)


def load_controller_initializer_checkpoint(
    path: str | Path,
    *,
    system: RuntimeSystem,
) -> ControllerCheckpoint:
    """Load a forward controller as parameters-only initialization.

    Unlike ``load_controller_checkpoint``, this boundary deliberately permits
    a different experiment config and target energy.  It still requires the
    exact controller, SDE, assets, coordinate/mapping convention, and
    parameter PyTree expected by ``system``.  No source optimizer, replay
    buffer, or training step is returned.
    """

    checkpoint_io = importlib.import_module("cg_bms_jax.checkpoint.io")
    loader = getattr(checkpoint_io, "restore_controller_only", None)
    if loader is None:
        raise AttributeError(
            "cg_bms_jax.checkpoint.io must define restore_controller_only"
        )
    resolved = Path(path).expanduser().resolve()
    payload = loader(resolved, target_variables=system.initial_variables, verify=True)
    metadata = _restore_metadata(_field(payload, "metadata"))
    if metadata.role not in {"forward", "forward_pretrain"}:
        raise ValueError(
            "Controller initializer must have role 'forward' or "
            f"'forward_pretrain', found {metadata.role!r}"
        )

    # AssetProvenance excludes target-restraint configuration, so an exact
    # fingerprint match permits energy/target changes while pinning the PMF,
    # mapping, units, temperature, standardization, and density convention.
    from cg_bms_jax.experiment.training_support import asset_provenance

    expected_assets = asset_provenance(system)
    mismatches: list[str] = []
    if metadata.model_signature != system.identity.get("model_signature"):
        mismatches.append("model_signature")
    if metadata.sde_signature != system.identity.get("sde_signature"):
        mismatches.append("sde_signature")
    if metadata.assets.fingerprint() != expected_assets.fingerprint():
        mismatches.append("asset_fingerprint")
    expected_coordinate = system.identity.get("coordinate_signature")
    if (
        metadata.coordinate_signature is not None
        and expected_coordinate is not None
        and metadata.coordinate_signature != expected_coordinate
    ):
        mismatches.append("coordinate_signature")
    expected_species = system.identity.get("species_signature")
    if (
        metadata.species_signature is not None
        and expected_species is not None
        and metadata.species_signature != expected_species
    ):
        mismatches.append("species_signature")
    expected_training_data = system.identity.get("training_data_sha256")
    if (
        metadata.training_data_sha256 is not None
        and expected_training_data is not None
        and metadata.training_data_sha256 != expected_training_data
    ):
        mismatches.append("training_data_sha256")
    if metadata.role == "forward_pretrain":
        # ``forward_pretrain`` was introduced together with the pinned flow_b
        # data boundary, so unlike legacy forward checkpoints there is no
        # compatibility reason to accept missing or merely non-empty data
        # provenance.  Both fields must identify this runtime's exact training
        # archive; otherwise an initializer trained on flow_ub or another
        # trajectory could be mislabeled as the supported warm start.
        if metadata.training_data_sha256 != expected_training_data:
            mismatches.append("training_data_sha256")
        if metadata.warmstart_data_sha256 != expected_training_data:
            mismatches.append("warmstart_data_sha256")

    source_experiment = _experiment_mapping(_field(payload, "config"))
    if tuple(int(value) for value in source_experiment.get("state_shape", ())) != tuple(
        system.event_shape
    ):
        mismatches.append("state_shape")
    if source_experiment.get("coordinate_mode") != system.experiment.get(
        "coordinate_mode"
    ):
        mismatches.append("coordinate_mode")
    source_coordinates = source_experiment.get("coordinates")
    expected_coordinates = system.experiment.get("coordinates")
    if isinstance(source_coordinates, Mapping) or isinstance(expected_coordinates, Mapping):
        if not isinstance(source_coordinates, Mapping) or not isinstance(
            expected_coordinates, Mapping
        ):
            mismatches.append("coordinate_config")
        else:
            source_com = source_coordinates.get("com_sigma")
            expected_com = expected_coordinates.get("com_sigma")
            if source_com is None or expected_com is None:
                if source_com != expected_com:
                    mismatches.append("com_sigma")
            elif not np.isclose(
                float(source_com), float(expected_com), rtol=0.0, atol=1.0e-12
            ):
                mismatches.append("com_sigma")
    if mismatches:
        raise ValueError(
            "Controller initializer is incompatible with the configured runtime: "
            + ", ".join(dict.fromkeys(mismatches))
        )

    variables = _restore_variables(payload)
    expected_params, expected_constants = split_flax_variables(system.initial_variables)
    restored_params, restored_constants = split_flax_variables(variables)
    _require_controller_tree_compatible(
        restored_params,
        expected_params,
        label=" params",
    )
    _require_controller_tree_compatible(
        restored_constants,
        expected_constants,
        label=" constants",
        compare_discrete_values=True,
    )
    digest = str(_field(payload, "sha256"))
    if len(digest) != 64:
        raise ValueError("controller initializer did not return a verified SHA256")
    return ControllerCheckpoint(variables, metadata, digest, resolved)


def load_forward_controller_for_kind(
    path: str | Path,
    *,
    controller_kind: str,
    system: RuntimeSystem,
) -> ControllerCheckpoint:
    """Load a formal forward or an explicitly requested warm controller.

    A warm controller is restored through the parameters-only initializer
    boundary.  That boundary pins its model, SDE, assets, coordinates, species,
    data provenance, and parameter tree without pretending that the supervised
    checkpoint was produced by fixed-target BMS training.
    """

    kind = str(controller_kind).lower()
    if kind == "forward":
        return load_controller_checkpoint(path, expected_role="forward", system=system)
    if kind == "forward_pretrain":
        checkpoint = load_controller_initializer_checkpoint(path, system=system)
        if checkpoint.metadata.role != "forward_pretrain":
            raise ValueError(
                "forward_controller_kind=forward_pretrain requires a "
                "forward_pretrain checkpoint"
            )
        if checkpoint.metadata.schema_version < 2:
            raise ValueError(
                "forward_pretrain backward/PF ablations require schema-v2 provenance"
            )
        required_identity = {
            "formal_target_signature": system.identity.get(
                "formal_target_signature"
            ),
            "training_target_signature": system.identity.get(
                "training_target_signature"
            ),
            "topology_signature": system.identity.get("topology_signature"),
            "coordinate_signature": system.identity.get("coordinate_signature"),
            "species_signature": system.identity.get("species_signature"),
            "training_data_sha256": system.identity.get("training_data_sha256"),
        }
        mismatches = [
            name
            for name, expected in required_identity.items()
            if expected is None or getattr(checkpoint.metadata, name) != expected
        ]
        if mismatches:
            raise ValueError(
                "forward_pretrain is not an exact runtime identity for backward/PF: "
                + ", ".join(mismatches)
            )
        return checkpoint
    raise ValueError(
        "forward_controller_kind must be 'forward' or 'forward_pretrain'"
    )


def _require_pretrain_backward_compatible(
    forward: CheckpointMetadata,
    backward: CheckpointMetadata,
) -> None:
    """Check PF identities while allowing warm/training config differences.

    Warm pretraining and backward matching have different optimizer-loop
    sections, so their full experiment config hashes cannot match.  Every
    identity that enters the PF dynamics or target remains strict.
    """

    comparisons = {
        "schema_version": (forward.schema_version, backward.schema_version),
        "model_signature": (forward.model_signature, backward.model_signature),
        "sde_signature": (forward.sde_signature, backward.sde_signature),
        "formal_target_signature": (
            forward.formal_target_signature,
            backward.formal_target_signature,
        ),
        "training_target_signature": (
            forward.training_target_signature,
            backward.training_target_signature,
        ),
        "topology_signature": (
            forward.topology_signature,
            backward.topology_signature,
        ),
        "asset_fingerprint": (
            forward.assets.fingerprint(),
            backward.assets.fingerprint(),
        ),
    }
    required = {
        "coordinate_signature": (
            forward.coordinate_signature,
            backward.coordinate_signature,
        ),
        "species_signature": (
            forward.species_signature,
            backward.species_signature,
        ),
        "training_data_sha256": (
            forward.training_data_sha256,
            backward.training_data_sha256,
        ),
        "warmstart_data_sha256": (
            forward.warmstart_data_sha256,
            backward.warmstart_data_sha256,
        ),
    }
    mismatches = [name for name, pair in comparisons.items() if pair[0] != pair[1]]
    mismatches.extend(
        name
        for name, pair in required.items()
        if pair[0] is None or pair[1] is None or pair[0] != pair[1]
    )
    if mismatches:
        raise ValueError(
            "incompatible warm/backward checkpoint metadata: "
            + ", ".join(mismatches)
        )


def require_forward_backward_compatible(
    forward: ControllerCheckpoint,
    backward: ControllerCheckpoint,
    *,
    forward_controller_kind: str = "forward",
) -> None:
    """Reject any PF-ODE assembled from unrelated controller checkpoints."""

    kind = str(forward_controller_kind).lower()
    if backward.metadata.role != "backward":
        raise ValueError("PF-ODE requires a role=backward checkpoint")
    if kind == "forward":
        if forward.metadata.role != "forward":
            raise ValueError("forward_controller_kind=forward requires role=forward")
        forward.metadata.require_compatible(backward.metadata)
    elif kind == "forward_pretrain":
        if forward.metadata.role != "forward_pretrain":
            raise ValueError(
                "forward_controller_kind=forward_pretrain requires "
                "role=forward_pretrain"
            )
        _require_pretrain_backward_compatible(forward.metadata, backward.metadata)
    else:
        raise ValueError(
            "forward_controller_kind must be 'forward' or 'forward_pretrain'"
        )
    if backward.metadata.parent_forward_sha256 != forward.digest:
        raise ValueError(
            "Backward checkpoint was not trained against this exact frozen forward checkpoint: "
            f"{backward.metadata.parent_forward_sha256!r} != {forward.digest!r}"
        )


def metadata_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)


__all__ = [
    "ControllerCheckpoint",
    "load_forward_controller_for_kind",
    "load_controller_checkpoint",
    "load_controller_initializer_checkpoint",
    "metadata_json",
    "require_forward_backward_compatible",
    "run_hydra_entry",
]
