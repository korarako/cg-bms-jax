from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from cg_bms_jax.checkpoint import (
    CheckpointMetadata,
    RestoredControllerCheckpoint,
    canonical_config_sha256,
    restore_controller_only,
    save_checkpoint,
)
from cg_bms_jax.data import init_replay_buffer
from cg_bms_jax.experiment.common import load_controller_initializer_checkpoint
from cg_bms_jax.experiment.training_support import asset_provenance, checkpoint_metadata
from cg_bms_jax.training import initialize_train_state, make_optimizer


def _system() -> SimpleNamespace:
    params = {"w": jnp.zeros((2, 1), dtype=jnp.float32)}
    constants = {
        "constants": {"atom_indices": jnp.asarray([0, 1], dtype=jnp.int32)}
    }
    experiment = {
        "state_shape": [1],
        "coordinate_mode": "identity",
        "kT": 1.0,
        "target": {"mode": "new-energy-target"},
    }
    return SimpleNamespace(
        initial_variables={"params": params, **constants},
        event_shape=(1,),
        experiment=experiment,
        transform=None,
        identity={
            "model_signature": "model-v1",
            "sde_signature": "sde-v1",
            "pmf_revision": "pmf-revision-v1",
            "pmf_sha256": "1" * 64,
            "coordinate_signature": "2" * 64,
            "species_signature": "3" * 64,
            "training_data_sha256": "4" * 64,
        },
    )


def _source_config() -> dict[str, object]:
    # The energy target intentionally differs from _system().experiment.
    return {
        "state_shape": [1],
        "coordinate_mode": "identity",
        "kT": 1.0,
        "target": {"mode": "bridge-pretraining-objective"},
    }


def _metadata(system: SimpleNamespace, config: dict[str, object], step: int) -> CheckpointMetadata:
    return CheckpointMetadata(
        role="forward_pretrain",
        global_step=step,
        created_at_utc=datetime.now(UTC).isoformat(),
        code_revision="test",
        config_sha256=canonical_config_sha256(config),
        model_signature=system.identity["model_signature"],
        sde_signature=system.identity["sde_signature"],
        assets=asset_provenance(system),
        coordinate_signature=system.identity["coordinate_signature"],
        species_signature=system.identity["species_signature"],
        training_data_sha256=system.identity["training_data_sha256"],
        warmstart_data_sha256=system.identity["training_data_sha256"],
    )


def _trained_source_state(system: SimpleNamespace):
    params = {"w": jnp.asarray([[1.0], [-2.0]], dtype=jnp.float32)}
    constants = {
        "constants": {"atom_indices": jnp.asarray([0, 1], dtype=jnp.int32)}
    }
    optimizer = make_optimizer(learning_rate=1.0e-3, gradient_clip_norm=1.0)
    state = initialize_train_state(params, constants, optimizer)
    gradients = jax.tree_util.tree_map(jnp.ones_like, state.params)
    for _ in range(3):
        updates, optimizer_state = optimizer.update(
            gradients, state.optimizer_state, state.params
        )
        state = state._replace(
            params=optax.apply_updates(state.params, updates),
            optimizer_state=optimizer_state,
            step=state.step + 1,
        )
    return state


def test_controller_only_initializer_allows_target_change_and_resets_training_state(
    tmp_path,
) -> None:
    system = _system()
    source_state = _trained_source_state(system)
    config = _source_config()
    path = tmp_path / "forward_pretrain_step_00000003"
    digest = save_checkpoint(
        path,
        state=source_state,
        config=config,
        metadata=_metadata(system, config, step=3),
    )

    partial = restore_controller_only(
        path,
        target_variables=system.initial_variables,
    )
    assert partial.sha256 == digest
    assert not hasattr(partial, "optimizer_state")
    assert not hasattr(partial, "step")

    initializer = load_controller_initializer_checkpoint(path, system=system)
    assert initializer.metadata.role == "forward_pretrain"
    assert initializer.digest == digest
    np.testing.assert_allclose(
        initializer.variables["params"]["w"], source_state.params["w"]
    )

    # This is exactly the boundary used by train_forward: a new optimizer and
    # empty replay are constructed around only the imported controller tree.
    optimizer = make_optimizer(learning_rate=2.0e-4, gradient_clip_norm=10.0)
    fresh = initialize_train_state(
        initializer.variables["params"],
        {"constants": initializer.variables["constants"]},
        optimizer,
    )
    assert int(fresh.step) == 0
    energy_metadata = checkpoint_metadata(
        role="forward",
        state=fresh,
        system=system,
        saved_config=system.experiment,
        warmstart_data_sha256=initializer.metadata.warmstart_data_sha256,
        initial_controller_path=str(initializer.path),
        initial_controller_sha256=initializer.digest,
        initial_controller_role=initializer.metadata.role,
    )
    assert energy_metadata.initial_controller_path == str(initializer.path)
    assert energy_metadata.initial_controller_sha256 == digest
    assert energy_metadata.initial_controller_role == "forward_pretrain"
    assert energy_metadata.warmstart_data_sha256 == system.identity[
        "training_data_sha256"
    ]
    replay = init_replay_buffer(
        8,
        {
            "endpoint_0": jnp.zeros((2, 1)),
            "endpoint_1": jnp.zeros((2, 1)),
            "target_score": jnp.zeros((2, 1)),
            "valid": jnp.ones((2,), dtype=jnp.bool_),
        },
    )
    assert int(replay.size) == 0
    assert int(replay.cursor) == 0


def test_controller_initializer_rejects_incompatible_identity_or_parameter_tree(
    monkeypatch,
    tmp_path,
) -> None:
    system = _system()
    config = _source_config()
    metadata = _metadata(system, config, step=3)
    base = RestoredControllerCheckpoint(
        params={"w": jnp.ones((3, 1), dtype=jnp.float32)},
        constants={
            "constants": {"atom_indices": jnp.asarray([0, 1], dtype=jnp.int32)}
        },
        config=config,
        metadata=metadata,
        sha256="6" * 64,
    )

    import cg_bms_jax.checkpoint.io as checkpoint_io

    monkeypatch.setattr(checkpoint_io, "restore_controller_only", lambda *args, **kwargs: base)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_controller_initializer_checkpoint(tmp_path / "unused", system=system)

    incompatible = replace(base, metadata=replace(metadata, sde_signature="other-sde"))
    monkeypatch.setattr(
        checkpoint_io,
        "restore_controller_only",
        lambda *args, **kwargs: incompatible,
    )
    with pytest.raises(ValueError, match="sde_signature"):
        load_controller_initializer_checkpoint(tmp_path / "unused", system=system)

    wrong_warm_data = replace(
        base,
        params={"w": jnp.ones((2, 1), dtype=jnp.float32)},
        metadata=replace(metadata, warmstart_data_sha256="9" * 64),
    )
    monkeypatch.setattr(
        checkpoint_io,
        "restore_controller_only",
        lambda *args, **kwargs: wrong_warm_data,
    )
    with pytest.raises(ValueError, match="warmstart_data_sha256"):
        load_controller_initializer_checkpoint(tmp_path / "unused", system=system)
