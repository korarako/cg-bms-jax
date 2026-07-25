from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from cg_bms_jax.checkpoint import (
    AssetProvenance,
    CheckpointMetadata,
    canonical_config_sha256,
    restore_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from cg_bms_jax.experiment.common import (
    ControllerCheckpoint,
    require_forward_backward_compatible,
)
from cg_bms_jax.training import initialize_train_state, make_optimizer


def _metadata(config, step):
    assets = AssetProvenance(
        pmf_revision="test-revision",
        pmf_sha256="1" * 64,
        mapping_name="toy_identity",
        mapping_indices=(0,),
        temperature_kelvin=300.0,
        thermal_energy_kj_mol=2.494,
        coordinate_unit="dimensionless",
        standardization_std=1.0,
        num_particles=1,
        spatial_dimension=1,
    )
    return CheckpointMetadata(
        role="forward",
        global_step=step,
        created_at_utc=datetime.now(UTC).isoformat(),
        code_revision="test",
        config_sha256=canonical_config_sha256(config),
        model_signature="tiny-linear-v1",
        sde_signature="edm-test-v1",
        assets=assets,
    )


def _state():
    params = {"w": jnp.asarray([[1.0, -2.0]], dtype=jnp.float32)}
    constants = {"constants": {"atom_indices": jnp.asarray([0, 1], dtype=jnp.int32)}}
    optimizer = make_optimizer(learning_rate=1.0e-3, gradient_clip_norm=1.0)
    state = initialize_train_state(params, constants, optimizer)
    gradients = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, optimizer_state = optimizer.update(gradients, state.optimizer_state, state.params)
    return state._replace(
        params=optax.apply_updates(state.params, updates),
        optimizer_state=optimizer_state,
        step=state.step + 1,
    )


def test_checkpoint_round_trip_preserves_typed_optimizer_and_metadata(tmp_path):
    config = {"experiment": "toy", "training": {"learning_rate": 1.0e-3}}
    state = _state()
    path = tmp_path / "step_000001"
    digest = save_checkpoint(
        path,
        state=state,
        config=config,
        metadata=_metadata(config, step=1),
    )
    assert verify_checkpoint(path) == digest
    restored = restore_checkpoint(path, target_state=state)
    assert restored.sha256 == digest
    assert restored.config == config
    assert restored.metadata.global_step == 1
    assert int(restored.step) == 1
    np.testing.assert_allclose(restored.params["w"], state.params["w"])
    np.testing.assert_array_equal(
        restored.constants["constants"]["atom_indices"],
        state.constants["constants"]["atom_indices"],
    )
    assert jax.tree_util.tree_structure(restored.optimizer_state) == jax.tree_util.tree_structure(
        state.optimizer_state
    )
    with pytest.raises(FileExistsError):
        save_checkpoint(path, state=state, config=config, metadata=_metadata(config, step=1))


def test_checkpoint_checksum_detects_manifest_tampering(tmp_path):
    config = {"experiment": "toy"}
    state = _state()
    path = tmp_path / "checkpoint"
    save_checkpoint(path, state=state, config=config, metadata=_metadata(config, step=1))
    manifest = path / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checkpoint(path)


def test_checkpoint_rejects_metadata_config_or_step_mismatch(tmp_path):
    config = {"experiment": "toy"}
    state = _state()
    wrong_config_metadata = _metadata({"experiment": "different"}, step=1)
    with pytest.raises(ValueError, match="config_sha256"):
        save_checkpoint(
            tmp_path / "bad_config",
            state=state,
            config=config,
            metadata=wrong_config_metadata,
        )
    with pytest.raises(ValueError, match="global_step"):
        save_checkpoint(
            tmp_path / "bad_step",
            state=state,
            config=config,
            metadata=_metadata(config, step=2),
        )


def test_schema_v2_checkpoint_requires_and_compares_target_identity():
    base = _metadata({"experiment": "canonical"}, step=1)
    with pytest.raises(ValueError, match="schema-v2 molecular checkpoints"):
        replace(base, schema_version=2)

    canonical = replace(
        base,
        schema_version=2,
        formal_target_signature="formal-v1",
        training_target_signature="training-v1",
        topology_signature="topology-v1",
    )
    canonical.require_compatible(canonical)
    changed = replace(canonical, topology_signature="topology-v2")
    with pytest.raises(ValueError, match="topology_signature"):
        canonical.require_compatible(changed)


def test_explicit_pretrain_pf_compatibility_ignores_only_full_config_hash():
    base = replace(
        _metadata({"experiment": "warm"}, step=10),
        role="forward_pretrain",
        coordinate_signature="2" * 64,
        species_signature="3" * 64,
        training_data_sha256="4" * 64,
        warmstart_data_sha256="4" * 64,
        schema_version=2,
        formal_target_signature="formal-v1",
        training_target_signature="training-v1",
        topology_signature="topology-v1",
    )
    forward = ControllerCheckpoint(
        variables={},
        metadata=base,
        digest="a" * 64,
        path=Path("warm"),
    )
    backward_metadata = replace(
        base,
        role="backward",
        config_sha256=canonical_config_sha256({"experiment": "backward"}),
        parent_forward_sha256=forward.digest,
    )
    backward = ControllerCheckpoint(
        variables={},
        metadata=backward_metadata,
        digest="b" * 64,
        path=Path("backward"),
    )

    require_forward_backward_compatible(
        forward,
        backward,
        forward_controller_kind="forward_pretrain",
    )
    with pytest.raises(ValueError, match="role=forward"):
        require_forward_backward_compatible(forward, backward)

    wrong_sde = replace(
        backward,
        metadata=replace(backward.metadata, sde_signature="other-sde"),
    )
    with pytest.raises(ValueError, match="sde_signature"):
        require_forward_backward_compatible(
            forward,
            wrong_sde,
            forward_controller_kind="forward_pretrain",
        )

    missing_coordinate = replace(
        backward,
        metadata=replace(backward.metadata, coordinate_signature=None),
    )
    with pytest.raises(ValueError, match="coordinate_signature"):
        require_forward_backward_compatible(
            forward,
            missing_coordinate,
            forward_controller_kind="forward_pretrain",
        )

    wrong_parent = replace(
        backward,
        metadata=replace(backward.metadata, parent_forward_sha256="c" * 64),
    )
    with pytest.raises(ValueError, match="exact frozen forward"):
        require_forward_backward_compatible(
            forward,
            wrong_parent,
            forward_controller_kind="forward_pretrain",
        )
