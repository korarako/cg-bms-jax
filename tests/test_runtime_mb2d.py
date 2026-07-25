from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.data import bridge_data_sha256
from cg_bms_jax.experiment.training_support import asset_provenance
from cg_bms_jax.runtime import build_runtime_system, compose_config


def test_mb2d_smoke_config_builds_asset_free_runtime_and_affine_map() -> None:
    config = compose_config("train_forward", ["experiment=mb2d_analytic_smoke"])
    system = build_runtime_system(
        config,
        key=jax.random.PRNGKey(11),
        load_potential=False,
    )
    state = jnp.zeros((4, 2), dtype=jnp.float32)
    control = system.apply(system.initial_variables, 0.4, state)
    physical = system.to_physical(state)

    assert system.event_shape == (2,)
    assert system.source.event_shape == (2,)
    assert system.source.scale == pytest.approx(1.0)
    assert control.shape == state.shape
    assert np.asarray(jnp.isfinite(control)).all()
    np.testing.assert_allclose(physical, np.full((4, 2), 25.0), rtol=0.0, atol=0.0)
    assert system.potential is None
    assert system.identity["experiment_family"] == "mb2d_analytic"
    assert system.identity["density_mode"] == "ambient_exact"
    assert system.domain_metadata()["support_mode"] == "cartesian_box"
    np.testing.assert_array_equal(
        system.support_mask(jnp.asarray([[0.0, 0.0], [3.0, 0.0]])),
        np.asarray([True, False]),
    )


def test_mb2d_runtime_loads_analytic_target_without_external_assets() -> None:
    config = compose_config("train_forward", ["experiment=mb2d_analytic_smoke"])
    system = build_runtime_system(config, load_potential=True)
    result = system.evaluate_target(
        jnp.asarray([[0.0, 0.0], [3.0, 0.0]]),
        include_training_wall=True,
    )
    assert result.energy.shape == (2,)
    np.testing.assert_array_equal(result.valid_mask, np.asarray([True, False]))
    assert np.asarray(jnp.isfinite(result.score)).all()


def test_mb2d_identity_pins_affine_target_and_synthetic_endpoint_spec() -> None:
    base_config = compose_config(
        "train_forward",
        ["experiment=mb2d_analytic_smoke"],
    )
    changed_affine_config = compose_config(
        "train_forward",
        [
            "experiment=mb2d_analytic_smoke",
            "experiment.affine.scale=[9.0,10.0]",
        ],
    )
    changed_beta_config = compose_config(
        "train_forward",
        [
            "experiment=mb2d_analytic_smoke",
            "experiment.target.beta=0.5",
            "experiment.kT=2.0",
        ],
    )
    base = build_runtime_system(base_config, load_potential=False)
    changed_affine = build_runtime_system(changed_affine_config, load_potential=False)
    changed_beta = build_runtime_system(changed_beta_config, load_potential=False)

    expected_bridge_digest = bridge_data_sha256(
        dict(base_config.experiment.bridge_data)
    )
    assert base.identity["training_data_sha256"] == expected_bridge_digest
    assert (
        base.identity["synthetic_endpoint_spec_sha256"]
        == expected_bridge_digest
    )
    assert (
        base.identity["coordinate_signature"]
        != changed_affine.identity["coordinate_signature"]
    )
    assert (
        base.identity["formal_target_signature"]
        != changed_beta.identity["formal_target_signature"]
    )
    assert base.identity["pmf_sha256"] == base.identity["formal_target_signature"]
    assert len(base.identity["topology_signature"]) == 64


def test_mb2d_runtime_rejects_duplicate_temperature_disagreement() -> None:
    config = compose_config(
        "train_forward",
        [
            "experiment=mb2d_analytic_smoke",
            "experiment.target.beta=0.5",
        ],
    )
    with pytest.raises(ValueError, match="kT must equal"):
        build_runtime_system(config, load_potential=False)


def test_mb2d_checkpoint_assets_use_the_2d_affine_identity() -> None:
    config = compose_config(
        "train_forward",
        ["experiment=mb2d_analytic_smoke"],
    )
    system = build_runtime_system(config, load_potential=False)
    assets = asset_provenance(system)

    assert assets.mapping_name == "mb2d_affine"
    assert assets.mapping_indices == (0, 1)
    assert assets.coordinate_unit == "cg_bg_mb_coordinate"
    assert assets.standardization_std == pytest.approx(10.0)
    assert assets.event_size == 2
    assert assets.density_mode == "ambient_2d"
    assert assets.domain == "finite_cartesian_box"


def test_existing_mb1d_runtime_dispatch_is_unchanged() -> None:
    config = compose_config("train_forward", ["experiment=mb_cg1d_smoke"])
    system = build_runtime_system(config, load_potential=False)
    assert system.event_shape == (1,)
    assert system.identity["experiment_family"] == "mb_cg1d"
    assert system.physical_map is None
