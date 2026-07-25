from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cg_bms_jax.data import bridge_data_sha256
from cg_bms_jax.experiment import pretrain_flat_bridge
from cg_bms_jax.runtime import build_runtime_system, compose_config


class _System:
    transform = None
    event_shape = (2,)

    def __init__(self, experiment, digest: str):
        self.experiment = experiment
        self.identity = {"training_data_sha256": digest}


def _experiment():
    return {
        "coordinate_mode": "mb2d_affine",
        "state_shape": [2],
        "affine": {"offset": [25.0, 25.0], "scale": [10.0, 10.0]},
        "bridge_data": {
            "schema_version": 1,
            "distribution": "full_support_diagonal_gaussian_mixture_v1",
            "coordinate_space": "physical",
            "seed": 5,
            "num_endpoints": 128,
            "components": [
                {
                    "label": "biased_mode",
                    "mean": [48.0, 8.0],
                    "scale": [2.5, 2.5],
                    "weight": 0.9,
                },
                {
                    "label": "broad_tail",
                    "mean": [25.0, 25.0],
                    "scale": [15.0, 15.0],
                    "weight": 0.1,
                },
            ],
        },
    }


def test_distribution_requires_runtime_to_pin_exact_bridge_spec() -> None:
    experiment = _experiment()
    digest = bridge_data_sha256(experiment["bridge_data"])
    distribution, bank = pretrain_flat_bridge._bridge_distribution(
        _System(experiment, digest)
    )
    assert distribution.provenance_sha256 == digest
    assert bank.bridge_data_sha256 == digest
    assert bank.endpoint_1.shape == (128, 2)

    with pytest.raises(ValueError, match="identity"):
        pretrain_flat_bridge._bridge_distribution(
            _System(experiment, "0" * 64)
        )


def test_endpoint_manifest_records_bias_and_full_support(tmp_path: Path) -> None:
    experiment = _experiment()
    digest = bridge_data_sha256(experiment["bridge_data"])
    distribution, bank = pretrain_flat_bridge._bridge_distribution(
        _System(experiment, digest)
    )
    pretrain_flat_bridge._save_endpoint_artifacts(
        output_dir=tmp_path,
        distribution=distribution,
        bank=bank,
        save_endpoint_bank=True,
    )

    manifest = (tmp_path / "flat_bridge_endpoint_manifest.json").read_text(
        encoding="utf-8"
    )
    assert '"full_support": true' in manifest
    assert '"basin_right"' not in manifest
    assert '"biased_mode"' in manifest
    with np.load(tmp_path / "flat_bridge_endpoints.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["endpoint_1"], bank.endpoint_1)
        assert str(data["bridge_data_sha256"]) == digest


def test_entry_point_is_declared() -> None:
    pyproject = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert (
        'cg-bms-pretrain-flat-bridge = '
        '"cg_bms_jax.experiment.pretrain_flat_bridge:main"'
    ) in pyproject


def test_released_pretrain_config_and_runtime_share_endpoint_identity() -> None:
    config = compose_config("pretrain_flat_bridge")
    system = build_runtime_system(config, load_potential=False)
    distribution, bank = pretrain_flat_bridge._bridge_distribution(system)

    expected = bridge_data_sha256(dict(config.experiment.bridge_data))
    assert config.experiment.name == "mb2d_analytic"
    assert system.event_shape == (2,)
    assert system.identity["training_data_sha256"] == expected
    assert distribution.provenance_sha256 == expected
    assert bank.bridge_data_sha256 == expected
