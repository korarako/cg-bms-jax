from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from cg_bms_jax.data.flat_bridge import (
    bridge_data_sha256,
    bridge_endpoint_bank_from_config,
)
from cg_bms_jax.data.mb2d_equilibrium import (
    MB2DEquilibriumDataset,
    generate_mb2d_equilibrium_dataset,
    load_mb2d_equilibrium_dataset,
    save_mb2d_equilibrium_dataset,
)
from cg_bms_jax.evaluation.mb2d import (
    analytic_mb2d_reference,
    mb2d_distribution_metrics,
)
from cg_bms_jax.experiment import generate_mb2d_equilibrium as generator_cli
from cg_bms_jax.potential.mb2d import AnalyticMB2DPotential
from cg_bms_jax.runtime import build_runtime_system, compose_config

EQUILIBRIUM_ABI = "equilibrium_endpoint_npz_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate(
    *,
    seed: int = 17,
    num_train: int = 30_000,
    num_validation: int = 1_000,
    num_test: int = 1_000,
):
    potential = AnalyticMB2DPotential(
        beta=1.0,
        offset=(25.0, 25.0),
        scale=(10.0, 10.0),
        physical_box=((0.0, 50.0), (0.0, 50.0)),
    )
    dataset = generate_mb2d_equilibrium_dataset(
        potential,
        num_train=num_train,
        num_validation=num_validation,
        num_test=num_test,
        grid_resolution=(128, 128),
        seed=seed,
        dtype=np.float32,
    )
    return potential, dataset


def test_exact_grid_dataset_is_deterministic_jittered_and_in_support() -> None:
    potential, first = _generate()
    _, second = _generate()
    _, changed_seed = _generate(seed=18)

    np.testing.assert_array_equal(first.state, second.state)
    np.testing.assert_array_equal(first.physical, second.physical)
    np.testing.assert_array_equal(first.split, second.split)
    assert not np.array_equal(first.state, changed_seed.state)
    assert first.state.shape == (32_000, 2)
    assert first.state.dtype == np.float32
    assert first.physical.dtype == np.float32
    assert first.distribution == EQUILIBRIUM_ABI
    assert first.sha256 is None
    assert first.num_samples == 32_000

    assert np.count_nonzero(first.split == "train") == 30_000
    assert np.count_nonzero(first.split == "validation") == 1_000
    assert np.count_nonzero(first.split == "test") == 1_000
    np.testing.assert_array_equal(first.split_state("train"), first.state[:30_000])
    assert np.asarray(potential.formal_support_mask(first.state)).all()
    np.testing.assert_allclose(
        np.asarray(potential.state_to_physical(first.state)),
        first.physical,
        rtol=0.0,
        atol=4.0e-6,
    )

    # Sampling a categorical midpoint cell without within-cell jitter would
    # leave every normalized coordinate at fractional offset exactly 0.5.
    lower = np.asarray(potential.state_domain.lower)
    upper = np.asarray(potential.state_domain.upper)
    width = (upper - lower) / np.asarray(first.grid_resolution)
    fractional = np.mod((first.state[:2_000] - lower) / width, 1.0)
    assert not np.allclose(fractional, 0.5, rtol=0.0, atol=2.0e-4)
    assert np.unique(np.round(fractional, decimals=3), axis=0).shape[0] > 100


def test_exact_grid_training_split_matches_the_analytic_target() -> None:
    _potential, dataset = _generate()
    reference = analytic_mb2d_reference(beta=1.0, bins=40)
    metrics = mb2d_distribution_metrics(
        dataset.split_physical("train"),
        reference=reference,
    )

    # These tolerances are dominated by 30k-sample histogram noise, not by a
    # model approximation.  A biased endpoint mixture fails by a wide margin.
    assert metrics["in_domain_mass"] == pytest.approx(1.0)
    assert metrics["js_2d"] < 0.035
    assert metrics["basin_l1_error"] < 0.04
    assert abs(metrics["energy_mean_error"]) < 0.25


def test_equilibrium_archive_round_trip_pins_file_and_target_identity(
    tmp_path: Path,
) -> None:
    potential, generated = _generate(
        num_train=128,
        num_validation=32,
        num_test=64,
    )
    path = tmp_path / "mb2d_exact_cgbg_v1.npz"
    digest = save_mb2d_equilibrium_dataset(generated, path)

    assert digest == _sha256(path)
    assert len(digest) == 64
    loaded = load_mb2d_equilibrium_dataset(
        path,
        expected_sha256=digest,
        expected_target=potential,
    )
    assert loaded.sha256 == digest
    assert loaded.path == path.resolve()
    assert loaded.distribution == EQUILIBRIUM_ABI
    assert loaded.target_beta == pytest.approx(1.0)
    assert loaded.affine_offset == pytest.approx((25.0, 25.0))
    assert loaded.affine_scale == pytest.approx((10.0, 10.0))
    np.testing.assert_allclose(
        loaded.physical_box,
        ((0.0, 50.0), (0.0, 50.0)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(loaded.state, generated.state)
    np.testing.assert_array_equal(loaded.physical, generated.physical)
    np.testing.assert_array_equal(loaded.split, generated.split)

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "distribution",
            "state",
            "physical",
            "split",
            "target_beta",
            "affine_offset",
            "affine_scale",
            "physical_box",
            "grid_resolution",
            "seed",
        }
        assert required.issubset(archive.files)
        assert str(archive["distribution"]) == EQUILIBRIUM_ABI

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_mb2d_equilibrium_dataset(
            path,
            expected_sha256="0" * 64,
            expected_target=potential,
        )


@pytest.mark.parametrize(
    "target",
    [
        AnalyticMB2DPotential(beta=0.5),
        AnalyticMB2DPotential(scale=(9.0, 10.0)),
        AnalyticMB2DPotential(physical_box=((0.0, 49.0), (0.0, 50.0))),
    ],
)
def test_equilibrium_archive_rejects_wrong_formal_target(
    tmp_path: Path,
    target: AnalyticMB2DPotential,
) -> None:
    _potential, generated = _generate(
        num_train=32,
        num_validation=8,
        num_test=8,
    )
    path = tmp_path / "endpoints.npz"
    digest = save_mb2d_equilibrium_dataset(generated, path)

    with pytest.raises(ValueError, match="target"):
        load_mb2d_equilibrium_dataset(
            path,
            expected_sha256=digest,
            expected_target=target,
        )


def test_equilibrium_bridge_config_selects_hash_pinned_training_split(
    tmp_path: Path,
) -> None:
    potential, generated = _generate(
        num_train=48,
        num_validation=16,
        num_test=24,
    )
    data_dir = tmp_path / "data" / "mb2d_exact_cgbg_v1"
    data_dir.mkdir(parents=True)
    path = data_dir / "endpoints.npz"
    digest = save_mb2d_equilibrium_dataset(generated, path)
    bridge_data = {
        "schema_version": 1,
        "distribution": EQUILIBRIUM_ABI,
        "coordinate_space": "state",
        "path": "data/mb2d_exact_cgbg_v1/endpoints.npz",
        "sha256": digest,
        "split": "train",
        "num_endpoints": 40,
        "target_beta": 1.0,
    }

    bank = bridge_endpoint_bank_from_config(
        bridge_data,
        affine_offset=potential.offset,
        affine_scale=potential.scale,
        physical_box=potential.physical_box,
        base_dir=tmp_path,
    )
    assert bank.num_endpoints == 40
    assert bank.event_shape == (2,)
    assert bank.distribution == EQUILIBRIUM_ABI
    assert bank.dataset_sha256 == digest
    assert bank.split_name == "train"
    assert bank.bridge_data_sha256 == bridge_data_sha256(bridge_data)
    np.testing.assert_array_equal(
        bank.endpoint_1,
        generated.split_state("train")[:40],
    )
    np.testing.assert_array_equal(
        bank.physical,
        generated.split_physical("train")[:40],
    )
    assert bank.component_index.shape == (40,)
    assert np.all(bank.component_index == -1)
    assert bank.component_labels == ()
    assert bank.component_counts == ()

    tampered = dict(bridge_data, sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bridge_endpoint_bank_from_config(
            tampered,
            affine_offset=potential.offset,
            affine_scale=potential.scale,
            physical_box=potential.physical_box,
            base_dir=tmp_path,
        )

    too_large = dict(bridge_data, num_endpoints=49)
    with pytest.raises(ValueError, match="num_endpoints"):
        bridge_endpoint_bank_from_config(
            too_large,
            affine_offset=potential.offset,
            affine_scale=potential.scale,
            physical_box=potential.physical_box,
            base_dir=tmp_path,
        )


def test_released_equilibrium_config_pins_dataset_and_differs_from_legacy() -> None:
    config = compose_config(
        "pretrain_flat_bridge",
        ["experiment=mb2d_analytic_equilibrium_bridge"],
    )
    system = build_runtime_system(config, load_potential=False)
    bridge_data = dict(config.experiment.bridge_data)
    bank = bridge_endpoint_bank_from_config(
        bridge_data,
        affine_offset=config.experiment.affine.offset,
        affine_scale=config.experiment.affine.scale,
        physical_box=config.experiment.target.box,
        base_dir=Path(__file__).resolve().parents[1],
    )
    legacy = compose_config(
        "pretrain_flat_bridge",
        ["experiment=mb2d_analytic"],
    )
    legacy_digest = bridge_data_sha256(dict(legacy.experiment.bridge_data))

    assert config.experiment.name == "mb2d_analytic_equilibrium_bridge"
    assert bank.num_endpoints == 100_000
    assert (
        bank.dataset_sha256
        == "f4c43d8feff619102e5cc1f8ad2dbd422d9a201453d356d4f59e48cb9032347d"
    )
    assert system.identity["training_data_sha256"] == bank.bridge_data_sha256
    assert (
        system.identity["equilibrium_endpoint_spec_sha256"]
        == bank.bridge_data_sha256
    )
    assert system.identity["synthetic_endpoint_spec_sha256"] is None
    assert bank.bridge_data_sha256 != legacy_digest


def test_generator_console_entry_point_is_declared() -> None:
    pyproject = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert (
        'cg-bms-generate-mb2d-equilibrium = '
        '"cg_bms_jax.experiment.generate_mb2d_equilibrium:main"'
    ) in pyproject


def test_generator_validation_uses_the_configured_physical_domain() -> None:
    domain = ((20.0, 50.0), (0.0, 40.0))
    rng = np.random.default_rng(41)
    physical = np.column_stack(
        (
            rng.uniform(*domain[0], size=2_000),
            rng.uniform(*domain[1], size=2_000),
        )
    )
    metrics = generator_cli._validation_metrics(
        physical=physical,
        beta=1.0,
        bins=32,
        domain=domain,
    )
    expected = analytic_mb2d_reference(
        beta=1.0,
        bins=32,
        domain=domain,
    )
    default = analytic_mb2d_reference(beta=1.0, bins=32)

    np.testing.assert_allclose(
        metrics["exact_energy_mean"],
        np.sum(expected.probability * expected.energy),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert not np.isclose(
        metrics["exact_energy_mean"],
        np.sum(default.probability * default.energy),
        rtol=0.0,
        atol=1.0e-3,
    )
    convergence = generator_cli._grid_convergence(
        beta=1.0,
        resolutions=(16, 32),
        domain=domain,
    )
    assert convergence["fine"]["energy_mean"] == pytest.approx(
        metrics["exact_energy_mean"],
        abs=1.0e-12,
    )


@pytest.mark.parametrize("preexisting", [False, True])
def test_failed_generator_acceptance_never_publishes_candidate_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: bool,
) -> None:
    domain = ((20.0, 50.0), (0.0, 40.0))
    potential = AnalyticMB2DPotential(physical_box=domain)
    physical = np.asarray(
        [[21.0, 1.0], [30.0, 10.0], [40.0, 20.0], [49.0, 39.0]],
        dtype=np.float32,
    )
    state = (
        physical - np.asarray(potential.offset, dtype=np.float32)
    ) / np.asarray(potential.scale, dtype=np.float32)
    dataset = MB2DEquilibriumDataset(
        state=state,
        physical=physical,
        reduced_energy=np.zeros(4, dtype=np.float32),
        split=np.asarray(
            ["train", "train", "validation", "test"],
            dtype="<U10",
        ),
        target_beta=1.0,
        affine_offset=potential.offset,
        affine_scale=potential.scale,
        physical_box=domain,
        grid_resolution=(16, 16),
        seed=7,
        target_signature="1" * 64,
    )
    system = SimpleNamespace(
        potential=potential,
        identity={
            "formal_target_signature": "2" * 64,
            "coordinate_signature": "3" * 64,
        },
    )
    monkeypatch.setattr(
        generator_cli,
        "build_runtime_system",
        lambda *_args, **_kwargs: system,
    )
    monkeypatch.setattr(
        generator_cli,
        "generate_mb2d_equilibrium_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    seen_domains: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def failed_validation(**kwargs):
        seen_domains.append(kwargs["domain"])
        return {
            "histogram_js": 1.0,
            "basin_l1_error": 0.0,
        }

    def converged_grid(**kwargs):
        seen_domains.append(kwargs["domain"])
        return {
            "energy_mean_abs_delta": 0.0,
            "basin_l1_delta": 0.0,
        }

    monkeypatch.setattr(generator_cli, "_validation_metrics", failed_validation)
    monkeypatch.setattr(generator_cli, "_grid_convergence", converged_grid)

    output = tmp_path / "endpoints.npz"
    manifest = output.with_suffix(".manifest.json")
    if preexisting:
        output.write_bytes(b"previously-accepted-npz")
        manifest.write_bytes(b"previously-accepted-manifest")
    config = OmegaConf.create(
        {
            "seed": 7,
            "output": str(output),
            "overwrite": True,
            "experiment": {"name": "custom_domain_mb2d"},
            "dataset": {
                "num_train": 2,
                "num_validation": 1,
                "num_test": 1,
                "grid_resolution": [16, 16],
                "validation_bins": 16,
                "maximum_histogram_js": 0.1,
                "maximum_basin_l1": 0.1,
                "grid_convergence_resolution": [16, 32],
                "maximum_grid_energy_mean_delta": 0.1,
                "maximum_grid_basin_l1": 0.1,
            },
        }
    )

    with pytest.raises(RuntimeError, match="failed validation"):
        generator_cli.run(config)

    assert seen_domains == [domain, domain]
    if preexisting:
        assert output.read_bytes() == b"previously-accepted-npz"
        assert manifest.read_bytes() == b"previously-accepted-manifest"
    else:
        assert not output.exists()
        assert not manifest.exists()
    assert not list(tmp_path.glob(".*.candidate.*"))
