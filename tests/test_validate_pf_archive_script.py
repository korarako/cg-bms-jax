from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_pf_archive.py"
SPEC = importlib.util.spec_from_file_location("validate_pf_archive_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_archive(
    path: Path,
    *,
    forward_sha: str = "f" * 64,
    include_sampling: bool = True,
) -> None:
    count = 4
    endpoint_sha = "e" * 64
    signatures = {
        "formal_target_signature": "1" * 64,
        "training_target_signature": "2" * 64,
        "coordinate_signature": "3" * 64,
        "model_signature": "4" * 64,
        "sde_signature": "5" * 64,
        "topology_signature": "6" * 64,
    }
    metadata = {
        **signatures,
        "forward_checkpoint_sha256": forward_sha,
        "backward_checkpoint_sha256": "b" * 64,
        "forward_controller_kind": "forward_pretrain",
        "endpoint_distribution": "equilibrium_endpoint_npz_v1",
        "equilibrium_endpoint_spec_sha256": endpoint_sha,
        "configured_endpoint_spec_sha256": endpoint_sha,
        "training_data_sha256": endpoint_sha,
        "forward_metadata": {
            **signatures,
            "warmstart_data_sha256": endpoint_sha,
        },
        "backward_metadata": {
            **signatures,
            "parent_forward_sha256": "f" * 64,
        },
        "likelihood": {
            "solver": "dopri5",
            "divergence": "exact",
            "dt0": 1.0e-3,
            "rtol": 1.0e-5,
            "atol": 1.0e-6,
            "max_steps": 16384,
        },
        "seed": 203,
    }
    if include_sampling:
        metadata["sampling"] = {
            "global_num_samples": count,
            "local_num_samples": count,
        }
    np.savez_compressed(
        path,
        R=np.zeros((count, 2)),
        U=np.zeros(count),
        logp=np.zeros(count),
        logw=np.full(count, -np.log(count)),
        weights=np.full(count, 1.0 / count),
        sampler_kind=np.asarray("pf_ode"),
        density_mode=np.asarray("ambient_exact"),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_validate_pf_archive_accepts_matching_provenance(tmp_path: Path) -> None:
    archive = tmp_path / "pf.npz"
    _write_archive(archive)
    result = MODULE.validate_pf_archive(
        archive,
        forward_sha256="f" * 64,
        backward_sha256="b" * 64,
        forward_controller_kind="forward_pretrain",
        seed=203,
        num_samples=4,
        density_mode="ambient_exact",
        endpoint_distribution="equilibrium_endpoint_npz_v1",
        likelihood_dt0=1.0e-3,
        likelihood_rtol=1.0e-5,
        likelihood_atol=1.0e-6,
        likelihood_max_steps=16384,
    )
    assert result["samples"] == 4
    assert result["sampling_provenance_present"] is True


def test_validate_pf_archive_accepts_legacy_size_provenance(tmp_path: Path) -> None:
    archive = tmp_path / "pf.npz"
    _write_archive(archive, include_sampling=False)
    result = MODULE.validate_pf_archive(
        archive,
        forward_sha256="f" * 64,
        backward_sha256="b" * 64,
        forward_controller_kind="forward_pretrain",
        seed=203,
        num_samples=4,
        density_mode="ambient_exact",
        endpoint_distribution="equilibrium_endpoint_npz_v1",
        likelihood_dt0=1.0e-3,
        likelihood_rtol=1.0e-5,
        likelihood_atol=1.0e-6,
        likelihood_max_steps=16384,
    )
    assert result["samples"] == 4
    assert result["sampling_provenance_present"] is False


def test_validate_pf_archive_rejects_stale_checkpoint(tmp_path: Path) -> None:
    archive = tmp_path / "pf.npz"
    _write_archive(archive, forward_sha="0" * 64)
    with pytest.raises(ValueError, match="forward_checkpoint_sha256"):
        MODULE.validate_pf_archive(
            archive,
            forward_sha256="f" * 64,
            backward_sha256="b" * 64,
            forward_controller_kind="forward_pretrain",
            seed=203,
            num_samples=4,
            density_mode="ambient_exact",
            endpoint_distribution="equilibrium_endpoint_npz_v1",
            likelihood_dt0=1.0e-3,
            likelihood_rtol=1.0e-5,
            likelihood_atol=1.0e-6,
            likelihood_max_steps=16384,
        )
