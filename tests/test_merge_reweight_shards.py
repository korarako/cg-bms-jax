from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_reweight_shards.py"
SPEC = importlib.util.spec_from_file_location("merge_reweight_shards", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MERGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGER)


def _write_shard(
    path: Path,
    *,
    batch_offset: int,
    initial: np.ndarray,
    reduced: np.ndarray,
    logq: np.ndarray,
    forward_sha: str = "1" * 64,
) -> None:
    count = reduced.size
    initial_logq = -0.5 * (
        np.sum(np.square(initial, dtype=np.float64), axis=1)
        + initial.shape[1] * np.log(2.0 * np.pi)
    )
    raw = -reduced - logq
    local = np.exp(raw - np.max(raw))
    local /= local.sum()
    pf_stage = path.with_name(f"{path.stem}.pf_stage.npz")
    np.savez_compressed(
        pf_stage,
        X_standardized=initial,
        logq_ambient=logq,
        initial_X=initial,
        initial_logq=initial_logq,
        ode_steps_per_batch=np.asarray([17]),
        metadata_json=np.asarray("{}"),
    )
    metadata = {
        "seed": 3,
        "forward_checkpoint_sha256": forward_sha,
        "backward_checkpoint_sha256": "2" * 64,
        "domain": {"box": 1, "anchor": 0, "margin": 0, "support_mode": "test"},
        "likelihood": {
            "rtol": 1.0e-5,
            "atol": 1.0e-6,
            "pf_stage": str(pf_stage),
        },
        "sampling": {
            "global_num_samples": 4,
            "batch_size": 2,
            "global_num_batches": 2,
            "batch_offset": batch_offset,
            "local_num_batches": 1,
            "local_num_samples": 2,
            "key_schedule": "test",
        },
    }
    np.savez_compressed(
        path,
        R=initial,
        U=reduced,
        logp=logq,
        logw=np.log(local),
        logw_raw=raw,
        weights=local,
        logq_ambient=logq,
        valid_mask=np.ones(count, dtype=bool),
        support_mask=np.ones(count, dtype=bool),
        sampler_kind=np.asarray("pf_ode"),
        density_mode=np.asarray("ambient_exact"),
        metadata_json=np.asarray(json.dumps(metadata)),
        X_standardized=initial,
        initial_X=initial,
        initial_logq=initial_logq,
        ode_steps_per_batch=np.asarray([17]),
        target_reduced_energy=reduced,
        U_target=reduced,
    )


def test_merge_recomputes_one_global_normalization(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_shard(
        first,
        batch_offset=0,
        initial=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        reduced=np.asarray([1.0, 2.0]),
        logq=np.asarray([-2.0, -3.0]),
    )
    _write_shard(
        second,
        batch_offset=1,
        initial=np.asarray([[0.0, 1.0], [1.0, 1.0]]),
        reduced=np.asarray([4.0, 8.0]),
        logq=np.asarray([-1.0, -2.0]),
    )
    output = tmp_path / "merged.npz"
    summary = MERGER.merge(
        output,
        [first, second],
        verify_standard_normal_source=True,
    )
    with np.load(output, allow_pickle=False) as archive:
        raw = -archive["target_reduced_energy"] - archive["logq_ambient"]
        expected = np.exp(raw - np.max(raw))
        expected /= expected.sum()
        np.testing.assert_allclose(archive["weights"], expected)
        np.testing.assert_allclose(archive["logw_raw"], raw)
        np.testing.assert_array_equal(archive["shard_id"], [0, 0, 1, 1])
    assert summary["num_samples"] == 4
    assert summary["standard_normal_source_verified"] is True


def test_merge_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_shard(
        first,
        batch_offset=0,
        initial=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        reduced=np.asarray([1.0, 2.0]),
        logq=np.asarray([-2.0, -3.0]),
    )
    _write_shard(
        second,
        batch_offset=1,
        initial=np.asarray([[0.0, 1.0], [1.0, 1.0]]),
        reduced=np.asarray([4.0, 8.0]),
        logq=np.asarray([-1.0, -2.0]),
        forward_sha="9" * 64,
    )
    with pytest.raises(ValueError, match="metadata differs"):
        MERGER.merge(tmp_path / "merged.npz", [first, second])


def test_merge_ignores_tampered_local_normalized_weights(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_shard(
        first,
        batch_offset=0,
        initial=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        reduced=np.asarray([1.0, 2.0]),
        logq=np.asarray([-2.0, -3.0]),
    )
    _write_shard(
        second,
        batch_offset=1,
        initial=np.asarray([[0.0, 1.0], [1.0, 1.0]]),
        reduced=np.asarray([4.0, 8.0]),
        logq=np.asarray([-1.0, -2.0]),
    )
    with np.load(first, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["weights"] = np.asarray([0.999, 0.001])
    payload["logw"] = np.log(payload["weights"])
    np.savez_compressed(first, **payload)

    output = tmp_path / "merged.npz"
    MERGER.merge(output, [first, second])
    with np.load(output, allow_pickle=False) as archive:
        raw = -archive["target_reduced_energy"] - archive["logq_ambient"]
        expected = np.exp(raw - np.max(raw))
        expected /= expected.sum()
        np.testing.assert_allclose(archive["weights"], expected)
