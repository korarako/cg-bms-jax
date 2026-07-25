from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from cg_bms_jax.data import load_ala2_warmstart_dataset, uniform_so3_matrices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_archive(path: Path, *, frames: int = 64) -> tuple[Path, np.ndarray, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.13, 0.00, 0.00],
            [0.17, 0.14, 0.00],
            [0.18, 0.16, 0.15],
            [0.30, 0.15, 0.02],
            [0.42, 0.17, 0.01],
        ],
        dtype=np.float64,
    )
    frame_index = np.arange(frames, dtype=np.float64)
    scale = 1.0 + 0.001 * frame_index[:, None, None]
    translation = np.stack(
        (
            1.80 + 0.002 * frame_index,
            1.85 - 0.001 * frame_index,
            1.90 + 0.0005 * frame_index,
        ),
        axis=-1,
    )[:, None, :]
    coordinates = (base[None, ...] * scale + translation).astype(np.float32)
    box = np.broadcast_to(np.eye(3) * 3.70465, (frames, 3, 3)).copy()
    species = np.broadcast_to(np.asarray([1, 3, 1, 1, 1, 3]), (frames, 6)).copy()
    mask = np.ones((frames, 6), dtype=bool)
    np.savez(path, R=coordinates, box=box, species=species, mask=mask)
    centred = coordinates.astype(np.float64) - coordinates.astype(np.float64).mean(axis=1, keepdims=True)
    return path, coordinates, float(centred.std())


def _load(path: Path, runtime_std: float | None = None):
    return load_ala2_warmstart_dataset(
        path,
        allowed_training_path=path,
        expected_sha256=_sha256(path),
        runtime_std_nm=runtime_std,
    )


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    displacement = coordinates[:, :, None, :] - coordinates[:, None, :, :]
    return np.linalg.norm(displacement, axis=-1)


def _signed_volume(coordinates: np.ndarray) -> np.ndarray:
    first = coordinates[:, 1] - coordinates[:, 0]
    second = coordinates[:, 2] - coordinates[:, 0]
    third = coordinates[:, 3] - coordinates[:, 0]
    return np.einsum("bi,bi->b", first, np.cross(second, third))


def test_load_computes_full_data_geometric_center_std_and_validates_runtime(tmp_path: Path) -> None:
    path, coordinates, expected_std = _write_archive(tmp_path / "flow_b" / "data.npz")
    dataset = _load(path, expected_std)

    assert dataset.coordinates_nm.shape == (64, 6, 3)
    assert dataset.standardization_std_nm == pytest.approx(expected_std, rel=0.0, abs=0.0)
    np.testing.assert_allclose(dataset.box_nm, np.eye(3) * 3.70465)
    np.testing.assert_array_equal(dataset.species, [1, 3, 1, 1, 1, 3])
    assert dataset.mask.all()
    np.testing.assert_array_equal(dataset.coordinates_nm, coordinates)

    with pytest.raises(ValueError, match="standardization differs"):
        _load(path, expected_std * 1.01)


def test_so3_augmentation_preserves_distances_and_chirality(tmp_path: Path) -> None:
    path, coordinates, expected_std = _write_archive(tmp_path / "flow_b" / "data.npz")
    dataset = _load(path)
    indices = np.asarray([0, 3, 9, 17, 31, 63])
    batch = dataset.prepare_indices(indices, augmentation_seed=291, dtype=np.float64)

    original = coordinates[indices].astype(np.float64)
    original -= original.mean(axis=1, keepdims=True)
    original /= expected_std
    transformed_shape = batch.endpoint_1 - batch.endpoint_1.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(
        _pairwise_distances(transformed_shape),
        _pairwise_distances(original),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        np.sign(_signed_volume(transformed_shape)),
        np.sign(_signed_volume(original)),
    )
    np.testing.assert_allclose(
        np.einsum("bij,bkj->bik", batch.rotations, batch.rotations),
        np.broadcast_to(np.eye(3), batch.rotations.shape),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.linalg.det(batch.rotations), 1.0, rtol=1.0e-12, atol=1.0e-12)


def test_shared_com_noise_has_cgbg_statistics_and_zero_mean_shape(tmp_path: Path) -> None:
    path, _coordinates, _expected_std = _write_archive(tmp_path / "flow_b" / "data.npz")
    dataset = _load(path)
    batch = dataset.sample_batch(
        np.asarray([0]),
        50_000,
        index_seed=7,
        augmentation_seed=11,
        replace=True,
    )

    centre = batch.endpoint_1.mean(axis=1)
    np.testing.assert_allclose(centre, batch.com_noise[:, 0], rtol=0.0, atol=3.0e-7)
    np.testing.assert_allclose(centre.mean(axis=0), 0.0, atol=8.0e-3)
    np.testing.assert_allclose(centre.std(axis=0), 1.0 / np.sqrt(6.0), rtol=1.5e-2, atol=0.0)
    shape = batch.endpoint_1 - centre[:, None, :]
    np.testing.assert_allclose(shape.mean(axis=1), 0.0, atol=3.0e-7)


def test_hash_and_training_asset_boundaries_reject_reference_or_tampering(tmp_path: Path) -> None:
    train_path, _coordinates, _expected_std = _write_archive(tmp_path / "flow_b" / "data.npz")
    reference_path, _reference, _reference_std = _write_archive(tmp_path / "flow_ub" / "data.npz", frames=32)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_ala2_warmstart_dataset(
            train_path,
            allowed_training_path=train_path,
            expected_sha256="0" * 64,
        )
    with pytest.raises(PermissionError, match="configured training asset"):
        load_ala2_warmstart_dataset(
            reference_path,
            allowed_training_path=train_path,
            forbidden_reference_paths=(reference_path,),
            expected_sha256=_sha256(train_path),
        )
    with pytest.raises(PermissionError, match="reference/evaluation"):
        load_ala2_warmstart_dataset(
            train_path,
            allowed_training_path=train_path,
            forbidden_reference_paths=(train_path,),
            expected_sha256=_sha256(train_path),
        )


def test_split_and_index_sampling_are_deterministic_and_disjoint(tmp_path: Path) -> None:
    path, _coordinates, _expected_std = _write_archive(tmp_path / "flow_b" / "data.npz", frames=101)
    dataset = _load(path)
    split_a = dataset.split_indices(seed=19, validation_fraction=0.2)
    split_b = dataset.split_indices(seed=19, validation_fraction=0.2)
    split_c = dataset.split_indices(seed=20, validation_fraction=0.2)

    np.testing.assert_array_equal(split_a.train, split_b.train)
    np.testing.assert_array_equal(split_a.validation, split_b.validation)
    assert not np.array_equal(split_a.train, split_c.train)
    assert len(split_a.validation) == 20
    assert set(split_a.train).isdisjoint(set(split_a.validation))
    np.testing.assert_array_equal(
        np.sort(np.concatenate((split_a.train, split_a.validation))),
        np.arange(dataset.num_frames),
    )

    batch_a = dataset.sample_batch(
        split_a.train,
        16,
        index_seed=31,
        augmentation_seed=37,
    )
    batch_b = dataset.sample_batch(
        split_a.train,
        16,
        index_seed=31,
        augmentation_seed=37,
    )
    batch_new_indices = dataset.sample_batch(
        split_a.train,
        16,
        index_seed=32,
        augmentation_seed=37,
    )
    batch_new_augmentation = dataset.sample_batch(
        split_a.train,
        16,
        index_seed=31,
        augmentation_seed=38,
    )
    np.testing.assert_array_equal(batch_a.indices, batch_b.indices)
    np.testing.assert_array_equal(batch_a.endpoint_1, batch_b.endpoint_1)
    assert not np.array_equal(batch_a.indices, batch_new_indices.indices)
    np.testing.assert_array_equal(batch_a.indices, batch_new_augmentation.indices)
    assert not np.array_equal(batch_a.endpoint_1, batch_new_augmentation.endpoint_1)


def test_uniform_so3_empty_batch_is_well_defined() -> None:
    rotations = uniform_so3_matrices(0, np.random.default_rng(3))
    assert rotations.shape == (0, 3, 3)


def test_singleton_leading_archive_metadata_is_supported(tmp_path: Path) -> None:
    path, coordinates, _expected_std = _write_archive(tmp_path / "source" / "data.npz", frames=7)
    with np.load(path, allow_pickle=False) as archive:
        box = archive["box"][0:1]
        species = archive["species"][0:1]
        mask = archive["mask"][0:1]
    singleton_path = tmp_path / "flow_b" / "data.npz"
    singleton_path.parent.mkdir(parents=True)
    np.savez(singleton_path, R=coordinates, box=box, species=species, mask=mask)

    dataset = _load(singleton_path)
    np.testing.assert_allclose(dataset.box_nm, np.eye(3) * 3.70465)
    np.testing.assert_array_equal(dataset.species, [1, 3, 1, 1, 1, 3])
    assert dataset.mask.all()
