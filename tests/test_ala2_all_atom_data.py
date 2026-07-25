from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from cg_bms_jax.data import (
    ALA2_BMS_ATOMIC_NUMBERS,
    ALA2_TAM_ATOMIC_NUMBERS,
    ALA2_TAM_TO_BMS_GATHER,
    load_ala2_all_atom_dataset,
    reorder_ala2_bms_to_tam,
    reorder_ala2_tam_to_bms,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory(frames: int) -> np.ndarray:
    base = np.asarray(
        [
            [0.000, 0.000, 0.000],
            [0.105, 0.010, 0.000],
            [0.020, 0.110, 0.005],
            [0.010, 0.015, 0.120],
        ]
        + [
            [0.02 * atom, 0.01 * (atom % 4), -0.005 * atom]
            for atom in range(4, 22)
        ],
        dtype=np.float64,
    )
    frame = np.arange(frames, dtype=np.float64)
    translation = np.stack(
        (
            1.0 + 0.002 * frame,
            -0.5 + 0.001 * frame,
            0.7 - 0.003 * frame,
        ),
        axis=-1,
    )[:, None, :]
    scale = 1.0 + 0.0005 * frame[:, None, None]
    return (base[None] * scale + translation).astype(np.float32)


def _write_archive(
    path: Path,
    *,
    frames: int = 30,
    flattened: bool = False,
    species: np.ndarray | None = None,
) -> tuple[Path, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates = _trajectory(frames)
    stored = coordinates.reshape(frames, 66) if flattened else coordinates
    if species is None:
        species = np.broadcast_to(
            np.asarray(ALA2_TAM_ATOMIC_NUMBERS, dtype=np.int64),
            (frames, 22),
        ).copy()
    np.savez(path, R=stored, species=species)
    return path, coordinates


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    delta = coordinates[:, :, None, :] - coordinates[:, None, :, :]
    return np.linalg.norm(delta, axis=-1)


def _signed_volume(coordinates: np.ndarray) -> np.ndarray:
    first = coordinates[:, 1] - coordinates[:, 0]
    second = coordinates[:, 2] - coordinates[:, 0]
    third = coordinates[:, 3] - coordinates[:, 0]
    return np.einsum("bi,bi->b", first, np.cross(second, third))


@pytest.mark.parametrize("flattened", [False, True])
def test_loader_converts_nm_to_angstrom_and_gathers_official_bms_order(
    tmp_path: Path,
    flattened: bool,
) -> None:
    path, source = _write_archive(
        tmp_path / "implicit_obc1_openmm.npz",
        frames=7,
        flattened=flattened,
    )
    dataset = load_ala2_all_atom_dataset(
        path,
        expected_sha256=_sha256(path),
        dtype=np.float64,
    )

    # The loader honours the requested float64 dtype before converting nm to
    # Angstrom, so the exact reference must perform the multiplication in the
    # same dtype rather than round once in float32.
    expected = (
        source[:, ALA2_TAM_TO_BMS_GATHER].astype(np.float64) * 10.0
    )
    np.testing.assert_allclose(dataset.coordinates_angstrom, expected, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(dataset.species, ALA2_BMS_ATOMIC_NUMBERS)
    assert dataset.coordinates_angstrom.shape == (7, 22, 3)
    assert dataset.coordinates_angstrom.dtype == np.float64
    assert dataset.auxiliary_com_std_angstrom == pytest.approx(1.0 / np.sqrt(22.0))
    assert not dataset.coordinates_angstrom.flags.writeable
    assert not dataset.species.flags.writeable


def test_public_atom_permutations_are_exact_inverses() -> None:
    tam = np.arange(22 * 3).reshape(2, 11, 3)
    tam = tam.reshape(22, 3)
    bms = reorder_ala2_tam_to_bms(tam)
    recovered = reorder_ala2_bms_to_tam(bms)
    np.testing.assert_array_equal(recovered, tam)

    tam_species = np.asarray(ALA2_TAM_ATOMIC_NUMBERS)
    bms_species = reorder_ala2_tam_to_bms(tam_species, atom_axis=0)
    np.testing.assert_array_equal(bms_species, ALA2_BMS_ATOMIC_NUMBERS)


def test_time_block_split_is_deterministic_disjoint_and_has_explicit_gaps(
    tmp_path: Path,
) -> None:
    path, _ = _write_archive(tmp_path / "implicit_obc1_openmm.npz", frames=30)
    dataset = load_ala2_all_atom_dataset(path)
    first = dataset.split_time_blocks(
        validation_fraction=0.2,
        test_fraction=0.2,
        gap_frames=2,
    )
    second = dataset.split_time_blocks(
        validation_fraction=0.2,
        test_fraction=0.2,
        gap_frames=2,
    )

    np.testing.assert_array_equal(first.train, np.arange(0, 16))
    np.testing.assert_array_equal(first.validation, np.arange(18, 23))
    np.testing.assert_array_equal(first.test, np.arange(25, 30))
    np.testing.assert_array_equal(first.gap, [16, 17, 23, 24])
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
        assert not left.flags.writeable
    np.testing.assert_array_equal(
        np.sort(np.concatenate(first)),
        np.arange(dataset.num_frames),
    )
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)


def test_time_block_split_supports_one_holdout_and_rejects_impossible_gaps(
    tmp_path: Path,
) -> None:
    path, _ = _write_archive(tmp_path / "implicit_obc1_openmm.npz", frames=10)
    dataset = load_ala2_all_atom_dataset(path)
    split = dataset.split_time_blocks(
        validation_fraction=0.0,
        test_fraction=0.25,
        gap_frames=2,
    )
    np.testing.assert_array_equal(split.train, np.arange(0, 6))
    np.testing.assert_array_equal(split.gap, [6, 7])
    np.testing.assert_array_equal(split.test, [8, 9])
    assert split.validation.size == 0

    with pytest.raises(ValueError, match="leave no usable"):
        dataset.split_time_blocks(
            validation_fraction=0.2,
            test_fraction=0.2,
            gap_frames=5,
        )
    with pytest.raises(ValueError, match="less than 1"):
        dataset.split_time_blocks(
            validation_fraction=0.5,
            test_fraction=0.5,
        )


def test_augmentation_centres_rotates_and_adds_auxiliary_com(
    tmp_path: Path,
) -> None:
    path, source = _write_archive(tmp_path / "implicit_obc1_openmm.npz", frames=12)
    dataset = load_ala2_all_atom_dataset(
        path,
        dtype=np.float64,
        auxiliary_com_std_angstrom=0.25,
    )
    indices = np.asarray([0, 3, 7, 11])
    first = dataset.prepare_indices(
        indices,
        augmentation_seed=91,
        dtype=np.float64,
    )
    repeated = dataset.prepare_indices(
        indices,
        augmentation_seed=91,
        dtype=np.float64,
    )
    changed = dataset.prepare_indices(
        indices,
        augmentation_seed=92,
        dtype=np.float64,
    )

    canonical = source[indices][:, ALA2_TAM_TO_BMS_GATHER].astype(np.float64) * 10.0
    canonical -= canonical.mean(axis=1, keepdims=True)
    transformed_shape = first.endpoint_1 - first.endpoint_1.mean(
        axis=1, keepdims=True
    )
    np.testing.assert_allclose(
        _pairwise_distances(transformed_shape),
        _pairwise_distances(canonical),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        np.sign(_signed_volume(transformed_shape)),
        np.sign(_signed_volume(canonical)),
    )
    np.testing.assert_allclose(
        first.endpoint_1.mean(axis=1),
        first.com_noise_angstrom[:, 0],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.einsum("bij,bkj->bik", first.rotations, first.rotations),
        np.broadcast_to(np.eye(3), first.rotations.shape),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.linalg.det(first.rotations),
        1.0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(first.endpoint_1, repeated.endpoint_1)
    assert not np.array_equal(first.endpoint_1, changed.endpoint_1)


def test_batch_index_and_augmentation_randomness_are_separate(tmp_path: Path) -> None:
    path, _ = _write_archive(tmp_path / "implicit_obc1_openmm.npz", frames=40)
    dataset = load_ala2_all_atom_dataset(path)
    candidates = dataset.split_time_blocks(
        validation_fraction=0.1,
        test_fraction=0.1,
        gap_frames=1,
    ).train
    first = dataset.sample_batch(
        candidates,
        10,
        index_seed=5,
        augmentation_seed=7,
    )
    repeated = dataset.sample_batch(
        candidates,
        10,
        index_seed=5,
        augmentation_seed=7,
    )
    new_augmentation = dataset.sample_batch(
        candidates,
        10,
        index_seed=5,
        augmentation_seed=8,
    )
    np.testing.assert_array_equal(first.indices, repeated.indices)
    np.testing.assert_array_equal(first.endpoint_1, repeated.endpoint_1)
    np.testing.assert_array_equal(first.indices, new_augmentation.indices)
    assert not np.array_equal(first.endpoint_1, new_augmentation.endpoint_1)


def test_loader_rejects_wrong_species_nonfinite_coordinates_and_bad_hash(
    tmp_path: Path,
) -> None:
    wrong_species = np.asarray(ALA2_TAM_ATOMIC_NUMBERS).copy()
    wrong_species[[0, 1]] = wrong_species[[1, 0]]
    wrong_path, _ = _write_archive(
        tmp_path / "wrong.npz",
        frames=3,
        species=wrong_species,
    )
    with pytest.raises(ValueError, match="documented AMBoltz/TAM"):
        load_ala2_all_atom_dataset(wrong_path)

    nonfinite_path, coordinates = _write_archive(tmp_path / "nonfinite.npz", frames=3)
    coordinates[1, 2, 0] = np.nan
    np.savez(
        nonfinite_path,
        R=coordinates,
        species=np.asarray(ALA2_TAM_ATOMIC_NUMBERS),
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_ala2_all_atom_dataset(nonfinite_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_ala2_all_atom_dataset(wrong_path, expected_sha256="0" * 64)
