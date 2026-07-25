"""Leakage-safe CG-BG Ala2 endpoint preparation for bridge warm starts.

CG-BG's Ala2 flow loader applies the following transform to the Cartesian
``R`` array stored in ``flow_b/data.npz``:

1. subtract each frame's *geometric* centre;
2. divide by one scalar standard deviation computed over the complete,
   centred training archive;
3. add one shared Gaussian translation with per-axis standard deviation
   ``1 / sqrt(N)``;
4. rotate the centred shape by an independent Haar-uniform SO(3) rotation,
   leaving the sampled shared translation unchanged.

This module reproduces that convention without importing Torch or CG-BG.  It
does not unwrap or wrap coordinates: the published ``flow_b`` archive already
contains contiguous Cartesian molecules in nm, and upstream CG-BG performs no
PBC operation in its flow data loader.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _validate_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    return normalized


def _constant_archive_field(
    value: np.ndarray,
    *,
    item_shape: tuple[int, ...],
    frame_count: int,
    label: str,
) -> np.ndarray:
    """Accept CG-BG's item, singleton-leading, or frame-leading metadata."""

    array = np.asarray(value)
    if array.shape == item_shape:
        return array
    if array.shape == (1, *item_shape):
        return array[0]
    if array.shape != (frame_count, *item_shape):
        raise ValueError(
            f"{label} must have shape {item_shape}, {(1, *item_shape)}, or "
            f"{(frame_count, *item_shape)}; got {array.shape}"
        )
    if not np.all(array == array[0]):
        raise ValueError(f"{label} must be constant across the CG-BG archive")
    return array[0]


class Ala2WarmStartSplit(NamedTuple):
    """Disjoint deterministic indices into one permitted ``flow_b`` asset."""

    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True)
class Ala2WarmStartBatch:
    """One transformed warm-start endpoint batch and its audit information."""

    endpoint_1: np.ndarray
    indices: np.ndarray
    rotations: np.ndarray
    com_noise: np.ndarray


@dataclass(frozen=True)
class Ala2WarmStartDataset:
    """Validated, in-memory view of the configured Ala2 warm-start asset.

    ``coordinates_nm`` is the raw Cartesian ``R`` array.  Callers should use
    :meth:`prepare_indices` or :meth:`sample_batch` to obtain the standardized
    ambient-18 endpoints consumed by the controller.
    """

    path: Path
    sha256: str
    coordinates_nm: np.ndarray
    box_nm: np.ndarray
    species: np.ndarray
    mask: np.ndarray
    standardization_std_nm: float

    @property
    def num_frames(self) -> int:
        return int(self.coordinates_nm.shape[0])

    @property
    def n_beads(self) -> int:
        return int(self.coordinates_nm.shape[1])

    @property
    def com_std(self) -> float:
        """CG-BG shared-COM standard deviation in standardized coordinates."""

        return self.n_beads**-0.5

    def split_indices(
        self,
        *,
        seed: int,
        validation_fraction: float = 0.1,
    ) -> Ala2WarmStartSplit:
        """Return a seeded, disjoint and exhaustive train/validation split."""

        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in [0, 1)")
        permutation = np.random.default_rng(seed).permutation(self.num_frames).astype(np.int64, copy=False)
        validation_size = int(math.floor(self.num_frames * validation_fraction))
        validation = permutation[:validation_size]
        train = permutation[validation_size:]
        train.setflags(write=False)
        validation.setflags(write=False)
        return Ala2WarmStartSplit(train=train, validation=validation)

    def prepare_indices(
        self,
        indices: np.ndarray,
        *,
        augmentation_seed: int,
        dtype: np.dtype | type = np.float32,
    ) -> Ala2WarmStartBatch:
        """Prepare selected frames exactly in CG-BG's ambient convention.

        The supplied indices determine data selection and ``augmentation_seed``
        determines only COM noise and rotations.  Keeping these two sources of
        randomness separate makes resumed runs and validation reproducible.
        """

        selected = np.asarray(indices, dtype=np.int64).copy()
        if selected.ndim != 1:
            raise ValueError("indices must be a one-dimensional integer array")
        if np.any(selected < 0) or np.any(selected >= self.num_frames):
            raise IndexError("warm-start indices are outside the configured training asset")

        output_dtype = np.dtype(dtype)
        if output_dtype.kind != "f":
            raise TypeError("dtype must be a floating-point dtype")

        # Use float64 for centre removal and scaling.  The serialized endpoint
        # is cast only after the transform, matching the runtime scalar std.
        physical = np.asarray(self.coordinates_nm[selected], dtype=np.float64)
        shape = physical - physical.mean(axis=1, keepdims=True)
        standardized_shape = shape / self.standardization_std_nm

        rng = np.random.default_rng(augmentation_seed)
        com_noise = rng.standard_normal(size=(selected.size, 1, 3)) * self.com_std
        ambient = standardized_shape + com_noise

        rotations = uniform_so3_matrices(selected.size, rng)
        offset = ambient.mean(axis=1, keepdims=True)
        rotated = np.einsum("bij,bnj->bni", rotations, ambient - offset) + offset

        endpoint = np.asarray(rotated, dtype=output_dtype)
        rotations = np.asarray(rotations, dtype=output_dtype)
        com_noise = np.asarray(com_noise, dtype=output_dtype)
        for value in (endpoint, selected, rotations, com_noise):
            value.setflags(write=False)
        return Ala2WarmStartBatch(
            endpoint_1=endpoint,
            indices=selected,
            rotations=rotations,
            com_noise=com_noise,
        )

    def sample_batch(
        self,
        candidates: np.ndarray,
        batch_size: int,
        *,
        index_seed: int,
        augmentation_seed: int,
        replace: bool = False,
        dtype: np.dtype | type = np.float32,
    ) -> Ala2WarmStartBatch:
        """Deterministically sample indices, then independently augment them."""

        pool = np.asarray(candidates, dtype=np.int64)
        if pool.ndim != 1 or pool.size == 0:
            raise ValueError("candidates must be a non-empty one-dimensional array")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not replace and batch_size > pool.size:
            raise ValueError("batch_size exceeds candidate count with replace=False")
        chosen = np.random.default_rng(index_seed).choice(pool, size=batch_size, replace=replace)
        return self.prepare_indices(chosen, augmentation_seed=augmentation_seed, dtype=dtype)


def uniform_so3_matrices(batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Haar-uniform proper rotations using CG-BG's quaternion method."""

    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if batch_size == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    quaternion = rng.standard_normal(size=(batch_size, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = (quaternion[:, index] for index in range(4))

    rotation = np.empty((batch_size, 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1.0 - 2.0 * y**2 - 2.0 * z**2
    rotation[:, 0, 1] = 2.0 * x * y - 2.0 * w * z
    rotation[:, 0, 2] = 2.0 * x * z + 2.0 * w * y
    rotation[:, 1, 0] = 2.0 * x * y + 2.0 * w * z
    rotation[:, 1, 1] = 1.0 - 2.0 * x**2 - 2.0 * z**2
    rotation[:, 1, 2] = 2.0 * y * z - 2.0 * w * x
    rotation[:, 2, 0] = 2.0 * x * z - 2.0 * w * y
    rotation[:, 2, 1] = 2.0 * y * z + 2.0 * w * x
    rotation[:, 2, 2] = 1.0 - 2.0 * x**2 - 2.0 * y**2
    return rotation


def load_ala2_warmstart_dataset(
    path: str | Path,
    *,
    allowed_training_path: str | Path,
    expected_sha256: str,
    forbidden_reference_paths: tuple[str | Path, ...] = (),
    runtime_std_nm: float | None = None,
    std_rtol: float = 5.0e-7,
    std_atol: float = 1.0e-10,
) -> Ala2WarmStartDataset:
    """Load only the configured, hash-pinned ``flow_b`` training archive.

    ``allowed_training_path`` must come from ``experiment.assets.train`` and
    ``forbidden_reference_paths`` should contain all evaluation/reference
    assets from the same config.  This explicit path boundary, followed by the
    pinned SHA-256 check, prevents an accidental warm start on ``flow_ub``.
    """

    actual_path = _resolved(path)
    allowed_path = _resolved(allowed_training_path)
    if actual_path != allowed_path:
        raise PermissionError(
            f"Warm-start input must equal the configured training asset: {allowed_path}"
        )
    forbidden = {_resolved(candidate) for candidate in forbidden_reference_paths}
    if actual_path in forbidden:
        raise PermissionError("A configured reference/evaluation asset cannot be used for warm start")
    if not actual_path.is_file():
        raise FileNotFoundError(actual_path)

    expected = _validate_sha256(expected_sha256)
    actual_sha = _sha256_file(actual_path)
    if actual_sha != expected:
        raise ValueError(f"Warm-start asset SHA-256 mismatch: expected {expected}, got {actual_sha}")

    with np.load(actual_path, allow_pickle=False) as archive:
        missing = {"R", "box", "species", "mask"}.difference(archive.files)
        if missing:
            raise KeyError(f"CG-BG warm-start archive is missing {sorted(missing)}")
        coordinates = np.asarray(archive["R"])
        boxes = np.asarray(archive["box"])
        species_all = np.asarray(archive["species"])
        masks_all = np.asarray(archive["mask"])

    if coordinates.ndim != 3 or coordinates.shape[1:] != (6, 3):
        raise ValueError(f"Expected Cartesian Ala2 R with shape (B,6,3), got {coordinates.shape}")
    if coordinates.dtype.kind != "f" or not np.all(np.isfinite(coordinates)):
        raise ValueError("Ala2 R must contain finite floating-point Cartesian coordinates in nm")
    frame_count = coordinates.shape[0]
    box = np.asarray(
        _constant_archive_field(
            boxes,
            item_shape=(3, 3),
            frame_count=frame_count,
            label="box",
        ),
        dtype=np.float64,
    )
    species = np.asarray(
        _constant_archive_field(
            species_all,
            item_shape=(6,),
            frame_count=frame_count,
            label="species",
        ),
        dtype=np.int64,
    )
    mask = np.asarray(
        _constant_archive_field(
            masks_all,
            item_shape=(6,),
            frame_count=frame_count,
            label="mask",
        ),
        dtype=bool,
    )
    if not np.allclose(box, np.diag(np.diag(box)), rtol=0.0, atol=0.0):
        raise ValueError("Only the orthorhombic CG-BG Ala2 box is supported")
    if np.any(np.diag(box) <= 0.0):
        raise ValueError("The CG-BG box lengths must be positive")
    if not np.all(mask):
        raise ValueError("The six-bead warm-start archive must contain six valid beads per frame")

    physical64 = np.asarray(coordinates, dtype=np.float64)
    centred = physical64 - physical64.mean(axis=1, keepdims=True)
    computed_std = float(centred.std())
    if not math.isfinite(computed_std) or computed_std <= 0.0:
        raise ValueError("The full-data centred coordinate standard deviation must be positive")
    if runtime_std_nm is not None:
        runtime_std = float(runtime_std_nm)
        if not math.isfinite(runtime_std) or runtime_std <= 0.0:
            raise ValueError("runtime_std_nm must be finite and positive")
        if not np.isclose(runtime_std, computed_std, rtol=std_rtol, atol=std_atol):
            raise ValueError(
                "Warm-start standardization differs from the runtime PMF bundle: "
                f"computed={computed_std:.17g} nm, runtime={runtime_std:.17g} nm"
            )
        standardization_std = runtime_std
    else:
        standardization_std = computed_std

    # Keep only the arrays used after validation, and make accidental mutation
    # of the hash-identified dataset fail loudly.
    coordinates = np.asarray(coordinates)
    for value in (coordinates, box, species, mask):
        value.setflags(write=False)
    return Ala2WarmStartDataset(
        path=actual_path,
        sha256=actual_sha,
        coordinates_nm=coordinates,
        box_nm=box,
        species=species,
        mask=mask,
        standardization_std_nm=standardization_std,
    )


__all__ = [
    "Ala2WarmStartBatch",
    "Ala2WarmStartDataset",
    "Ala2WarmStartSplit",
    "load_ala2_warmstart_dataset",
    "uniform_so3_matrices",
]
