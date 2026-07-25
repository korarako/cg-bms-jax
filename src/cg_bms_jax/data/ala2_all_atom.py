"""All-atom Ala2 trajectory preparation in the official BMS atom order.

The AMBoltz/TAM ``implicit_obc1_openmm.npz`` archive stores Cartesian
coordinates in nanometres and uses a different atom order from the Ala2
topology published with BridgeMatchingSampler (BMS).  This module makes that
boundary explicit:

* load only the ``R`` and ``species`` arrays;
* convert nanometres to Angstrom;
* gather atoms into the official BMS order;
* split the time-ordered trajectory into contiguous blocks separated by
  excluded gap frames;
* remove the geometric centre, apply a Haar-uniform proper rotation, and add a
  shared auxiliary Cartesian centre.

The auxiliary centre has standard deviation ``1 / sqrt(22)`` Angstrom by
default.  It therefore lifts the 63-dimensional centred molecular shape to a
full-rank 66-dimensional Cartesian endpoint without changing any internal
distance or chirality.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from cg_bms_jax.data.ala2_warmstart import uniform_so3_matrices

ALA2_NUM_ATOMS = 22
NM_TO_ANGSTROM = 10.0

# ``output[..., bms_index, :] = input[..., ALA2_TAM_TO_BMS_GATHER[bms_index], :]``.
# Keeping the complete gather in the data layer prevents silent reuse of TAM
# torsion/restraint indices after coordinates have entered the BMS model.
ALA2_TAM_TO_BMS_GATHER: tuple[int, ...] = (
    1,
    0,
    2,
    3,
    4,
    5,
    6,
    10,
    7,
    11,
    12,
    13,
    14,
    15,
    8,
    9,
    16,
    17,
    18,
    19,
    20,
    21,
)
ALA2_BMS_TO_TAM_GATHER: tuple[int, ...] = tuple(
    int(value) for value in np.argsort(np.asarray(ALA2_TAM_TO_BMS_GATHER))
)

ALA2_TAM_ATOM_NAMES: tuple[str, ...] = (
    "ACE-CH3",
    "ACE-H1",
    "ACE-H2",
    "ACE-H3",
    "ACE-C",
    "ACE-O",
    "ALA-N",
    "ALA-CA",
    "ALA-C",
    "ALA-O",
    "ALA-H",
    "ALA-HA",
    "ALA-CB",
    "ALA-HB1",
    "ALA-HB2",
    "ALA-HB3",
    "NME-N",
    "NME-H",
    "NME-C",
    "NME-H1",
    "NME-H2",
    "NME-H3",
)
ALA2_BMS_ATOM_NAMES: tuple[str, ...] = tuple(
    ALA2_TAM_ATOM_NAMES[index] for index in ALA2_TAM_TO_BMS_GATHER
)

ALA2_TAM_ATOMIC_NUMBERS: tuple[int, ...] = (
    6,
    1,
    1,
    1,
    6,
    8,
    7,
    6,
    6,
    8,
    1,
    1,
    6,
    1,
    1,
    1,
    7,
    1,
    6,
    1,
    1,
    1,
)
ALA2_BMS_ATOMIC_NUMBERS: tuple[int, ...] = tuple(
    ALA2_TAM_ATOMIC_NUMBERS[index] for index in ALA2_TAM_TO_BMS_GATHER
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    return normalized


def _normalize_atom_axis(array: np.ndarray, atom_axis: int) -> int:
    axis = int(atom_axis)
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim:
        raise np.AxisError(atom_axis, ndim=array.ndim)
    if array.shape[axis] != ALA2_NUM_ATOMS:
        raise ValueError(
            f"atom axis must have length {ALA2_NUM_ATOMS}, got shape {array.shape}"
        )
    return axis


def reorder_ala2_tam_to_bms(
    values: np.ndarray,
    *,
    atom_axis: int = -2,
) -> np.ndarray:
    """Gather an Ala2 array from the AMBoltz/TAM order into BMS order."""

    array = np.asarray(values)
    axis = _normalize_atom_axis(array, atom_axis)
    return np.take(array, ALA2_TAM_TO_BMS_GATHER, axis=axis)


def reorder_ala2_bms_to_tam(
    values: np.ndarray,
    *,
    atom_axis: int = -2,
) -> np.ndarray:
    """Gather an Ala2 array from the official BMS order back into TAM order."""

    array = np.asarray(values)
    axis = _normalize_atom_axis(array, atom_axis)
    return np.take(array, ALA2_BMS_TO_TAM_GATHER, axis=axis)


def _constant_species(value: np.ndarray, *, frame_count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (ALA2_NUM_ATOMS,):
        species = array
    elif array.shape == (1, ALA2_NUM_ATOMS):
        species = array[0]
    elif array.shape == (frame_count, ALA2_NUM_ATOMS):
        if not np.all(array == array[0]):
            raise ValueError("species must be constant across the Ala2 trajectory")
        species = array[0]
    else:
        raise ValueError(
            "species must have shape (22,), (1,22), or (num_frames,22); "
            f"got {array.shape}"
        )
    if species.dtype.kind not in "iu":
        raise TypeError("species must contain integer atomic numbers")
    return np.asarray(species, dtype=np.int64)


class Ala2AllAtomTimeSplit(NamedTuple):
    """Chronological blocks and the frames excluded between adjacent blocks."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    gap: np.ndarray


@dataclass(frozen=True)
class Ala2AllAtomBatch:
    """One augmented all-atom endpoint batch in Angstrom and BMS atom order."""

    endpoint_1: np.ndarray
    indices: np.ndarray
    rotations: np.ndarray
    com_noise_angstrom: np.ndarray


@dataclass(frozen=True)
class Ala2AllAtomDataset:
    """Validated in-memory view of an AMBoltz/TAM all-atom Ala2 trajectory."""

    path: Path
    sha256: str
    coordinates_angstrom: np.ndarray
    species: np.ndarray
    auxiliary_com_std_angstrom: float

    @property
    def num_frames(self) -> int:
        return int(self.coordinates_angstrom.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.coordinates_angstrom.shape[1])

    def split_time_blocks(
        self,
        *,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        gap_frames: int = 0,
    ) -> Ala2AllAtomTimeSplit:
        """Split one ordered trajectory into train/validation/test time blocks.

        Fractions are applied to the frames remaining after reserving one gap
        between each pair of non-empty adjacent blocks.  Any rounding remainder
        is assigned to the training block.  No random permutation is involved.
        """

        validation_fraction = float(validation_fraction)
        test_fraction = float(test_fraction)
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in [0, 1)")
        if not 0.0 <= test_fraction < 1.0:
            raise ValueError("test_fraction must lie in [0, 1)")
        if validation_fraction + test_fraction >= 1.0:
            raise ValueError("validation_fraction + test_fraction must be less than 1")
        if isinstance(gap_frames, bool) or int(gap_frames) != gap_frames:
            raise TypeError("gap_frames must be an integer")
        gap_frames = int(gap_frames)
        if gap_frames < 0:
            raise ValueError("gap_frames must be non-negative")

        active_holdouts = int(validation_fraction > 0.0) + int(test_fraction > 0.0)
        usable_frames = self.num_frames - active_holdouts * gap_frames
        if usable_frames <= 0:
            raise ValueError("gap_frames leave no usable trajectory frames")

        validation_size = int(math.floor(usable_frames * validation_fraction))
        test_size = int(math.floor(usable_frames * test_fraction))
        if validation_fraction > 0.0 and validation_size == 0:
            raise ValueError("trajectory is too short for a non-empty validation block")
        if test_fraction > 0.0 and test_size == 0:
            raise ValueError("trajectory is too short for a non-empty test block")
        train_size = usable_frames - validation_size - test_size
        if train_size <= 0:
            raise ValueError("trajectory split leaves no training frames")

        cursor = 0
        train = np.arange(cursor, cursor + train_size, dtype=np.int64)
        cursor += train_size
        gaps: list[np.ndarray] = []

        if validation_size:
            if gap_frames:
                gaps.append(np.arange(cursor, cursor + gap_frames, dtype=np.int64))
                cursor += gap_frames
            validation = np.arange(
                cursor, cursor + validation_size, dtype=np.int64
            )
            cursor += validation_size
        else:
            validation = np.empty(0, dtype=np.int64)

        if test_size:
            if gap_frames:
                gaps.append(np.arange(cursor, cursor + gap_frames, dtype=np.int64))
                cursor += gap_frames
            test = np.arange(cursor, cursor + test_size, dtype=np.int64)
            cursor += test_size
        else:
            test = np.empty(0, dtype=np.int64)

        if cursor != self.num_frames:
            raise RuntimeError(
                f"internal time-block accounting error: used {cursor}/{self.num_frames}"
            )
        gap = (
            np.concatenate(gaps)
            if gaps
            else np.empty(0, dtype=np.int64)
        )
        for value in (train, validation, test, gap):
            value.setflags(write=False)
        return Ala2AllAtomTimeSplit(
            train=train,
            validation=validation,
            test=test,
            gap=gap,
        )

    def prepare_indices(
        self,
        indices: np.ndarray,
        *,
        augmentation_seed: int,
        dtype: np.dtype | type = np.float32,
    ) -> Ala2AllAtomBatch:
        """Centre, rotate, and add an independent shared auxiliary centre."""

        selected = np.asarray(indices, dtype=np.int64).copy()
        if selected.ndim != 1:
            raise ValueError("indices must be a one-dimensional integer array")
        if np.any(selected < 0) or np.any(selected >= self.num_frames):
            raise IndexError("indices are outside the all-atom Ala2 trajectory")

        output_dtype = np.dtype(dtype)
        if output_dtype.kind != "f":
            raise TypeError("dtype must be a floating-point dtype")

        physical = np.asarray(self.coordinates_angstrom[selected], dtype=np.float64)
        shape = physical - physical.mean(axis=1, keepdims=True)
        rng = np.random.default_rng(augmentation_seed)
        rotations = uniform_so3_matrices(selected.size, rng)
        rotated_shape = np.einsum("bij,bnj->bni", rotations, shape)
        com_noise = rng.standard_normal(size=(selected.size, 1, 3))
        com_noise *= self.auxiliary_com_std_angstrom
        endpoint = rotated_shape + com_noise

        endpoint = np.asarray(endpoint, dtype=output_dtype)
        rotations = np.asarray(rotations, dtype=output_dtype)
        com_noise = np.asarray(com_noise, dtype=output_dtype)
        for value in (endpoint, selected, rotations, com_noise):
            value.setflags(write=False)
        return Ala2AllAtomBatch(
            endpoint_1=endpoint,
            indices=selected,
            rotations=rotations,
            com_noise_angstrom=com_noise,
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
    ) -> Ala2AllAtomBatch:
        """Select chronological-split indices, then independently augment."""

        pool = np.asarray(candidates, dtype=np.int64)
        if pool.ndim != 1 or pool.size == 0:
            raise ValueError("candidates must be a non-empty one-dimensional array")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not replace and batch_size > pool.size:
            raise ValueError("batch_size exceeds candidate count with replace=False")
        chosen = np.random.default_rng(index_seed).choice(
            pool, size=batch_size, replace=replace
        )
        return self.prepare_indices(
            chosen,
            augmentation_seed=augmentation_seed,
            dtype=dtype,
        )


def load_ala2_all_atom_dataset(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    dtype: np.dtype | type = np.float32,
    auxiliary_com_std_angstrom: float | None = None,
    validate_species: bool = True,
) -> Ala2AllAtomDataset:
    """Load ``R/species`` from an AMBoltz implicit-OBC1 Ala2 NPZ archive.

    ``R`` may have shape ``(B,22,3)`` or ``(B,66)`` and is interpreted as
    nanometres in TAM atom order.  Returned coordinates always have shape
    ``(B,22,3)``, unit Angstrom, and official BMS atom order.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual_sha256 = _sha256_file(resolved)
    if expected_sha256 is not None:
        expected = _validate_sha256(expected_sha256)
        if actual_sha256 != expected:
            raise ValueError(
                "All-atom Ala2 asset SHA-256 mismatch: "
                f"expected {expected}, got {actual_sha256}"
            )

    output_dtype = np.dtype(dtype)
    if output_dtype.kind != "f":
        raise TypeError("dtype must be a floating-point dtype")
    with np.load(resolved, allow_pickle=False) as archive:
        missing = {"R", "species"}.difference(archive.files)
        if missing:
            raise KeyError(f"All-atom Ala2 archive is missing {sorted(missing)}")
        coordinates = np.asarray(archive["R"])
        species_all = np.asarray(archive["species"])

    if coordinates.ndim == 2 and coordinates.shape[1] == ALA2_NUM_ATOMS * 3:
        coordinates = coordinates.reshape((-1, ALA2_NUM_ATOMS, 3))
    if coordinates.ndim != 3 or coordinates.shape[1:] != (ALA2_NUM_ATOMS, 3):
        raise ValueError(
            "R must have shape (num_frames,22,3) or (num_frames,66); "
            f"got {coordinates.shape}"
        )
    if coordinates.dtype.kind != "f":
        raise TypeError("R must contain floating-point coordinates in nanometres")
    if coordinates.shape[0] == 0:
        raise ValueError("R must contain at least one trajectory frame")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("R contains non-finite coordinates")

    source_species = _constant_species(
        species_all,
        frame_count=int(coordinates.shape[0]),
    )
    if validate_species and not np.array_equal(
        source_species,
        np.asarray(ALA2_TAM_ATOMIC_NUMBERS),
    ):
        raise ValueError(
            "species does not match the documented AMBoltz/TAM Ala2 atom order"
        )

    coordinates = np.asarray(
        reorder_ala2_tam_to_bms(coordinates, atom_axis=1),
        dtype=output_dtype,
    )
    # Cast before scaling.  Multiplying float32 input and only then promoting
    # to float64 loses roughly 1e-6 Angstrom and breaks strict rotation audits.
    coordinates = coordinates * np.asarray(
        NM_TO_ANGSTROM,
        dtype=output_dtype,
    )
    species = np.asarray(
        reorder_ala2_tam_to_bms(source_species, atom_axis=0),
        dtype=np.int64,
    )

    if auxiliary_com_std_angstrom is None:
        com_std = ALA2_NUM_ATOMS**-0.5
    else:
        com_std = float(auxiliary_com_std_angstrom)
    if not math.isfinite(com_std) or com_std <= 0.0:
        raise ValueError("auxiliary_com_std_angstrom must be finite and positive")

    for value in (coordinates, species):
        value.setflags(write=False)
    return Ala2AllAtomDataset(
        path=resolved,
        sha256=actual_sha256,
        coordinates_angstrom=coordinates,
        species=species,
        auxiliary_com_std_angstrom=com_std,
    )


__all__ = [
    "ALA2_BMS_ATOM_NAMES",
    "ALA2_BMS_ATOMIC_NUMBERS",
    "ALA2_BMS_TO_TAM_GATHER",
    "ALA2_NUM_ATOMS",
    "ALA2_TAM_ATOM_NAMES",
    "ALA2_TAM_ATOMIC_NUMBERS",
    "ALA2_TAM_TO_BMS_GATHER",
    "Ala2AllAtomBatch",
    "Ala2AllAtomDataset",
    "Ala2AllAtomTimeSplit",
    "NM_TO_ANGSTROM",
    "load_ala2_all_atom_dataset",
    "reorder_ala2_bms_to_tam",
    "reorder_ala2_tam_to_bms",
]
