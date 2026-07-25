"""Versioned equilibrium endpoint datasets for the analytic MB2D target.

The bridge-matching baseline needs target-distributed endpoints, but an MD
trajectory is unnecessary for the analytic two-dimensional Muller--Brown
system.  This module samples a high-resolution midpoint quadrature of the
finite-box Boltzmann law and jitters uniformly inside the selected cells.

The resulting law is a documented numerical approximation to the continuous
analytic target, not a finite-time MD trajectory.  Every dataset records the
target, coordinate transform, grid resolution, split membership, and RNG seed
inside the NPZ.  The complete file is SHA-256 pinned by experiment configs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

EQUILIBRIUM_ENDPOINT_ABI = "equilibrium_endpoint_npz_v1"
EQUILIBRIUM_GENERATOR_ABI = "midpoint_categorical_uniform_jitter_v1"
_SPLIT_NAMES = ("train", "validation", "test")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _pair(value: Sequence[float], *, name: str) -> tuple[float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 2 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain exactly two finite values")
    return result


def _box(
    value: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    result = tuple(tuple(float(bound) for bound in axis) for axis in value)
    if len(result) != 2 or any(len(axis) != 2 for axis in result):
        raise ValueError("physical_box must have shape (2, 2)")
    if any(
        not all(math.isfinite(bound) for bound in axis) or axis[0] >= axis[1]
        for axis in result
    ):
        raise ValueError("physical_box bounds must be finite and increasing")
    return result  # type: ignore[return-value]


def mb2d_target_spec(target: Any) -> dict[str, Any]:
    """Return the immutable target/coordinate identity used by a dataset."""

    try:
        beta = float(target.beta)
        offset = _pair(target.offset, name="offset")
        scale = _pair(target.scale, name="scale")
        physical_box = _box(target.physical_box)
    except AttributeError as error:
        raise TypeError(
            "target must expose beta, offset, scale and physical_box"
        ) from error
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("target beta must be finite and positive")
    if any(value <= 0.0 for value in scale):
        raise ValueError("target scale must be strictly positive")
    return {
        "implementation_abi": "analytic_muller_brown_finite_box_v1",
        "energy": "cg_bg_muller_brown_unbiased",
        "beta": beta,
        "affine_offset": list(offset),
        "affine_scale": list(scale),
        "physical_box": [list(axis) for axis in physical_box],
    }


@dataclass(frozen=True)
class MB2DEquilibriumDataset:
    """In-memory representation of one hash-pinned endpoint dataset."""

    state: np.ndarray
    physical: np.ndarray
    reduced_energy: np.ndarray
    split: np.ndarray
    target_beta: float
    affine_offset: tuple[float, float]
    affine_scale: tuple[float, float]
    physical_box: tuple[tuple[float, float], tuple[float, float]]
    grid_resolution: tuple[int, int]
    seed: int
    target_signature: str
    generator_abi: str = EQUILIBRIUM_GENERATOR_ABI
    sha256: str | None = None
    path: Path | None = None

    def __post_init__(self) -> None:
        state = np.asarray(self.state)
        physical = np.asarray(self.physical)
        reduced_energy = np.asarray(self.reduced_energy)
        split = np.asarray(self.split)
        if state.ndim != 2 or state.shape[1] != 2:
            raise ValueError("state must have shape (N, 2)")
        if physical.shape != state.shape:
            raise ValueError("physical must have the same shape as state")
        if reduced_energy.shape != (state.shape[0],):
            raise ValueError("reduced_energy must have shape (N,)")
        if split.shape != (state.shape[0],):
            raise ValueError("split must have shape (N,)")
        if state.shape[0] == 0:
            raise ValueError("an equilibrium endpoint dataset cannot be empty")
        if not (
            np.all(np.isfinite(state))
            and np.all(np.isfinite(physical))
            and np.all(np.isfinite(reduced_energy))
        ):
            raise ValueError("endpoint coordinates and energies must be finite")
        split_values = {str(value) for value in np.unique(split)}
        if not split_values or not split_values.issubset(set(_SPLIT_NAMES)):
            raise ValueError(
                f"split values must be drawn from {_SPLIT_NAMES}, got {split_values}"
            )

        beta = float(self.target_beta)
        offset = _pair(self.affine_offset, name="affine_offset")
        scale = _pair(self.affine_scale, name="affine_scale")
        box = _box(self.physical_box)
        resolution = tuple(int(value) for value in self.grid_resolution)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("target_beta must be finite and positive")
        if any(value <= 0.0 for value in scale):
            raise ValueError("affine_scale must be strictly positive")
        if len(resolution) != 2 or any(value < 2 for value in resolution):
            raise ValueError("grid_resolution must contain two integers >= 2")
        if self.generator_abi != EQUILIBRIUM_GENERATOR_ABI:
            raise ValueError(
                f"generator_abi must be {EQUILIBRIUM_GENERATOR_ABI!r}"
            )
        if len(self.target_signature) != 64:
            raise ValueError("target_signature must be a SHA-256 digest")
        if self.sha256 is not None and len(self.sha256) != 64:
            raise ValueError("sha256 must be a SHA-256 digest")
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path).resolve())

        reconstructed = (
            state.astype(np.float64) * np.asarray(scale, dtype=np.float64)
            + np.asarray(offset, dtype=np.float64)
        )
        if not np.allclose(
            reconstructed,
            physical.astype(np.float64),
            rtol=0.0,
            atol=8.0e-5,
        ):
            raise ValueError("state/physical arrays violate the affine transform")
        lower = np.asarray([axis[0] for axis in box], dtype=np.float64)
        upper = np.asarray([axis[1] for axis in box], dtype=np.float64)
        if not np.all(
            (physical.astype(np.float64) >= lower)
            & (physical.astype(np.float64) <= upper)
        ):
            raise ValueError("equilibrium endpoints must lie inside physical_box")

        for value in (state, physical, reduced_energy, split):
            value.setflags(write=False)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "physical", physical)
        object.__setattr__(self, "reduced_energy", reduced_energy)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "target_beta", beta)
        object.__setattr__(self, "affine_offset", offset)
        object.__setattr__(self, "affine_scale", scale)
        object.__setattr__(self, "physical_box", box)
        object.__setattr__(self, "grid_resolution", resolution)

    @property
    def num_samples(self) -> int:
        return int(self.state.shape[0])

    @property
    def distribution(self) -> str:
        return EQUILIBRIUM_ENDPOINT_ABI

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            name: int(np.count_nonzero(self.split == name))
            for name in _SPLIT_NAMES
        }

    @property
    def split_sha256(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in _SPLIT_NAMES:
            mask = self.split == name
            result[name] = _array_digest(
                self.state[mask],
                self.physical[mask],
                self.reduced_energy[mask],
            )
        return result

    @property
    def cell_area_state(self) -> float:
        physical_width = np.asarray(
            [axis[1] - axis[0] for axis in self.physical_box],
            dtype=np.float64,
        )
        state_width = physical_width / np.asarray(
            self.affine_scale,
            dtype=np.float64,
        )
        return float(
            np.prod(
                state_width / np.asarray(self.grid_resolution, dtype=np.float64)
            )
        )

    def split_state(self, name: str) -> np.ndarray:
        if name not in _SPLIT_NAMES:
            raise ValueError(f"unknown dataset split {name!r}")
        return self.state[self.split == name]

    def split_physical(self, name: str) -> np.ndarray:
        if name not in _SPLIT_NAMES:
            raise ValueError(f"unknown dataset split {name!r}")
        return self.physical[self.split == name]

    def metadata(self) -> dict[str, Any]:
        return {
            "distribution": EQUILIBRIUM_ENDPOINT_ABI,
            "generator_abi": self.generator_abi,
            "target_beta": self.target_beta,
            "kT": 1.0 / self.target_beta,
            "affine_offset": list(self.affine_offset),
            "affine_scale": list(self.affine_scale),
            "physical_box": [list(axis) for axis in self.physical_box],
            "grid_resolution": list(self.grid_resolution),
            "cell_area_state": self.cell_area_state,
            "rng_algorithm": "numpy.random.Generator(PCG64)",
            "seed": self.seed,
            "target_signature": self.target_signature,
            "split_counts": self.split_counts,
            "split_sha256": self.split_sha256,
            "num_samples": self.num_samples,
        }


def generate_mb2d_equilibrium_dataset(
    target: Any,
    *,
    num_train: int,
    num_validation: int,
    num_test: int,
    grid_resolution: int | tuple[int, int],
    seed: int,
    dtype: np.dtype | type = np.float32,
) -> MB2DEquilibriumDataset:
    """Sample a jittered midpoint quadrature of the finite-box Boltzmann law."""

    counts = {
        "train": int(num_train),
        "validation": int(num_validation),
        "test": int(num_test),
    }
    if counts["train"] <= 0 or any(value < 0 for value in counts.values()):
        raise ValueError("num_train must be positive and other split sizes non-negative")
    if isinstance(grid_resolution, int):
        resolution = (int(grid_resolution), int(grid_resolution))
    else:
        resolution = tuple(int(value) for value in grid_resolution)
    if len(resolution) != 2 or any(value < 2 for value in resolution):
        raise ValueError("grid_resolution must contain two integers >= 2")
    output_dtype = np.dtype(dtype)
    if output_dtype.kind != "f":
        raise TypeError("dtype must be floating point")

    target_spec = mb2d_target_spec(target)
    reference = target.normalized_grid(resolution, dtype=jnp.float32)
    probability = np.asarray(
        jax.device_get(reference.probability_mass),
        dtype=np.float64,
    ).reshape(-1)
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("target grid returned invalid probability masses")
    probability /= np.sum(probability)

    total = sum(counts.values())
    rng = np.random.default_rng(int(seed))
    flat_index = rng.choice(
        probability.size,
        size=total,
        replace=True,
        p=probability,
    )
    cell_index = np.column_stack(
        (flat_index // resolution[1], flat_index % resolution[1])
    )
    box = _box(target_spec["physical_box"])
    offset = _pair(target_spec["affine_offset"], name="affine_offset")
    scale = _pair(target_spec["affine_scale"], name="affine_scale")
    physical_lower = np.asarray([axis[0] for axis in box], dtype=np.float64)
    physical_upper = np.asarray([axis[1] for axis in box], dtype=np.float64)
    state_lower = (
        physical_lower - np.asarray(offset, dtype=np.float64)
    ) / np.asarray(scale, dtype=np.float64)
    state_upper = (
        physical_upper - np.asarray(offset, dtype=np.float64)
    ) / np.asarray(scale, dtype=np.float64)
    cell_width = (state_upper - state_lower) / np.asarray(
        resolution, dtype=np.float64
    )
    jitter = rng.random((total, 2))
    state64 = state_lower + (cell_index + jitter) * cell_width
    physical64 = (
        state64 * np.asarray(scale, dtype=np.float64)
        + np.asarray(offset, dtype=np.float64)
    )

    # Store the actual analytic energy at the jittered coordinates, not the
    # selected cell midpoint.  This field is diagnostic; training consumes
    # only state coordinates.
    reduced_energy_parts: list[np.ndarray] = []
    for start in range(0, total, 65536):
        state_chunk = jnp.asarray(state64[start : start + 65536], dtype=jnp.float32)
        energy_chunk = target.energy(
            state_chunk,
            include_training_wall=False,
        )
        reduced_energy_parts.append(
            np.asarray(
                jax.device_get(energy_chunk),
                dtype=np.float64,
            )
            * float(target_spec["beta"])
        )
    reduced_energy = np.concatenate(reduced_energy_parts, axis=0)
    split = np.concatenate(
        [
            np.full(count, name, dtype="<U10")
            for name, count in counts.items()
            if count > 0
        ],
        axis=0,
    )
    return MB2DEquilibriumDataset(
        state=np.asarray(state64, dtype=output_dtype),
        physical=np.asarray(physical64, dtype=output_dtype),
        reduced_energy=np.asarray(reduced_energy, dtype=output_dtype),
        split=split,
        target_beta=float(target_spec["beta"]),
        affine_offset=offset,
        affine_scale=scale,
        physical_box=box,
        grid_resolution=resolution,
        seed=int(seed),
        target_signature=_canonical_digest(target_spec),
    )


def save_mb2d_equilibrium_dataset(
    dataset: MB2DEquilibriumDataset,
    path: str | Path,
) -> str:
    """Write one self-describing NPZ and return its byte-level SHA-256."""

    output = Path(path)
    if output.suffix.lower() != ".npz":
        raise ValueError("equilibrium endpoint path must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        distribution=np.asarray(EQUILIBRIUM_ENDPOINT_ABI),
        metadata_json=np.asarray(
            json.dumps(
                dataset.metadata(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        state=dataset.state,
        physical=dataset.physical,
        reduced_energy=dataset.reduced_energy,
        split=dataset.split,
        target_beta=np.asarray(dataset.target_beta, dtype=np.float64),
        affine_offset=np.asarray(dataset.affine_offset, dtype=np.float64),
        affine_scale=np.asarray(dataset.affine_scale, dtype=np.float64),
        physical_box=np.asarray(dataset.physical_box, dtype=np.float64),
        grid_resolution=np.asarray(dataset.grid_resolution, dtype=np.int64),
        seed=np.asarray(dataset.seed, dtype=np.int64),
    )
    return _file_sha256(output)


def _expected_target_signature(expected_target: Any) -> str:
    if isinstance(expected_target, Mapping):
        if "target_signature" in expected_target:
            return str(expected_target["target_signature"])
        spec = {
            "implementation_abi": str(
                expected_target.get(
                    "implementation_abi",
                    "analytic_muller_brown_finite_box_v1",
                )
            ),
            "energy": str(
                expected_target.get("energy", "cg_bg_muller_brown_unbiased")
            ),
            "beta": float(expected_target["beta"]),
            "affine_offset": [
                float(value) for value in expected_target["affine_offset"]
            ],
            "affine_scale": [
                float(value) for value in expected_target["affine_scale"]
            ],
            "physical_box": [
                [float(bound) for bound in axis]
                for axis in expected_target["physical_box"]
            ],
        }
        return _canonical_digest(spec)
    return _canonical_digest(mb2d_target_spec(expected_target))


def load_mb2d_equilibrium_dataset(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_target: Any | None = None,
) -> MB2DEquilibriumDataset:
    """Load and validate a pinned equilibrium endpoint dataset."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_sha256 = _file_sha256(source)
    if actual_sha256 != str(expected_sha256).lower():
        raise ValueError(
            "equilibrium endpoint dataset SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "distribution",
            "metadata_json",
            "state",
            "physical",
            "reduced_energy",
            "split",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"equilibrium endpoint dataset is missing {missing}")
        distribution = str(np.asarray(archive["distribution"]).reshape(()))
        if distribution != EQUILIBRIUM_ENDPOINT_ABI:
            raise ValueError(
                f"dataset distribution must be {EQUILIBRIUM_ENDPOINT_ABI!r}"
            )
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).reshape(())))
        if not isinstance(metadata, dict):
            raise TypeError("dataset metadata_json must decode to a mapping")
        dataset = MB2DEquilibriumDataset(
            state=np.asarray(archive["state"]),
            physical=np.asarray(archive["physical"]),
            reduced_energy=np.asarray(archive["reduced_energy"]),
            split=np.asarray(archive["split"]),
            target_beta=float(metadata["target_beta"]),
            affine_offset=tuple(metadata["affine_offset"]),
            affine_scale=tuple(metadata["affine_scale"]),
            physical_box=tuple(tuple(axis) for axis in metadata["physical_box"]),
            grid_resolution=tuple(metadata["grid_resolution"]),
            seed=int(metadata["seed"]),
            target_signature=str(metadata["target_signature"]),
            generator_abi=str(metadata["generator_abi"]),
            sha256=actual_sha256,
            path=source.resolve(),
        )
    if int(metadata.get("num_samples", -1)) != dataset.num_samples:
        raise ValueError("dataset metadata num_samples does not match stored arrays")
    expected_counts = {
        str(name): int(value)
        for name, value in metadata.get("split_counts", {}).items()
    }
    if expected_counts != dataset.split_counts:
        raise ValueError("dataset metadata split_counts does not match stored arrays")
    expected_split_sha256 = {
        str(name): str(value)
        for name, value in metadata.get("split_sha256", {}).items()
    }
    if expected_split_sha256 != dataset.split_sha256:
        raise ValueError("dataset metadata split_sha256 does not match stored arrays")
    if expected_target is not None:
        expected_signature = _expected_target_signature(expected_target)
        if dataset.target_signature != expected_signature:
            raise ValueError(
                "equilibrium endpoint target identity does not match the experiment"
            )
    return replace(dataset, sha256=actual_sha256)


__all__ = [
    "EQUILIBRIUM_ENDPOINT_ABI",
    "EQUILIBRIUM_GENERATOR_ABI",
    "MB2DEquilibriumDataset",
    "generate_mb2d_equilibrium_dataset",
    "load_mb2d_equilibrium_dataset",
    "mb2d_target_spec",
    "save_mb2d_equilibrium_dataset",
]
