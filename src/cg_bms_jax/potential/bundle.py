"""Pinned, serialisable descriptions of CG-BG PMF assets."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_HF_REPO = "bojuntum/CGPeptides"
DEFAULT_HF_REVISION = "39765bbcfee382e5f30445589d7fe28ebb6cfff8"
DEFAULT_CGBG_REVISION = "948aaeff8a6b25de38b6e7b1112041c1cfd40573"
DEFAULT_CHEMTRAIN_REVISION = "a97ca2dd60c8327f574f269d02ec5edbccbae6b8"
K_B_KJ_MOL_K = 8.31446261815324e-3


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first(value: np.ndarray, ndim: int) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == ndim + 1:
        value = value[0]
    if value.ndim != ndim:
        raise ValueError(f"Expected {ndim} or {ndim + 1} dimensions, got {value.shape}")
    return value


def load_trusted_pickle(
    path: str | Path,
    *,
    expected_sha256: str | None,
    trusted: bool = False,
) -> Any:
    """Load a pinned pickle only after an explicit trust decision and hash check."""

    path = Path(path)
    if not trusted:
        raise PermissionError(
            "CG-BG checkpoints are Python pickles and can execute code. "
            "Verify their pinned source and SHA-256, then pass trusted=True."
        )
    if expected_sha256 is None:
        raise ValueError("A pinned SHA-256 is required before loading a pickle")
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise ValueError(f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - guarded by explicit trust + digest
    if isinstance(value, Mapping) and "params" in value:
        value = value["params"]
    return value


@dataclass(frozen=True)
class MBPMFBundle:
    checkpoint_path: str
    checkpoint_sha256: str | None
    data_path: str | None = None
    data_sha256: str | None = None
    kT: float = 1.0
    hf_repo_id: str = DEFAULT_HF_REPO
    hf_revision: str = DEFAULT_HF_REVISION
    model_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "hidden_dim": 128,
            "n_layers": 4,
            "num_rbf_centers": 100,
            "sigma": 5.0,
        }
    )

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CGBGAla2Bundle:
    data_path: str
    checkpoint_path: str
    box_nm: tuple[tuple[float, float, float], ...]
    species: tuple[int, ...]
    mask: tuple[bool, ...]
    reference_nm: tuple[tuple[float, float, float], ...]
    standardization_std_nm: float
    data_sha256: str | None = None
    checkpoint_sha256: str | None = None
    temperature_kelvin: float = 300.0
    hf_repo_id: str = DEFAULT_HF_REPO
    hf_revision: str = DEFAULT_HF_REVISION
    cgbg_revision: str = DEFAULT_CGBG_REVISION
    chemtrain_revision: str = DEFAULT_CHEMTRAIN_REVISION
    model_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 256,
            "r_cutoff": 0.5,
            "hidden_irreps": "32x0e+32x1o",
            "readout_irreps": "16x0e",
            "output_irreps": "1x0e",
            "max_ell": 3,
            "num_interactions": 2,
            "correlation": 3,
        }
    )
    schema_version: int = 1

    def __post_init__(self) -> None:
        box = np.asarray(self.box_nm, dtype=float)
        if box.shape != (3, 3):
            raise ValueError(f"box_nm must have shape (3,3), got {box.shape}")
        if not np.allclose(box, np.diag(np.diag(box))):
            raise ValueError("Only the orthorhombic CG-BG Ala2 box is supported")
        if np.any(np.diag(box) <= 0):
            raise ValueError("Box lengths must be positive")
        if len(self.species) != 6 or len(self.mask) != 6:
            raise ValueError("The core-beta Ala2 checkpoint requires exactly six beads")
        if np.asarray(self.reference_nm).shape != (6, 3):
            raise ValueError("reference_nm must have shape (6,3)")
        if self.standardization_std_nm <= 0:
            raise ValueError("standardization_std_nm must be positive")
        if self.temperature_kelvin <= 0:
            raise ValueError("temperature_kelvin must be positive")

    @property
    def kT_kj_mol(self) -> float:
        return K_B_KJ_MOL_K * self.temperature_kelvin

    @property
    def box_lengths_nm(self) -> np.ndarray:
        return np.diag(np.asarray(self.box_nm, dtype=float))

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def require_assets(self) -> None:
        for path_string, expected, label in (
            (self.data_path, self.data_sha256, "data"),
            (self.checkpoint_path, self.checkpoint_sha256, "checkpoint"),
        ):
            path = Path(path_string)
            if not path.is_file():
                raise FileNotFoundError(path)
            if expected is None:
                raise ValueError(f"No pinned SHA-256 for PMF {label}")
            actual = sha256_file(path)
            if actual != expected.lower():
                raise ValueError(f"PMF {label} SHA-256 mismatch: expected {expected}, got {actual}")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> CGBGAla2Bundle:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_data_file(
        cls,
        data_path: str | Path,
        checkpoint_path: str | Path,
        *,
        temperature_kelvin: float = 300.0,
        model_config: Mapping[str, Any] | None = None,
    ) -> CGBGAla2Bundle:
        data_path = Path(data_path).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()
        with np.load(data_path, allow_pickle=False) as archive:
            missing = {"R", "box", "species", "mask"}.difference(archive.files)
            if missing:
                raise KeyError(f"CG-BG data archive is missing {sorted(missing)}")
            coordinates = np.asarray(archive["R"], dtype=np.float64)
            box = _first(archive["box"], 2).astype(np.float64)
            species = _first(archive["species"], 1).astype(np.int64)
            mask = _first(archive["mask"], 1).astype(bool)
        if coordinates.ndim != 3 or coordinates.shape[1:] != (6, 3):
            raise ValueError(f"Expected Ala2 coordinates (B,6,3), got {coordinates.shape}")
        # Flow data are Cartesian nm in CG-BG.  Keep the test explicit instead
        # of silently accepting a fractional archive as Cartesian.
        reference = coordinates[0]
        centred = coordinates - coordinates.mean(axis=1, keepdims=True)
        std = float(centred.std())
        kwargs: dict[str, Any] = {}
        if model_config is not None:
            kwargs["model_config"] = dict(model_config)
        return cls(
            data_path=str(data_path),
            checkpoint_path=str(checkpoint_path),
            box_nm=tuple(tuple(float(v) for v in row) for row in box),
            species=tuple(int(v) for v in species),
            mask=tuple(bool(v) for v in mask),
            reference_nm=tuple(tuple(float(v) for v in row) for row in reference),
            standardization_std_nm=std,
            data_sha256=sha256_file(data_path),
            checkpoint_sha256=sha256_file(checkpoint_path),
            temperature_kelvin=temperature_kelvin,
            **kwargs,
        )

