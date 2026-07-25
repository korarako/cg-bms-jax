"""Versioned endpoint banks for flat-state bridge pretraining.

Two endpoint identities are supported without conflating them:

* a deterministic non-degenerate Gaussian mixture used for the historical
  full-support proposal-bias stress test;
* a file-backed, SHA-pinned exact-grid Boltzmann dataset used by the canonical
  equilibrium MB2D bridge experiment.

For the analytic Muller--Brown experiment, physical ``(x, y)`` coordinates
are mapped to the standardized controller state with the experiment's affine
transform. File-backed datasets additionally verify the target temperature,
domain, affine transform, split, and byte-level archive digest.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cg_bms_jax.data.mb2d_equilibrium import (
    EQUILIBRIUM_ENDPOINT_ABI,
    MB2DEquilibriumDataset,
    load_mb2d_equilibrium_dataset,
)

_DISTRIBUTION_ABI = "full_support_diagonal_gaussian_mixture_v1"


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy while rejecting non-reproducible values."""

    # OmegaConf containers expose Mapping/Sequence interfaces.  Recursing
    # explicitly avoids importing Hydra in the data layer.
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(subvalue) for key, subvalue in item.items()}
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            return [convert(subvalue) for subvalue in item]
        if isinstance(item, np.generic):
            return item.item()
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        raise TypeError(
            "bridge_data must contain only mappings, sequences and JSON scalars; "
            f"found {type(item).__name__}"
        )

    result = convert(value)
    if not isinstance(result, dict):
        raise TypeError("bridge_data must be a mapping")
    # ``allow_nan=False`` is the final finite-number guard for nested values.
    json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return result


def bridge_data_sha256(bridge_data: Mapping[str, Any]) -> str:
    """Return the canonical identity of a synthetic endpoint specification."""

    plain = _plain_mapping(bridge_data)
    payload = json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GaussianMixtureComponent:
    """One non-degenerate diagonal Gaussian in physical coordinates."""

    label: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weight: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Gaussian-mixture component labels must be non-empty")
        if not self.mean or len(self.mean) != len(self.scale):
            raise ValueError("component mean and scale must have the same positive size")
        if not all(math.isfinite(value) for value in self.mean):
            raise ValueError("component means must be finite")
        if not all(math.isfinite(value) and value > 0.0 for value in self.scale):
            raise ValueError(
                "component scales must be finite and strictly positive to retain full support"
            )
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("component weights must be finite and strictly positive")


@dataclass(frozen=True)
class FlatBridgeEndpointBank:
    """A reproducible endpoint bank and its complete data provenance."""

    endpoint_1: np.ndarray
    physical: np.ndarray
    component_index: np.ndarray
    component_labels: tuple[str, ...]
    component_counts: tuple[int, ...]
    bridge_data_sha256: str
    seed: int
    distribution: str = _DISTRIBUTION_ABI
    dataset_sha256: str | None = None
    split_name: str | None = None
    source_metadata: Mapping[str, Any] | None = None

    @property
    def num_endpoints(self) -> int:
        return int(self.endpoint_1.shape[0])

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.endpoint_1.shape[1:])


@dataclass(frozen=True)
class FullSupportGaussianMixture:
    """A mode-biased but everywhere-positive law in a flat physical space."""

    components: tuple[GaussianMixtureComponent, ...]
    affine_offset: tuple[float, ...]
    affine_scale: tuple[float, ...]
    seed: int
    num_endpoints: int
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("at least one Gaussian-mixture component is required")
        dimension = len(self.affine_offset)
        if dimension == 0 or len(self.affine_scale) != dimension:
            raise ValueError("affine offset and scale must have the same positive size")
        if not all(math.isfinite(value) for value in self.affine_offset):
            raise ValueError("affine offsets must be finite")
        if not all(
            math.isfinite(value) and value > 0.0 for value in self.affine_scale
        ):
            raise ValueError("affine scales must be finite and strictly positive")
        if self.num_endpoints <= 0:
            raise ValueError("num_endpoints must be positive")
        if any(len(component.mean) != dimension for component in self.components):
            raise ValueError("every component must match the affine dimension")
        labels = tuple(component.label for component in self.components)
        if len(set(labels)) != len(labels):
            raise ValueError("Gaussian-mixture component labels must be unique")
        weight_sum = math.fsum(component.weight for component in self.components)
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"Gaussian-mixture component weights must sum to one; got {weight_sum:.17g}"
            )
        if len(self.provenance_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.provenance_sha256.lower()
        ):
            raise ValueError("provenance_sha256 must be a SHA-256 hexadecimal digest")

    @property
    def dimension(self) -> int:
        return len(self.affine_offset)

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(
            [component.weight for component in self.components], dtype=np.float64
        )

    def to_state(self, physical: np.ndarray) -> np.ndarray:
        value = np.asarray(physical, dtype=np.float64)
        if value.shape[-1:] != (self.dimension,):
            raise ValueError(
                f"physical coordinates must end in dimension {self.dimension}"
            )
        return (value - np.asarray(self.affine_offset)) / np.asarray(
            self.affine_scale
        )

    def to_physical(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=np.float64)
        if value.shape[-1:] != (self.dimension,):
            raise ValueError(f"state must end in dimension {self.dimension}")
        return value * np.asarray(self.affine_scale) + np.asarray(
            self.affine_offset
        )

    def log_prob_physical(self, physical: np.ndarray) -> np.ndarray:
        """Evaluate the normalized mixture log density.

        A finite value at every finite coordinate follows from positive
        component weights and strictly positive diagonal scales.
        """

        value = np.asarray(physical, dtype=np.float64)
        if value.shape[-1:] != (self.dimension,):
            raise ValueError(
                f"physical coordinates must end in dimension {self.dimension}"
            )
        terms: list[np.ndarray] = []
        normalizer_constant = self.dimension * math.log(2.0 * math.pi)
        for component in self.components:
            mean = np.asarray(component.mean, dtype=np.float64)
            scale = np.asarray(component.scale, dtype=np.float64)
            standardized = (value - mean) / scale
            log_density = -0.5 * (
                np.sum(standardized * standardized, axis=-1)
                + normalizer_constant
                + 2.0 * np.sum(np.log(scale))
            )
            terms.append(math.log(component.weight) + log_density)
        return np.logaddexp.reduce(np.stack(terms, axis=0), axis=0)

    def generate(
        self,
        *,
        dtype: np.dtype | type = np.float32,
    ) -> FlatBridgeEndpointBank:
        output_dtype = np.dtype(dtype)
        if output_dtype.kind != "f":
            raise TypeError("dtype must be a floating-point dtype")
        rng = np.random.default_rng(self.seed)
        component_index = rng.choice(
            len(self.components),
            size=self.num_endpoints,
            replace=True,
            p=self.weights,
        ).astype(np.int64, copy=False)
        means = np.asarray(
            [component.mean for component in self.components], dtype=np.float64
        )
        scales = np.asarray(
            [component.scale for component in self.components], dtype=np.float64
        )
        physical = (
            means[component_index]
            + scales[component_index]
            * rng.standard_normal((self.num_endpoints, self.dimension))
        )
        endpoint_1 = np.asarray(self.to_state(physical), dtype=output_dtype)
        physical = np.asarray(physical, dtype=output_dtype)
        counts = tuple(
            int(value)
            for value in np.bincount(
                component_index, minlength=len(self.components)
            )
        )
        for value in (endpoint_1, physical, component_index):
            value.setflags(write=False)
        return FlatBridgeEndpointBank(
            endpoint_1=endpoint_1,
            physical=physical,
            component_index=component_index,
            component_labels=tuple(
                component.label for component in self.components
            ),
            component_counts=counts,
            bridge_data_sha256=self.provenance_sha256,
            seed=self.seed,
            distribution=_DISTRIBUTION_ABI,
            source_metadata={
                "full_support": True,
                "coordinate_space": "physical",
            },
        )


@dataclass(frozen=True)
class EquilibriumEndpointSource:
    """Hash-pinned finite-support endpoint data generated from the exact grid."""

    dataset: MB2DEquilibriumDataset
    split_name: str
    num_endpoints: int
    affine_offset: tuple[float, ...]
    affine_scale: tuple[float, ...]
    provenance_sha256: str

    def __post_init__(self) -> None:
        if self.dataset.sha256 is None:
            raise ValueError("equilibrium endpoint source requires a loaded dataset")
        if self.split_name not in {"train", "validation", "test"}:
            raise ValueError("equilibrium endpoint split is invalid")
        available = self.dataset.split_state(self.split_name).shape[0]
        if self.num_endpoints <= 0 or self.num_endpoints > available:
            raise ValueError(
                "bridge_data.num_endpoints must be positive and no larger than "
                f"the selected split ({available})"
            )
        if tuple(self.affine_offset) != tuple(self.dataset.affine_offset):
            raise ValueError("dataset affine_offset does not match the experiment")
        if tuple(self.affine_scale) != tuple(self.dataset.affine_scale):
            raise ValueError("dataset affine_scale does not match the experiment")

    @property
    def dimension(self) -> int:
        return len(self.affine_offset)

    @property
    def dataset_sha256(self) -> str:
        assert self.dataset.sha256 is not None
        return self.dataset.sha256

    def generate(
        self,
        *,
        dtype: np.dtype | type = np.float32,
    ) -> FlatBridgeEndpointBank:
        output_dtype = np.dtype(dtype)
        if output_dtype.kind != "f":
            raise TypeError("dtype must be a floating-point dtype")
        endpoint_1 = np.asarray(
            self.dataset.split_state(self.split_name)[: self.num_endpoints],
            dtype=output_dtype,
        )
        physical = np.asarray(
            self.dataset.split_physical(self.split_name)[: self.num_endpoints],
            dtype=output_dtype,
        )
        component_index = np.full((self.num_endpoints,), -1, dtype=np.int64)
        for value in (endpoint_1, physical, component_index):
            value.setflags(write=False)
        return FlatBridgeEndpointBank(
            endpoint_1=endpoint_1,
            physical=physical,
            component_index=component_index,
            component_labels=(),
            component_counts=(),
            bridge_data_sha256=self.provenance_sha256,
            seed=self.dataset.seed,
            distribution=EQUILIBRIUM_ENDPOINT_ABI,
            dataset_sha256=self.dataset_sha256,
            split_name=self.split_name,
            source_metadata={
                "full_support": False,
                "formal_support": "finite_cartesian_box",
                "coordinate_space": "state",
                "target_beta": self.dataset.target_beta,
                "target_signature": self.dataset.target_signature,
                "grid_resolution": list(self.dataset.grid_resolution),
                "dataset_split_counts": self.dataset.split_counts,
            },
        )


def full_support_mixture_from_config(
    bridge_data: Mapping[str, Any],
    *,
    affine_offset: Sequence[float],
    affine_scale: Sequence[float],
) -> FullSupportGaussianMixture:
    """Validate ``experiment.bridge_data`` and construct its sampling law."""

    plain = _plain_mapping(bridge_data)
    distribution = str(plain.get("distribution", ""))
    if distribution != _DISTRIBUTION_ABI:
        raise ValueError(
            f"bridge_data.distribution must be {_DISTRIBUTION_ABI!r}"
        )
    schema_version = int(plain.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError("bridge_data.schema_version must equal 1")
    coordinate_space = str(plain.get("coordinate_space", ""))
    if coordinate_space != "physical":
        raise ValueError("bridge_data.coordinate_space must be 'physical'")
    components_value = plain.get("components")
    if not isinstance(components_value, list) or not components_value:
        raise ValueError("bridge_data.components must be a non-empty list")
    components: list[GaussianMixtureComponent] = []
    for value in components_value:
        if not isinstance(value, dict):
            raise TypeError("every bridge_data component must be a mapping")
        components.append(
            GaussianMixtureComponent(
                label=str(value["label"]),
                mean=tuple(float(item) for item in value["mean"]),
                scale=tuple(float(item) for item in value["scale"]),
                weight=float(value["weight"]),
            )
        )
    return FullSupportGaussianMixture(
        components=tuple(components),
        affine_offset=tuple(float(value) for value in affine_offset),
        affine_scale=tuple(float(value) for value in affine_scale),
        seed=int(plain["seed"]),
        num_endpoints=int(plain["num_endpoints"]),
        provenance_sha256=bridge_data_sha256(plain),
    )


def bridge_endpoint_source_from_config(
    bridge_data: Mapping[str, Any],
    *,
    affine_offset: Sequence[float],
    affine_scale: Sequence[float],
    physical_box: Sequence[Sequence[float]] | None = None,
    base_dir: str | Path | None = None,
    expected_target: Any | None = None,
) -> FullSupportGaussianMixture | EquilibriumEndpointSource:
    """Construct either the legacy biased source or a pinned equilibrium source."""

    plain = _plain_mapping(bridge_data)
    distribution = str(plain.get("distribution", ""))
    if distribution == _DISTRIBUTION_ABI:
        return full_support_mixture_from_config(
            plain,
            affine_offset=affine_offset,
            affine_scale=affine_scale,
        )
    if distribution != EQUILIBRIUM_ENDPOINT_ABI:
        raise ValueError(
            "bridge_data.distribution must be one of "
            f"{_DISTRIBUTION_ABI!r}, {EQUILIBRIUM_ENDPOINT_ABI!r}"
        )
    if int(plain.get("schema_version", 0)) != 1:
        raise ValueError("equilibrium bridge_data.schema_version must equal 1")
    if str(plain.get("coordinate_space", "")) != "state":
        raise ValueError(
            "equilibrium bridge_data.coordinate_space must be 'state'"
        )
    path_value = str(plain.get("path", "")).strip()
    if not path_value:
        raise ValueError("equilibrium bridge_data.path must be non-empty")
    raw_path = Path(path_value).expanduser()
    if not raw_path.is_absolute():
        root = Path.cwd() if base_dir is None else Path(base_dir)
        raw_path = root / raw_path
    path = raw_path.resolve()
    expected_sha256 = str(plain.get("sha256", "")).lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError(
            "equilibrium bridge_data.sha256 must be a lowercase SHA-256 digest"
        )
    split_name = str(plain.get("split", ""))
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(
            "equilibrium bridge_data.split must be train, validation or test"
        )
    configured_beta = float(plain.get("target_beta", math.nan))
    if not math.isfinite(configured_beta) or configured_beta <= 0.0:
        raise ValueError(
            "equilibrium bridge_data.target_beta must be finite and positive"
        )
    if expected_target is None and physical_box is not None:
        expected_target = {
            "implementation_abi": "analytic_muller_brown_finite_box_v1",
            "energy": "cg_bg_muller_brown_unbiased",
            "beta": configured_beta,
            "affine_offset": [float(value) for value in affine_offset],
            "affine_scale": [float(value) for value in affine_scale],
            "physical_box": [
                [float(bound) for bound in axis] for axis in physical_box
            ],
        }
    dataset = load_mb2d_equilibrium_dataset(
        path,
        expected_sha256=expected_sha256,
        expected_target=expected_target,
    )
    if not math.isclose(
        dataset.target_beta,
        configured_beta,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "equilibrium bridge_data.target_beta does not match the dataset"
        )
    return EquilibriumEndpointSource(
        dataset=dataset,
        split_name=split_name,
        num_endpoints=int(plain["num_endpoints"]),
        affine_offset=tuple(float(value) for value in affine_offset),
        affine_scale=tuple(float(value) for value in affine_scale),
        provenance_sha256=bridge_data_sha256(plain),
    )


def bridge_endpoint_bank_from_config(
    bridge_data: Mapping[str, Any],
    *,
    affine_offset: Sequence[float],
    affine_scale: Sequence[float],
    physical_box: Sequence[Sequence[float]] | None = None,
    base_dir: str | Path | None = None,
    expected_target: Any | None = None,
    dtype: np.dtype | type = np.float32,
) -> FlatBridgeEndpointBank:
    """Load/generate the configured flat bridge endpoint bank."""

    source = bridge_endpoint_source_from_config(
        bridge_data,
        affine_offset=affine_offset,
        affine_scale=affine_scale,
        physical_box=physical_box,
        base_dir=base_dir,
        expected_target=expected_target,
    )
    return source.generate(dtype=dtype)


__all__ = [
    "EquilibriumEndpointSource",
    "FlatBridgeEndpointBank",
    "FullSupportGaussianMixture",
    "GaussianMixtureComponent",
    "bridge_endpoint_bank_from_config",
    "bridge_endpoint_source_from_config",
    "bridge_data_sha256",
    "full_support_mixture_from_config",
]
