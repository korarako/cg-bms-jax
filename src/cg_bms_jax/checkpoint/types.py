"""Serialization-neutral checkpoint dataclasses.

Orbax or another storage layer may serialize these objects, but the scientific
identity checks live here and do not depend on a checkpoint backend.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AssetProvenance:
    """Immutable identity of the PMF, coordinate map and density convention."""

    pmf_revision: str
    pmf_sha256: str
    mapping_name: str
    mapping_indices: tuple[int, ...]
    temperature_kelvin: float
    thermal_energy_kj_mol: float
    coordinate_unit: str
    standardization_std: float
    num_particles: int
    spatial_dimension: int
    density_mode: str = "ambient_18d_aux_com"
    correction_mode: str = "exact_ambient"
    domain: str = "anchor_0_half_open_minimum_image"

    def __post_init__(self) -> None:
        if not self.pmf_revision or not self.pmf_sha256:
            raise ValueError("PMF revision and SHA256 must be recorded")
        if not self.mapping_name or not self.mapping_indices:
            raise ValueError("mapping identity and indices must be recorded")
        positive = (
            self.temperature_kelvin,
            self.thermal_energy_kj_mol,
            self.standardization_std,
            self.num_particles,
            self.spatial_dimension,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in positive):
            raise ValueError("thermodynamic, scale and shape values must be finite and positive")
        if not self.coordinate_unit or not self.density_mode or not self.correction_mode or not self.domain:
            raise ValueError("coordinate and density conventions must be explicit")

    @property
    def event_size(self) -> int:
        return int(self.num_particles * self.spatial_dimension)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe canonical representation."""

        result = asdict(self)
        result["mapping_indices"] = list(self.mapping_indices)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> AssetProvenance:
        values = dict(values)
        values["mapping_indices"] = tuple(values["mapping_indices"])
        return cls(**values)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata required to reject incompatible forward/backward checkpoints."""

    role: Literal["forward", "backward", "forward_pretrain"]
    global_step: int
    created_at_utc: str
    code_revision: str
    config_sha256: str
    model_signature: str
    sde_signature: str
    assets: AssetProvenance
    parent_forward_sha256: str | None = None
    formal_target_signature: str | None = None
    training_target_signature: str | None = None
    topology_signature: str | None = None
    coordinate_signature: str | None = None
    species_signature: str | None = None
    training_data_sha256: str | None = None
    warmstart_data_sha256: str | None = None
    initial_controller_path: str | None = None
    initial_controller_sha256: str | None = None
    initial_controller_role: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.role not in {"forward", "backward", "forward_pretrain"}:
            raise ValueError("role must be 'forward', 'backward', or 'forward_pretrain'")
        if self.global_step < 0 or self.schema_version <= 0:
            raise ValueError("global_step must be non-negative and schema_version positive")
        required = (
            self.created_at_utc,
            self.code_revision,
            self.config_sha256,
            self.model_signature,
            self.sde_signature,
        )
        if not all(required):
            raise ValueError("checkpoint provenance fields must not be empty")
        if self.role == "backward" and not self.parent_forward_sha256:
            raise ValueError("a backward checkpoint must identify its frozen forward checkpoint")
        initializer = (
            self.initial_controller_path,
            self.initial_controller_sha256,
            self.initial_controller_role,
        )
        if any(value is not None for value in initializer) and not all(initializer):
            raise ValueError(
                "initial controller path, SHA256, and role must be recorded together"
            )
        if self.initial_controller_role not in {None, "forward", "forward_pretrain"}:
            raise ValueError(
                "initial_controller_role must be 'forward' or 'forward_pretrain'"
            )
        digest_fields = {
            "training_data_sha256": self.training_data_sha256,
            "warmstart_data_sha256": self.warmstart_data_sha256,
            "initial_controller_sha256": self.initial_controller_sha256,
        }
        for name, digest in digest_fields.items():
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest.lower())
            ):
                raise ValueError(f"{name} must be a SHA256 hex digest when provided")
        if self.schema_version >= 2 and not all(
            (
                self.formal_target_signature,
                self.training_target_signature,
                self.topology_signature,
            )
        ):
            raise ValueError(
                "schema-v2 molecular checkpoints require formal/training target "
                "and topology signatures"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["assets"] = self.assets.to_dict()
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CheckpointMetadata:
        values = dict(values)
        values["assets"] = AssetProvenance.from_dict(values["assets"])
        return cls(**values)

    def require_compatible(self, other: CheckpointMetadata) -> None:
        """Raise when two controllers cannot participate in one PF-ODE."""

        comparisons = {
            "schema_version": (self.schema_version, other.schema_version),
            "config_sha256": (self.config_sha256, other.config_sha256),
            "model_signature": (self.model_signature, other.model_signature),
            "sde_signature": (self.sde_signature, other.sde_signature),
            "formal_target_signature": (
                self.formal_target_signature,
                other.formal_target_signature,
            ),
            "training_target_signature": (
                self.training_target_signature,
                other.training_target_signature,
            ),
            "topology_signature": (self.topology_signature, other.topology_signature),
            "asset_fingerprint": (self.assets.fingerprint(), other.assets.fingerprint()),
        }
        mismatches = [name for name, pair in comparisons.items() if pair[0] != pair[1]]
        optional_comparisons = {
            "coordinate_signature": (
                self.coordinate_signature,
                other.coordinate_signature,
            ),
            "species_signature": (self.species_signature, other.species_signature),
            "training_data_sha256": (
                self.training_data_sha256,
                other.training_data_sha256,
            ),
        }
        mismatches.extend(
            name
            for name, pair in optional_comparisons.items()
            if pair[0] is not None and pair[1] is not None and pair[0] != pair[1]
        )
        if mismatches:
            raise ValueError("incompatible checkpoint metadata: " + ", ".join(mismatches))


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Backend-neutral training payload passed to an Orbax storage adapter."""

    params: Any
    optimizer_state: Any
    metadata: CheckpointMetadata
