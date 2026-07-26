#!/usr/bin/env python3
"""Build the lightweight, result-only cg-bms-jax release freeze.

The builder deliberately does *not* copy checkpoints or PF sample archives.
It verifies and hashes those large inputs, records their scientific metadata,
re-evaluates every frozen formal no-clip result with the repository's current
evaluators, and publishes only small metrics, figures, configs and provenance.

The release scope is intentionally limited to:

* MB CG1D: three-seed cold Energy-BMS + PF reweighting;
* analytic MB2D: three-seed cold Energy-BMS, exact-equilibrium bridge,
  exact-equilibrium bridge-to-Energy, and the two legacy deliberately biased
  synthetic-endpoint bridge stress tests;
* six-bead CG Ala2: the strict positive release-tolerance PF10k result and the
  in-root W10k-to-E100k-to-B100k PF20k mixed/low-ESS result.

All-atom Ala2 is explicitly outside this release and is rejected if it appears
in checkpoint or PF metadata.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
FORMAL_ENERGY_LOWER_QUANTILE = 0.005
FORMAL_ENERGY_UPPER_QUANTILE = 0.995
FORMAL_ENERGY_VIEW_PADDING = 0.05
RELEASE_PALETTE = {
    "exact_or_true": "#000000",
    "reference": "#8FD18B",
    "proposal": "#F2A174",
    "reweighted": "#4472C4",
    "reweighted_fill": "#B9CBEA",
}
RELEASE_FILL_STYLE = {
    "exact_or_true": {
        "rendering": "line_only",
        "edgecolor": "#000000",
        "facecolor": None,
        "alpha": 1.0,
    },
    "reference": {
        "rendering": "filled_with_outline",
        "edgecolor": "#8FD18B",
        "facecolor": "#8FD18B",
        "alpha": 0.52,
    },
    "proposal": {
        "rendering": "filled_with_outline",
        "edgecolor": "#F2A174",
        "facecolor": "#F2A174",
        "alpha": 0.56,
    },
    "reweighted": {
        "rendering": "filled_with_outline",
        "edgecolor": "#4472C4",
        "facecolor": "#B9CBEA",
        "alpha": 0.58,
    },
}
MB2D_FES_COLORMAP = "viridis"
MB2D_FES_VMIN = 0.0
MB2D_FES_VMAX_KT = 12.0
MB2D_PHYSICAL_LIMITS = [[0.0, 50.0], [0.0, 50.0]]
DEFAULT_PROJECT_ROOT = Path("/ds/project/weilong/ke/cg-bms-jax")
DEFAULT_RELEASE_NAME = "release_v0.1.0"
MB2D_EQUILIBRIUM_ENDPOINT_SHA256 = (
    "f4c43d8feff619102e5cc1f8ad2dbd422d9a201453d356d4f59e48cb9032347d"
)
ALL_ATOM_MARKERS = (
    "ala2_aa_ambient66",
    "ambient_66d",
    "all_atom",
    "allatom",
)


@dataclass(frozen=True)
class ArmInput:
    """One formal proposal/reweighting result and its matched checkpoints."""

    system: str
    arm: str
    release_role: str
    data_identity: str
    seed: int
    expected_samples: int
    forward_checkpoint: Path
    backward_checkpoint: Path
    pf_archive: Path
    config_path: Path
    forward_role: str
    forward_step: int
    backward_step: int
    expected_event_shape: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.system}/{self.arm}/seed_{self.seed}"


@dataclass(frozen=True)
class CheckpointProvenance:
    path: str
    sha256: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PFProvenance:
    path: str
    sha256: str
    size_bytes: int
    sample_count: int
    arrays: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    sampler_kind: str
    density_mode: str
    sampling_audit: dict[str, Any]
    endpoint_metadata_audit: dict[str, Any]
    no_clip_audit: dict[str, Any]


def _release_family(system: str) -> str:
    """Map internal result names onto the public release-family contract."""

    return "mb2d_analytic" if system == "mb2d" else system


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _hash_checkpoint_file(hasher: Any, path: Path, relative: str) -> None:
    encoded = relative.replace(os.sep, "/").encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)


def _checkpoint_sha256(path: Path) -> str:
    """Reproduce cg_bms_jax.checkpoint.checkpoint_sha256 without JAX import."""

    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.relative_to(path).as_posix() != "sha256.txt"
    )
    hasher = hashlib.sha256()
    for file_path in files:
        _hash_checkpoint_file(
            hasher,
            file_path,
            file_path.relative_to(path).as_posix(),
        )
    return hasher.hexdigest()


def _scalar_string(value: Any, *, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{field} must be a scalar string")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _reject_all_atom_payload(payload: Any, *, source: str) -> None:
    serialized = json.dumps(payload, sort_keys=True, default=str).lower()
    matched = [marker for marker in ALL_ATOM_MARKERS if marker in serialized]
    if matched:
        raise ValueError(
            f"All-atom Ala2 is excluded from this release; {source} contains "
            f"{matched}"
        )


def _checkpoint_provenance(path: Path) -> CheckpointProvenance:
    path = _absolute(path)
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {path}")
    digest_path = path / "sha256.txt"
    manifest_path = path / "manifest.json"
    if not digest_path.is_file():
        raise FileNotFoundError(f"checkpoint digest is missing: {digest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {manifest_path}")
    expected = digest_path.read_text(encoding="ascii").strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"malformed checkpoint digest: {digest_path}")
    actual = _checkpoint_sha256(path)
    if not secrets.compare_digest(expected, actual):
        raise ValueError(
            f"checkpoint checksum mismatch for {path}: "
            f"expected {expected}, computed {actual}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"checkpoint manifest must be an object: {manifest_path}")
    _reject_all_atom_payload(manifest, source=str(manifest_path))
    return CheckpointProvenance(
        path=str(path),
        sha256=actual,
        manifest=manifest,
    )


def _checkpoint_metadata(
    checkpoint: CheckpointProvenance,
) -> dict[str, Any]:
    if checkpoint.manifest.get("format") != "cg-bms-jax-orbax":
        raise ValueError(
            f"unsupported checkpoint format in {checkpoint.path}: "
            f"{checkpoint.manifest.get('format')!r}"
        )
    if checkpoint.manifest.get("format_version") != 1:
        raise ValueError(
            f"unsupported checkpoint format version in {checkpoint.path}"
        )
    metadata = checkpoint.manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(
            f"checkpoint manifest metadata is missing: {checkpoint.path}"
        )
    return metadata


def _expected_runtime_identity(
    config_path: Path,
) -> tuple[dict[str, Any], str]:
    """Build the identity from the committed config and pinned asset manifest."""

    from omegaconf import OmegaConf

    from cg_bms_jax.checkpoint import canonical_config_sha256
    from cg_bms_jax.runtime import build_runtime_system

    loaded = OmegaConf.load(config_path)
    experiment = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(experiment, dict):
        raise TypeError(f"experiment config must decode to a mapping: {config_path}")
    system = build_runtime_system(experiment, load_potential=False)
    return dict(system.identity), canonical_config_sha256(experiment)


def _expected_pf_seed(arm: ArmInput) -> int:
    return arm.seed if arm.system == "ala2_cg" else arm.seed + 200


def _expected_likelihood(arm: ArmInput) -> dict[str, Any]:
    if arm.arm == "mixed_low_ess_warm10k_energy100k_pf20k":
        return {
            "solver": "dopri5",
            "dt0": 5.0e-3,
            "rtol": 1.0e-5,
            "atol": 1.0e-5,
            "max_steps": 16384,
            "divergence": "exact",
        }
    return {
        "solver": "dopri5",
        "dt0": 1.0e-3,
        "rtol": 1.0e-5,
        "atol": 1.0e-6,
        "max_steps": 16384,
        "divergence": "exact",
    }


def _equal_float(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )


def _validate_arm_identity(
    arm: ArmInput,
    *,
    forward: CheckpointProvenance,
    backward: CheckpointProvenance,
    pf: PFProvenance,
    expected_identity: Mapping[str, Any],
    expected_config_sha256: str,
) -> None:
    """Lock a path label to its actual controller, target and data identity."""

    forward_metadata = _checkpoint_metadata(forward)
    backward_metadata = _checkpoint_metadata(backward)
    mismatches: list[str] = []

    def require_equal(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            mismatches.append(
                f"{label}: expected {expected!r}, found {actual!r}"
            )

    require_equal("forward.role", forward_metadata.get("role"), arm.forward_role)
    require_equal("backward.role", backward_metadata.get("role"), "backward")
    require_equal(
        "forward.global_step",
        forward_metadata.get("global_step"),
        arm.forward_step,
    )
    require_equal(
        "backward.global_step",
        backward_metadata.get("global_step"),
        arm.backward_step,
    )
    require_equal(
        "backward.parent_forward_sha256",
        backward_metadata.get("parent_forward_sha256"),
        forward.sha256,
    )
    require_equal(
        "forward.config_sha256",
        forward_metadata.get("config_sha256"),
        expected_config_sha256,
    )
    require_equal(
        "backward.config_sha256",
        backward_metadata.get("config_sha256"),
        expected_config_sha256,
    )

    compatible_fields = (
        "schema_version",
        "model_signature",
        "sde_signature",
        "formal_target_signature",
        "training_target_signature",
        "topology_signature",
        "coordinate_signature",
        "species_signature",
        "training_data_sha256",
        "warmstart_data_sha256",
    )
    for field in compatible_fields:
        require_equal(
            f"forward/backward.{field}",
            backward_metadata.get(field),
            forward_metadata.get(field),
        )
    require_equal(
        "forward/backward.assets",
        backward_metadata.get("assets"),
        forward_metadata.get("assets"),
    )

    checkpoint_identity_fields = (
        "model_signature",
        "sde_signature",
        "formal_target_signature",
        "training_target_signature",
        "topology_signature",
        "coordinate_signature",
        "species_signature",
        "training_data_sha256",
    )
    for field in checkpoint_identity_fields:
        expected = expected_identity.get(field)
        if expected is not None:
            require_equal(
                f"forward.{field}",
                forward_metadata.get(field),
                expected,
            )
    assets = forward_metadata.get("assets")
    if not isinstance(assets, Mapping):
        mismatches.append("forward.assets: missing mapping")
    else:
        require_equal(
            "forward.assets.pmf_sha256",
            assets.get("pmf_sha256"),
            expected_identity.get("pmf_sha256"),
        )
        require_equal(
            "forward.assets.pmf_revision",
            assets.get("pmf_revision"),
            expected_identity.get("pmf_revision"),
        )
        require_equal(
            "forward.assets.num_particles",
            assets.get("num_particles"),
            int(np.prod(arm.expected_event_shape[:-1]))
            if len(arm.expected_event_shape) == 2
            else 1,
        )

    pf_identity_fields = (
        "experiment_family",
        "coordinate_mode",
        "model_signature",
        "sde_signature",
        "pmf_revision",
        "pmf_sha256",
        "training_data_sha256",
        "coordinate_signature",
        "species_signature",
        "density_mode",
        "target_mode",
        "target_implementation_abi",
        "formal_target_signature",
        "training_target_signature",
        "topology_signature",
    )
    for field in pf_identity_fields:
        expected = expected_identity.get(field)
        if expected is not None:
            require_equal(
                f"pf.metadata.{field}",
                pf.metadata.get(field),
                expected,
            )
    endpoint_fields: tuple[str, ...]
    if arm.data_identity == "endpoint_independent_energy_bms":
        endpoint_fields = ()
    elif arm.data_identity == "deliberately_biased_synthetic_full_support_v1":
        endpoint_fields = (
            "endpoint_distribution",
            "configured_endpoint_spec_sha256",
            "synthetic_endpoint_spec_sha256",
        )
    elif arm.data_identity == "equilibrium_exact_v1":
        endpoint_fields = (
            "endpoint_distribution",
            "configured_endpoint_spec_sha256",
            "equilibrium_endpoint_spec_sha256",
        )
    else:
        endpoint_fields = ()
    missing_endpoint_fields: list[str] = []
    for field in endpoint_fields:
        if field in pf.metadata:
            require_equal(
                f"pf.metadata.{field}",
                pf.metadata[field],
                expected_identity.get(field),
            )
        else:
            missing_endpoint_fields.append(field)
    audited_missing_endpoint_fields = [
        field
        for field in pf.endpoint_metadata_audit.get("missing_fields", [])
        if field in endpoint_fields
    ]
    require_equal(
        "pf.endpoint_metadata_audit.missing_fields",
        audited_missing_endpoint_fields,
        missing_endpoint_fields,
    )
    require_equal(
        "pf.forward_controller_kind",
        pf.metadata.get("forward_controller_kind"),
        arm.forward_role,
    )
    require_equal("pf.seed", pf.metadata.get("seed"), _expected_pf_seed(arm))
    require_equal(
        "pf.forward_metadata",
        pf.metadata.get("forward_metadata"),
        forward_metadata,
    )
    require_equal(
        "pf.backward_metadata",
        pf.metadata.get("backward_metadata"),
        backward_metadata,
    )
    sampling = pf.metadata.get("sampling")
    if sampling is None:
        require_equal(
            "pf.sampling_audit.schema_status",
            pf.sampling_audit.get("schema_status"),
            "legacy_mapping_absent",
        )
        require_equal(
            "pf.sampling_audit.archive_sample_count",
            pf.sampling_audit.get("archive_sample_count"),
            arm.expected_samples,
        )
        require_equal(
            "pf.sampling_audit.expected_sample_count",
            pf.sampling_audit.get("expected_sample_count"),
            arm.expected_samples,
        )
    elif not isinstance(sampling, Mapping):
        mismatches.append(
            "pf.sampling: expected a mapping or an absent legacy field, "
            f"found {type(sampling).__name__}"
        )
    else:
        require_equal(
            "pf.sampling.global_num_samples",
            sampling.get("global_num_samples"),
            arm.expected_samples,
        )
        require_equal(
            "pf.sampling.local_num_samples",
            sampling.get("local_num_samples"),
            arm.expected_samples,
        )
    likelihood = pf.metadata.get("likelihood")
    if not isinstance(likelihood, Mapping):
        mismatches.append("pf.likelihood: missing mapping")
    else:
        for field, expected in _expected_likelihood(arm).items():
            actual = likelihood.get(field)
            if isinstance(expected, float):
                if not _equal_float(actual, expected):
                    mismatches.append(
                        f"pf.likelihood.{field}: expected {expected!r}, "
                        f"found {actual!r}"
                    )
            else:
                require_equal(f"pf.likelihood.{field}", actual, expected)

    configured_data = expected_identity.get("training_data_sha256")
    initializer_fields = (
        forward_metadata.get("initial_controller_path"),
        forward_metadata.get("initial_controller_sha256"),
        forward_metadata.get("initial_controller_role"),
    )
    cold = arm.arm in {"energy_only", "cold_energy_only"}
    direct_bridge = arm.forward_role == "forward_pretrain"
    if cold:
        require_equal(
            "forward.warmstart_data_sha256",
            forward_metadata.get("warmstart_data_sha256"),
            None,
        )
        require_equal("forward.initializer", initializer_fields, (None, None, None))
    elif direct_bridge:
        require_equal(
            "forward.warmstart_data_sha256",
            forward_metadata.get("warmstart_data_sha256"),
            configured_data,
        )
        require_equal("forward.initializer", initializer_fields, (None, None, None))
    else:
        require_equal(
            "forward.warmstart_data_sha256",
            forward_metadata.get("warmstart_data_sha256"),
            configured_data,
        )
        if any(value is None for value in initializer_fields):
            mismatches.append(
                "forward initializer path/SHA/role must all be present"
            )
        else:
            require_equal(
                "forward.initial_controller_role",
                initializer_fields[2],
                "forward_pretrain",
            )
            initializer = _checkpoint_provenance(Path(str(initializer_fields[0])))
            initializer_metadata = _checkpoint_metadata(initializer)
            require_equal(
                "forward.initial_controller_sha256",
                initializer_fields[1],
                initializer.sha256,
            )
            require_equal(
                "initializer.role",
                initializer_metadata.get("role"),
                "forward_pretrain",
            )
            require_equal(
                "initializer.warmstart_data_sha256",
                initializer_metadata.get("warmstart_data_sha256"),
                configured_data,
            )

    if mismatches:
        raise ValueError(
            f"release identity lock failed for {arm.key}:\n  "
            + "\n  ".join(mismatches)
        )


def _top_mass(weights: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(weights.size * fraction)))
    return float(np.sort(weights)[-count:].sum())


def _validate_formal_no_clip(
    *,
    weights: np.ndarray,
    stored_logw_raw: np.ndarray,
    target_reduced_energy: np.ndarray,
    logq_ambient: np.ndarray,
    valid: np.ndarray,
    support: np.ndarray,
) -> dict[str, Any]:
    """Recompute exact-ambient weights rather than trusting archive labels."""

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    stored_logw_raw = np.asarray(stored_logw_raw, dtype=np.float64).reshape(-1)
    target_reduced_energy = np.asarray(
        target_reduced_energy,
        dtype=np.float64,
    ).reshape(-1)
    logq_ambient = np.asarray(logq_ambient, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    support = np.asarray(support, dtype=bool).reshape(-1)
    if not (
        weights.shape
        == stored_logw_raw.shape
        == target_reduced_energy.shape
        == logq_ambient.shape
        == valid.shape
        == support.shape
    ):
        raise ValueError("formal weight inputs have inconsistent shapes")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("formal PF weights must be finite and non-negative")
    if not np.isclose(weights.sum(), 1.0, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError(f"formal PF weights do not sum to one: {weights.sum()}")
    recomputed_logw_raw = -target_reduced_energy - logq_ambient
    included = (
        valid
        & support
        & np.isfinite(target_reduced_energy)
        & np.isfinite(logq_ambient)
        & np.isfinite(recomputed_logw_raw)
    )
    if not included.any():
        raise ValueError("formal PF archive contains no finite valid log weights")
    raw_difference = np.abs(
        stored_logw_raw[included] - recomputed_logw_raw[included]
    )
    maximum_raw_difference = float(np.max(raw_difference))
    if not np.allclose(
        stored_logw_raw[included],
        recomputed_logw_raw[included],
        rtol=5.0e-6,
        atol=1.0e-4,
    ):
        raise ValueError(
            "stored logw_raw does not equal "
            "-target_reduced_energy-logq_ambient on formal samples; "
            f"maximum absolute difference is {maximum_raw_difference:.6g}"
        )
    if np.any(weights[~included] != 0.0):
        raise ValueError("invalid/out-of-support samples must have exactly zero weight")
    expected = np.zeros_like(weights)
    shifted = recomputed_logw_raw[included] - np.max(
        recomputed_logw_raw[included]
    )
    expected[included] = np.exp(shifted)
    expected /= expected.sum()
    difference = np.abs(weights - expected)
    maximum_difference = float(np.max(difference))
    if not np.allclose(weights, expected, rtol=2.0e-5, atol=2.0e-8):
        raise ValueError(
            "PF archive is not a formal no-clip softmax(logw_raw) estimator; "
            f"maximum absolute weight difference is {maximum_difference:.6g}"
        )
    ess = float(1.0 / np.sum(weights**2))
    finite_logw = recomputed_logw_raw[included]
    return {
        "formal_no_clip": True,
        "raw_formula": "-target_reduced_energy-logq_ambient",
        "normalization": (
            "softmax(recomputed raw formula) over finite valid/support samples"
        ),
        "maximum_logw_raw_formula_difference": maximum_raw_difference,
        "maximum_weight_difference_from_raw_softmax": maximum_difference,
        "valid_mask_fraction": float(np.mean(valid)),
        "support_mask_fraction": float(np.mean(support)),
        "included_sample_fraction": float(np.mean(included)),
        "ess": ess,
        "ess_fraction": ess / float(weights.size),
        "max_weight": float(np.max(weights)),
        "top_0p1_percent_mass": _top_mass(weights, 0.001),
        "top_1_percent_mass": _top_mass(weights, 0.01),
        "logw_variance": float(np.var(finite_logw)),
        "logw_span": float(np.max(finite_logw) - np.min(finite_logw)),
    }


def _pf_provenance(
    path: Path,
    *,
    expected_samples: int,
    forward_sha256: str,
    backward_sha256: str,
    expected_event_shape: tuple[int, ...],
) -> PFProvenance:
    path = _absolute(path)
    if not path.is_file():
        raise FileNotFoundError(f"PF archive is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "R",
            "U",
            "logp",
            "logw",
            "logw_raw",
            "weights",
            "logq_ambient",
            "target_reduced_energy",
            "valid_mask",
            "support_mask",
            "sampler_kind",
            "density_mode",
            "metadata_json",
        }
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"PF archive {path} is missing {sorted(missing)}")
        arrays = {
            name: {
                "shape": list(archive[name].shape),
                "dtype": str(archive[name].dtype),
            }
            for name in sorted(archive.files)
        }
        sampler_kind = _scalar_string(
            archive["sampler_kind"],
            field="sampler_kind",
        )
        density_mode = _scalar_string(
            archive["density_mode"],
            field="density_mode",
        )
        if sampler_kind != "pf_ode":
            raise ValueError(f"{path} is not a PF-ODE archive")
        if density_mode != "ambient_exact":
            raise ValueError(
                f"{path} must use density_mode='ambient_exact', "
                f"found {density_mode!r}"
            )
        metadata = json.loads(
            _scalar_string(archive["metadata_json"], field="metadata_json")
        )
        if not isinstance(metadata, dict):
            raise TypeError(f"metadata_json in {path} must decode to an object")
        _reject_all_atom_payload(metadata, source=str(path))
        sample_count = int(np.asarray(archive["U"]).shape[0])
        if sample_count != expected_samples:
            raise ValueError(
                f"PF sample count mismatch for {path}: "
                f"expected {expected_samples}, found {sample_count}"
            )
        coordinates = np.asarray(archive["R"])
        if coordinates.shape != (sample_count, *expected_event_shape):
            raise ValueError(
                f"R shape mismatch in {path}: expected "
                f"{(sample_count, *expected_event_shape)}, "
                f"found {coordinates.shape}"
            )
        valid = np.asarray(archive["valid_mask"], dtype=bool).reshape(-1)
        support = np.asarray(archive["support_mask"], dtype=bool).reshape(-1)
        if valid.shape != (sample_count,) or support.shape != (sample_count,):
            raise ValueError(f"valid/support mask shape is invalid in {path}")
        logp = np.asarray(archive["logp"], dtype=np.float64).reshape(-1)
        logq = np.asarray(
            archive["logq_ambient"],
            dtype=np.float64,
        ).reshape(-1)
        finite_logp = np.isfinite(logp)
        finite_logq = np.isfinite(logq)
        if not np.array_equal(finite_logp, finite_logq) or not np.allclose(
            logp[finite_logp],
            logq[finite_logq],
            rtol=5.0e-6,
            atol=1.0e-4,
        ):
            raise ValueError(f"logp and logq_ambient disagree in {path}")
        no_clip_audit = _validate_formal_no_clip(
            weights=archive["weights"],
            stored_logw_raw=archive["logw_raw"],
            target_reduced_energy=archive["target_reduced_energy"],
            logq_ambient=archive["logq_ambient"],
            valid=valid,
            support=support,
        )
        sampling = metadata.get("sampling")
        if sampling is None:
            sampling_audit = {
                "schema_status": "legacy_mapping_absent",
                "accepted_by": (
                    "archive sample count, event shape, per-sample array "
                    "shape, PF seed, solver settings, checkpoint SHA-256, "
                    "and complete archive SHA-256 locks"
                ),
                "archive_sample_count": sample_count,
                "expected_sample_count": expected_samples,
                "coordinate_shape": list(coordinates.shape),
                "weights_shape": list(np.asarray(archive["weights"]).shape),
                "logq_ambient_shape": list(
                    np.asarray(archive["logq_ambient"]).shape
                ),
                "target_reduced_energy_shape": list(
                    np.asarray(archive["target_reduced_energy"]).shape
                ),
            }
        elif isinstance(sampling, Mapping):
            sampling_audit = {
                "schema_status": "mapping_present",
                "global_num_samples": sampling.get("global_num_samples"),
                "local_num_samples": sampling.get("local_num_samples"),
                "archive_sample_count": sample_count,
                "expected_sample_count": expected_samples,
            }
        else:
            raise TypeError(
                f"sampling metadata in {path} must be a mapping when present"
            )
        endpoint_identity_fields = (
            "endpoint_distribution",
            "configured_endpoint_spec_sha256",
            "synthetic_endpoint_spec_sha256",
            "equilibrium_endpoint_spec_sha256",
        )
        endpoint_metadata_audit = {
            "schema_status": (
                "all_fields_present"
                if all(field in metadata for field in endpoint_identity_fields)
                else "legacy_fields_partially_or_fully_absent"
            ),
            "present_fields": [
                field for field in endpoint_identity_fields if field in metadata
            ],
            "missing_fields": [
                field
                for field in endpoint_identity_fields
                if field not in metadata
            ],
            "accepted_missing_fields_require_checkpoint_data_linkage": True,
        }

    expected_metadata = {
        "forward_checkpoint_sha256": forward_sha256,
        "backward_checkpoint_sha256": backward_sha256,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    backward_metadata = metadata.get("backward_metadata")
    if not isinstance(backward_metadata, Mapping):
        mismatches["backward_metadata"] = {
            "expected": "mapping",
            "actual": type(backward_metadata).__name__,
        }
    elif backward_metadata.get("parent_forward_sha256") != forward_sha256:
        mismatches["backward_metadata.parent_forward_sha256"] = {
            "expected": forward_sha256,
            "actual": backward_metadata.get("parent_forward_sha256"),
        }
    if mismatches:
        raise ValueError(
            f"PF/checkpoint provenance mismatch for {path}: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return PFProvenance(
        path=str(path),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        sample_count=sample_count,
        arrays=arrays,
        metadata=metadata,
        sampler_kind=sampler_kind,
        density_mode=density_mode,
        sampling_audit=sampling_audit,
        endpoint_metadata_audit=endpoint_metadata_audit,
        no_clip_audit=no_clip_audit,
    )


def _default_path(
    explicit: Path | None,
    *,
    root: Path,
    relative: str,
) -> Path:
    return _absolute(explicit if explicit is not None else root / relative)


def _arm_inputs(args: argparse.Namespace) -> list[ArmInput]:
    root = _absolute(args.project_root)
    seeds = tuple(args.seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("--seeds must name exactly three distinct seeds")

    mb1d_root = _default_path(
        args.mb1d_root,
        root=root,
        relative="outputs/mb_cg1d_three_seed_20260724_run1",
    )
    mb2d_legacy_root = _default_path(
        args.mb2d_legacy_root,
        root=root,
        relative="outputs/mb2d_formal_matrix_20260724",
    )
    mb2d_equilibrium_root = _default_path(
        args.mb2d_equilibrium_root,
        root=root,
        relative="outputs/mb2d_equilibrium_bridge_matrix_v1",
    )
    ala2_root = _default_path(
        args.ala2_root,
        root=root,
        relative=(
            "outputs/"
            "ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3"
        ),
    )
    ala2_positive_root = _default_path(
        args.ala2_positive_root,
        root=ala2_root,
        relative=(
            "energy_from_warm10k_formal20k_20260719_2132/"
            "pf10k_release_tol_20260726"
        ),
    )
    ala2_positive_training_root = _default_path(
        args.ala2_positive_training_root,
        root=ala2_root,
        relative="energy_from_warm10k_formal20k_20260719_2132",
    )
    ala2_mixed_root = _default_path(
        args.ala2_mixed_root,
        root=ala2_root,
        relative="energy100k_from_warm10k_backward100k_manual1",
    )

    config = root / "configs" / "experiment"
    inputs: list[ArmInput] = []
    for seed in seeds:
        run = mb1d_root / f"seed_{seed}"
        inputs.append(
            ArmInput(
                system="mb_cg1d",
                arm="energy_only",
                release_role="primary",
                data_identity="endpoint_independent_energy_bms",
                seed=seed,
                expected_samples=args.mb_samples,
                forward_checkpoint=run / "forward" / "forward_step_00100000",
                backward_checkpoint=run / "backward" / "backward_step_00100000",
                pf_archive=run / f"samples_and_weights_{args.mb_samples}.npz",
                config_path=config / "mb_cg1d.yaml",
                forward_role="forward",
                forward_step=100_000,
                backward_step=100_000,
                expected_event_shape=(1,),
            )
        )

        legacy_run = mb2d_legacy_root / f"seed_{seed}"
        legacy_specs = (
            (
                "cold_energy_only",
                "primary",
                "endpoint_independent_energy_bms",
                legacy_run / "cold_forward" / "forward_step_00100000",
                legacy_run
                / "cold_energy_backward100k"
                / "backward_step_00100000",
                legacy_run / f"cold_energy_pf{args.mb_samples}.npz",
            ),
            (
                "biased_bridge_only",
                "biased_endpoint_stress_test",
                "deliberately_biased_synthetic_full_support_v1",
                legacy_run / "bridge" / "forward_pretrain_step_00100000",
                legacy_run
                / "bridge_only_backward100k"
                / "backward_step_00100000",
                legacy_run / f"bridge_only_pf{args.mb_samples}.npz",
            ),
            (
                "biased_bridge_energy",
                "biased_endpoint_stress_test",
                "deliberately_biased_synthetic_full_support_v1",
                legacy_run
                / "bridge_energy_forward"
                / "forward_step_00100000",
                legacy_run
                / "bridge_energy_backward100k"
                / "backward_step_00100000",
                legacy_run / f"bridge_energy_pf{args.mb_samples}.npz",
            ),
        )
        for arm, role, identity, forward, backward, pf in legacy_specs:
            inputs.append(
                ArmInput(
                    system="mb2d",
                    arm=arm,
                    release_role=role,
                    data_identity=identity,
                    seed=seed,
                    expected_samples=args.mb_samples,
                    forward_checkpoint=forward,
                    backward_checkpoint=backward,
                    pf_archive=pf,
                    config_path=config / "mb2d_analytic.yaml",
                    forward_role=(
                        "forward_pretrain"
                        if arm == "biased_bridge_only"
                        else "forward"
                    ),
                    forward_step=100_000,
                    backward_step=100_000,
                    expected_event_shape=(2,),
                )
            )

        equilibrium_run = mb2d_equilibrium_root / f"seed_{seed}"
        equilibrium_specs = (
            (
                "equilibrium_bridge_only",
                equilibrium_run
                / "equilibrium_bridge"
                / "forward_pretrain_step_00100000",
                equilibrium_run
                / "equilibrium_bridge_only_backward100k"
                / "backward_step_00100000",
                equilibrium_run
                / f"equilibrium_bridge_only_pf{args.mb_samples}.npz",
            ),
            (
                "equilibrium_bridge_energy",
                equilibrium_run
                / "equilibrium_bridge_energy_forward"
                / "forward_step_00100000",
                equilibrium_run
                / "equilibrium_bridge_energy_backward100k"
                / "backward_step_00100000",
                equilibrium_run
                / f"equilibrium_bridge_energy_pf{args.mb_samples}.npz",
            ),
        )
        for arm, forward, backward, pf in equilibrium_specs:
            inputs.append(
                ArmInput(
                    system="mb2d",
                    arm=arm,
                    release_role="primary",
                    data_identity="equilibrium_exact_v1",
                    seed=seed,
                    expected_samples=args.mb_samples,
                    forward_checkpoint=forward,
                    backward_checkpoint=backward,
                    pf_archive=pf,
                    config_path=(
                        config / "mb2d_analytic_equilibrium_bridge.yaml"
                    ),
                    forward_role=(
                        "forward_pretrain"
                        if arm == "equilibrium_bridge_only"
                        else "forward"
                    ),
                    forward_step=100_000,
                    backward_step=100_000,
                    expected_event_shape=(2,),
                )
            )

    inputs.extend(
        (
            ArmInput(
                system="ala2_cg",
                arm="positive_warm10k_energy20k_pf10k",
                release_role="primary",
                data_identity=(
                    "six-bead core-beta CG-BG MACE PMF + fixed support terms, "
                    "300 K, ambient18 auxiliary-COM; warm endpoints hash-pinned "
                    "CG-BG explicit core-beta data"
                ),
                seed=3,
                expected_samples=args.ala2_positive_samples,
                forward_checkpoint=(
                    ala2_positive_training_root
                    / "forward"
                    / "forward_step_00020000"
                ),
                backward_checkpoint=(
                    ala2_positive_training_root
                    / "backward_formal20k_run1"
                    / "backward_step_00020000"
                ),
                pf_archive=(
                    ala2_positive_root
                    / f"samples_and_weights_{args.ala2_positive_samples}.npz"
                ),
                config_path=config / "ala2_cg_legacy_positive_release.yaml",
                forward_role="forward",
                forward_step=20_000,
                backward_step=20_000,
                expected_event_shape=(6, 3),
            ),
            ArmInput(
                system="ala2_cg",
                arm="mixed_low_ess_warm10k_energy100k_pf20k",
                release_role="mixed_negative_ablation",
                data_identity=(
                    "six-bead core-beta CG-BG MACE PMF + fixed support terms, "
                    "300 K, ambient18 auxiliary-COM; warm endpoints hash-pinned "
                    "CG-BG explicit core-beta data"
                ),
                seed=3,
                expected_samples=args.ala2_mixed_samples,
                forward_checkpoint=(
                    ala2_mixed_root
                    / "forward"
                    / "forward_step_00100000"
                ),
                backward_checkpoint=(
                    ala2_mixed_root
                    / "backward"
                    / "backward_step_00100000"
                ),
                pf_archive=(
                    ala2_mixed_root
                    / "pf20k_eval_run1"
                    / f"samples_and_weights_{args.ala2_mixed_samples}.npz"
                ),
                config_path=(
                    config
                    / "ala2_cg_legacy_mixed_e100_release.yaml"
                ),
                forward_role="forward",
                forward_step=100_000,
                backward_step=100_000,
                expected_event_shape=(6, 3),
            ),
        )
    )
    return inputs


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{secrets.token_hex(6)}"
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
    )


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{secrets.token_hex(6)}"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_numeric_csv(
    path: Path,
    *,
    columns: Mapping[str, Any],
) -> None:
    """Stream equal-length numeric columns without building row dictionaries."""

    if not columns:
        raise ValueError(f"refusing to write an empty numeric CSV: {path}")
    arrays = {
        str(name): np.asarray(values).reshape(-1)
        for name, values in columns.items()
    }
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"numeric CSV columns have inconsistent sizes: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{secrets.token_hex(6)}"
    try:
        if path.suffix == ".gz":
            with temporary.open("wb") as raw_handle:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_handle,
                    mtime=0,
                ) as compressed:
                    with io.TextIOWrapper(
                        compressed,
                        encoding="utf-8",
                        newline="",
                    ) as handle:
                        writer = csv.writer(handle, lineterminator="\n")
                        writer.writerow(arrays)
                        writer.writerows(zip(*arrays.values(), strict=True))
        else:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(arrays)
                writer.writerows(zip(*arrays.values(), strict=True))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        destination.parent
        / f".{destination.name}.tmp-{secrets.token_hex(6)}"
    )
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_float(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"release metric {field} is non-finite: {number}")
    return number


def _result_row(
    arm: ArmInput,
    metrics: Mapping[str, Any],
    pf: PFProvenance,
    forward: CheckpointProvenance,
    backward: CheckpointProvenance,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "system": arm.system,
        "arm": arm.arm,
        "release_role": arm.release_role,
        "data_identity": arm.data_identity,
        "seed": arm.seed,
        "num_samples": pf.sample_count,
        "formal_no_clip": True,
        "forward_checkpoint_sha256": forward.sha256,
        "backward_checkpoint_sha256": backward.sha256,
        "pf_archive_sha256": pf.sha256,
        "valid_mask_fraction": _finite_float(
            pf.no_clip_audit["valid_mask_fraction"],
            field="valid_mask_fraction",
        ),
        "support_mask_fraction": _finite_float(
            pf.no_clip_audit["support_mask_fraction"],
            field="support_mask_fraction",
        ),
        "included_sample_fraction": _finite_float(
            pf.no_clip_audit["included_sample_fraction"],
            field="included_sample_fraction",
        ),
        "ess_fraction": _finite_float(
            pf.no_clip_audit["ess_fraction"],
            field="ess_fraction",
        ),
        "max_weight": _finite_float(
            pf.no_clip_audit["max_weight"],
            field="max_weight",
        ),
        "top_0p1_percent_mass": _finite_float(
            pf.no_clip_audit["top_0p1_percent_mass"],
            field="top_0p1_percent_mass",
        ),
        "top_1_percent_mass": _finite_float(
            pf.no_clip_audit["top_1_percent_mass"],
            field="top_1_percent_mass",
        ),
        "logw_variance": _finite_float(
            pf.no_clip_audit["logw_variance"],
            field="logw_variance",
        ),
        "logw_span": _finite_float(
            pf.no_clip_audit["logw_span"],
            field="logw_span",
        ),
    }
    if arm.system in {"mb_cg1d", "ala2_cg"}:
        proposal = metrics.get("unweighted_direct")
        weighted = metrics.get("weighted_direct")
        if not isinstance(proposal, Mapping) or not isinstance(
            weighted,
            Mapping,
        ):
            raise ValueError(f"{arm.key} is missing weighted direct metrics")
        row.update(
            {
                "proposal_js": _finite_float(
                    proposal["JS_Divergence"],
                    field="proposal_js",
                ),
                "weighted_js": _finite_float(
                    weighted["JS_Divergence"],
                    field="weighted_js",
                ),
                "proposal_pmf_error": _finite_float(
                    proposal["PMF_Error"],
                    field="proposal_pmf_error",
                ),
                "weighted_pmf_error": _finite_float(
                    weighted["PMF_Error"],
                    field="weighted_pmf_error",
                ),
            }
        )
        energy_wasserstein = metrics.get("energy_wasserstein")
        if isinstance(energy_wasserstein, Mapping):
            row["proposal_energy_wasserstein_w1"] = _finite_float(
                energy_wasserstein["W1"],
                field="proposal_energy_wasserstein_w1",
            )
            row["proposal_energy_wasserstein_w2"] = _finite_float(
                energy_wasserstein["W2"],
                field="proposal_energy_wasserstein_w2",
            )
    elif arm.system == "mb2d":
        proposal = metrics.get("unweighted")
        weighted = metrics.get("weighted")
        if not isinstance(proposal, Mapping) or not isinstance(
            weighted,
            Mapping,
        ):
            raise ValueError(f"{arm.key} is missing weighted MB2D metrics")
        row.update(
            {
                "proposal_js": _finite_float(
                    proposal["js_2d"],
                    field="proposal_js",
                ),
                "weighted_js": _finite_float(
                    weighted["js_2d"],
                    field="weighted_js",
                ),
                "proposal_pmf_error": _finite_float(
                    proposal["pmf_rmse"],
                    field="proposal_pmf_error",
                ),
                "weighted_pmf_error": _finite_float(
                    weighted["pmf_rmse"],
                    field="weighted_pmf_error",
                ),
                "proposal_basin_l1": _finite_float(
                    proposal["basin_l1_error"],
                    field="proposal_basin_l1",
                ),
                "weighted_basin_l1": _finite_float(
                    weighted["basin_l1_error"],
                    field="weighted_basin_l1",
                ),
                "proposal_energy_mean_error": _finite_float(
                    proposal["energy_mean_error"],
                    field="proposal_energy_mean_error",
                ),
                "weighted_energy_mean_error": _finite_float(
                    weighted["energy_mean_error"],
                    field="weighted_energy_mean_error",
                ),
            }
        )
        row["proposal_energy_mean_abs_error"] = abs(
            row["proposal_energy_mean_error"]
        )
        row["weighted_energy_mean_abs_error"] = abs(
            row["weighted_energy_mean_error"]
        )
    else:
        raise AssertionError(f"unhandled release system {arm.system!r}")
    row["js_improvement"] = row["proposal_js"] - row["weighted_js"]
    row["pmf_error_improvement"] = (
        row["proposal_pmf_error"] - row["weighted_pmf_error"]
    )
    return row


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metric_fields = (
        "proposal_js",
        "weighted_js",
        "js_improvement",
        "proposal_pmf_error",
        "weighted_pmf_error",
        "pmf_error_improvement",
        "proposal_basin_l1",
        "weighted_basin_l1",
        "proposal_energy_mean_error",
        "weighted_energy_mean_error",
        "proposal_energy_mean_abs_error",
        "weighted_energy_mean_abs_error",
        "proposal_energy_wasserstein_w1",
        "proposal_energy_wasserstein_w2",
        "valid_mask_fraction",
        "support_mask_fraction",
        "included_sample_fraction",
        "ess_fraction",
        "max_weight",
        "top_0p1_percent_mass",
        "top_1_percent_mass",
        "logw_variance",
        "logw_span",
    )
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["system"]),
            str(row["arm"]),
            str(row["release_role"]),
            str(row["data_identity"]),
        )
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (system, arm, role, identity), group in sorted(grouped.items()):
        for metric in metric_fields:
            values = [
                float(row[metric])
                for row in group
                if metric in row and row[metric] not in (None, "")
            ]
            if not values:
                continue
            result.append(
                {
                    "system": system,
                    "arm": arm,
                    "release_role": role,
                    "data_identity": identity,
                    "n": len(values),
                    "seeds": ",".join(
                        str(row["seed"]) for row in group
                    ),
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return result


def _copy_release_configs(
    *,
    stage: Path,
    inputs: Sequence[ArmInput],
    project_root: Path,
) -> list[dict[str, Any]]:
    sources = {arm.config_path for arm in inputs}
    sources.update(
        {
            project_root
            / "data"
            / "mb2d_equilibrium_exact_v1"
            / "endpoints.manifest.json",
            project_root
            / "data"
            / "mb2d_equilibrium_exact_v1"
            / "README.md",
        }
    )
    copied: list[dict[str, Any]] = []
    for source in sorted(sources):
        if not source.is_file():
            raise FileNotFoundError(f"release config/input manifest is missing: {source}")
        try:
            relative = source.relative_to(project_root)
        except ValueError:
            relative = Path("external") / source.name
        destination = (
            stage / relative
            if relative.parts and relative.parts[0] == "configs"
            else stage / "configs" / relative
        )
        _atomic_copy(source, destination)
        destination_sha256 = _sha256_file(destination)
        source_sha256 = _sha256_file(source)
        if destination_sha256 != source_sha256:
            raise RuntimeError(f"copied release config changed in transit: {source}")
        copied.append(
            {
                "source": str(source),
                "release_path": destination.relative_to(stage).as_posix(),
                "sha256": destination_sha256,
            }
        )
    return copied


def _git_provenance(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ("git", "-C", str(project_root), *arguments),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def _validate_visual_contract_sources(project_root: Path) -> dict[str, Any]:
    """Fail closed if release evaluator styling or shared FES scaling drifts."""

    mb_path = project_root / "src" / "cg_bms_jax" / "evaluation" / "mb.py"
    mb2d_path = project_root / "src" / "cg_bms_jax" / "evaluation" / "mb2d.py"
    ala2_path = project_root / "src" / "cg_bms_jax" / "evaluation" / "ala2.py"
    style_path = (
        project_root / "src" / "cg_bms_jax" / "evaluation" / "plot_style.py"
    )
    source_text = {
        path: path.read_text(encoding="utf-8")
        for path in (mb_path, mb2d_path, ala2_path, style_path)
    }
    requirements = {
        mb_path: (
            "EXACT_COLOR",
            "filled_stairs",
            "plot_style_metadata",
            'role="reference"',
            'role="proposal"',
            'role="reweighted"',
        ),
        mb2d_path: (
            "EXACT_COLOR",
            "filled_curve",
            "filled_stairs",
            "plot_style_metadata",
            'role="proposal"',
            'role="reweighted"',
            'cmap="viridis"',
            "vmin=0.0",
            "vmax=12.0 * float(kT)",
            'label="Free energy"',
            "reference.x_edges[0]",
            "reference.x_edges[-1]",
            "reference.y_edges[0]",
            "reference.y_edges[-1]",
        ),
        ala2_path: (
            "filled_stairs",
            "plot_style_metadata",
            'role="reference"',
            'role="proposal"',
            'role="reweighted"',
            '("reference", "proposal", "reweighted")',
            'f"{variant}_{slug}_ramachandran_fes.png"',
        ),
        style_path: (
            'EXACT_COLOR = "#000000"',
            'REFERENCE_COLOR = "#8FD18B"',
            'PROPOSAL_COLOR = "#F2A174"',
            'REWEIGHTED_COLOR = "#4472C4"',
            'REWEIGHTED_FILL_COLOR = "#B9CBEA"',
            "REFERENCE_ALPHA = 0.52",
            "PROPOSAL_ALPHA = 0.56",
            "REWEIGHTED_ALPHA = 0.58",
            '"density_rendering": "filled_with_outline"',
            '"two_dimensional_panels": "standalone_no_overlay"',
            "axis.fill_between(",
            "fill=True",
            "facecolor=fill_color",
            "color=edge_color",
        ),
    }
    failures: list[str] = []
    for path, tokens in requirements.items():
        text = source_text[path]
        for token in tokens:
            if token not in text:
                failures.append(f"{path}: missing visual-contract token {token!r}")
    if failures:
        raise RuntimeError(
            "MB release visual contract is not satisfied:\n  "
            + "\n  ".join(failures)
        )
    return {
        "palette": RELEASE_PALETTE,
        "fill_style": RELEASE_FILL_STYLE,
        "one_dimensional_density_rendering": "filled_with_outline",
        "exact_or_true_rendering": "black_line_only",
        "mb2d_fes": {
            "colormap": MB2D_FES_COLORMAP,
            "vmin": MB2D_FES_VMIN,
            "vmax_kT": MB2D_FES_VMAX_KT,
            "physical_limits": MB2D_PHYSICAL_LIMITS,
            "colorbar_label": "Free energy",
        },
        "ala2_ramachandran": {
            "rendering": "standalone_no_overlay",
            "panels": ["reference", "proposal", "reweighted"],
        },
        "source_files": {
            str(path): _sha256_file(path)
            for path in (mb_path, mb2d_path, ala2_path, style_path)
        },
        "verification": "PASS",
    }


def _capture_input_provenance(
    *,
    args: argparse.Namespace,
    project_root: Path,
    inputs: Sequence[ArmInput],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Hash every external datum and code/config input used by the freeze."""

    import yaml

    snapshots: dict[str, str] = {}

    def record(path: Path, *, role: str) -> dict[str, Any]:
        resolved = _absolute(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"release input is missing: {resolved}")
        digest = _sha256_file(resolved)
        snapshots[str(resolved)] = digest
        return {
            "path": str(resolved),
            "role": role,
            "sha256": digest,
            "size_bytes": resolved.stat().st_size,
        }

    asset_manifest_path = project_root / "assets" / "manifest.yaml"
    asset_manifest_record = record(
        asset_manifest_path,
        role="pinned_huggingface_asset_manifest",
    )
    asset_manifest = yaml.safe_load(
        asset_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(asset_manifest, Mapping):
        raise TypeError(f"asset manifest must be a mapping: {asset_manifest_path}")
    asset_entries = asset_manifest.get("files")
    if not isinstance(asset_entries, Mapping):
        raise TypeError(f"asset manifest files are missing: {asset_manifest_path}")
    required_asset_names = (
        "mb_train",
        "mb_pmf",
        "mb_reference",
        "ala2_train",
        "ala2_pmf",
        "ala2_reference",
        "ala2_topology",
        "ala2_implicit_reference",
    )
    assets: list[dict[str, Any]] = []
    for name in required_asset_names:
        entry = asset_entries.get(name)
        if not isinstance(entry, Mapping):
            raise KeyError(f"asset manifest entry is missing: {name}")
        path = project_root / "assets" / "cache" / str(entry["path"])
        asset = record(path, role=f"asset:{name}")
        expected_sha = str(entry.get("sha256", "")).lower()
        expected_size = int(entry.get("size_bytes", -1))
        if asset["sha256"] != expected_sha:
            raise ValueError(
                f"asset checksum mismatch for {name}: expected "
                f"{expected_sha}, found {asset['sha256']}"
            )
        if asset["size_bytes"] != expected_size:
            raise ValueError(
                f"asset size mismatch for {name}: expected {expected_size}, "
                f"found {asset['size_bytes']}"
            )
        asset.update(
            {
                "name": name,
                "manifest_path": str(entry["path"]),
                "manifest_revision": asset_manifest.get("revision"),
                "manifest_repo_id": asset_manifest.get("repo_id"),
            }
        )
        assets.append(asset)

    requested_references = {
        _absolute(args.mb1d_reference): "mb_reference",
        _absolute(args.ala2_reference): "ala2_reference",
        _absolute(args.ala2_implicit_reference): "ala2_implicit_reference",
    }
    indexed_assets = {
        _absolute(Path(str(asset["path"]))): str(asset["name"])
        for asset in assets
    }
    for path, expected_name in requested_references.items():
        if indexed_assets.get(path) != expected_name:
            raise ValueError(
                f"evaluation reference {path} is not the pinned "
                f"{expected_name} asset"
            )

    endpoint_manifest_path = (
        project_root
        / "data"
        / "mb2d_equilibrium_exact_v1"
        / "endpoints.manifest.json"
    )
    endpoint_manifest_record = record(
        endpoint_manifest_path,
        role="exact_equilibrium_endpoint_manifest",
    )
    endpoint_manifest = json.loads(
        endpoint_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(endpoint_manifest, Mapping):
        raise TypeError("MB2D endpoint manifest must be a mapping")
    validation = endpoint_manifest.get("validation")
    if not isinstance(validation, Mapping) or validation.get("accepted") is not True:
        raise ValueError("MB2D exact endpoint dataset was not accepted by validation")
    endpoint_path = project_root / str(endpoint_manifest.get("path", ""))
    endpoint_record = record(
        endpoint_path,
        role="exact_equilibrium_endpoint_dataset",
    )
    manifest_endpoint_sha = str(endpoint_manifest.get("dataset_sha256", "")).lower()
    if (
        manifest_endpoint_sha != MB2D_EQUILIBRIUM_ENDPOINT_SHA256
        or endpoint_record["sha256"] != MB2D_EQUILIBRIUM_ENDPOINT_SHA256
    ):
        raise ValueError(
            "MB2D equilibrium endpoint dataset is not the immutable "
            f"{MB2D_EQUILIBRIUM_ENDPOINT_SHA256} dataset"
        )

    config_records = [
        record(path, role="experiment_config")
        for path in sorted({_absolute(arm.config_path) for arm in inputs})
    ]
    code_paths = (
        Path(__file__).resolve(),
        project_root / "scripts" / "validate_release_scope.py",
        project_root / "src" / "cg_bms_jax" / "evaluation" / "__init__.py",
        project_root / "src" / "cg_bms_jax" / "evaluation" / "ala2.py",
        project_root / "src" / "cg_bms_jax" / "evaluation" / "mb.py",
        project_root / "src" / "cg_bms_jax" / "evaluation" / "mb2d.py",
        project_root / "src" / "cg_bms_jax" / "evaluation" / "metrics.py",
        project_root / "src" / "cg_bms_jax" / "evaluation" / "plot_style.py",
    )
    code_records = [record(path, role="release_evaluation_code") for path in code_paths]
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "asset_manifest": asset_manifest_record,
            "assets": assets,
            "mb2d_equilibrium": {
                "manifest_file": endpoint_manifest_record,
                "dataset_file": endpoint_record,
                "manifest": endpoint_manifest,
            },
            "experiment_configs": config_records,
            "evaluation_code": code_records,
        },
        snapshots,
    )


def _verify_input_snapshots(snapshots: Mapping[str, str]) -> None:
    """Close the normal evaluation TOCTOU window before publishing."""

    mismatches: list[str] = []
    for raw_path, expected in snapshots.items():
        path = Path(raw_path)
        if not path.is_file():
            mismatches.append(f"{path}: disappeared")
            continue
        actual = _sha256_file(path)
        if actual != expected:
            mismatches.append(f"{path}: expected {expected}, found {actual}")
    if mismatches:
        raise RuntimeError(
            "release inputs changed during evaluation:\n  "
            + "\n  ".join(mismatches)
        )


def _verify_large_inputs_unchanged(
    *,
    inputs: Sequence[ArmInput],
    provenance: Mapping[
        str,
        tuple[CheckpointProvenance, CheckpointProvenance, PFProvenance],
    ],
) -> None:
    mismatches: list[str] = []
    for arm in inputs:
        forward, backward, pf = provenance[arm.key]
        actual_forward = _checkpoint_sha256(arm.forward_checkpoint)
        actual_backward = _checkpoint_sha256(arm.backward_checkpoint)
        actual_pf = _sha256_file(arm.pf_archive)
        for label, path, expected, actual in (
            (
                "forward checkpoint",
                arm.forward_checkpoint,
                forward.sha256,
                actual_forward,
            ),
            (
                "backward checkpoint",
                arm.backward_checkpoint,
                backward.sha256,
                actual_backward,
            ),
            ("PF archive", arm.pf_archive, pf.sha256, actual_pf),
        ):
            if expected != actual:
                mismatches.append(
                    f"{arm.key} {label} {path}: expected {expected}, found {actual}"
                )
    if mismatches:
        raise RuntimeError(
            "large release inputs changed during evaluation:\n  "
            + "\n  ".join(mismatches)
        )


def _directory_size_bytes(path: Path) -> int:
    return int(sum(file.stat().st_size for file in path.rglob("*") if file.is_file()))


def _source_results_index(
    *,
    inputs: Sequence[ArmInput],
    provenance: Mapping[
        str,
        tuple[CheckpointProvenance, CheckpointProvenance, PFProvenance],
    ],
    input_provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Index immutable NAS sources without copying their heavy payloads."""

    rows: list[dict[str, Any]] = []
    for arm in inputs:
        forward, backward, pf = provenance[arm.key]
        for kind, source_path, digest, size in (
            (
                "forward_checkpoint",
                Path(forward.path),
                forward.sha256,
                _directory_size_bytes(Path(forward.path)),
            ),
            (
                "backward_checkpoint",
                Path(backward.path),
                backward.sha256,
                _directory_size_bytes(Path(backward.path)),
            ),
            (
                "pf_archive",
                Path(pf.path),
                pf.sha256,
                pf.size_bytes,
            ),
        ):
            rows.append(
                {
                    "system": _release_family(arm.system),
                    "arm": arm.arm,
                    "seed": arm.seed,
                    "kind": kind,
                    "path": str(source_path),
                    "size_bytes": int(size),
                    "sha256": digest,
                    "copied_into_release": False,
                }
            )
    for asset in input_provenance.get("assets", []):
        if not isinstance(asset, Mapping):
            continue
        name = str(asset.get("name"))
        system = (
            "mb_cg1d"
            if name.startswith("mb_")
            else ("ala2_cg" if name.startswith("ala2_") else "shared")
        )
        rows.append(
            {
                "system": system,
                "arm": "shared_input",
                "seed": "",
                "kind": f"pinned_asset:{name}",
                "path": str(asset["path"]),
                "size_bytes": int(asset["size_bytes"]),
                "sha256": str(asset["sha256"]),
                "copied_into_release": False,
            }
        )
    endpoint = input_provenance.get("mb2d_equilibrium")
    if isinstance(endpoint, Mapping):
        for name, kind in (
            ("manifest_file", "equilibrium_endpoint_manifest"),
            ("dataset_file", "equilibrium_endpoint_dataset"),
        ):
            record = endpoint.get(name)
            if isinstance(record, Mapping):
                rows.append(
                    {
                        "system": "mb2d_analytic",
                        "arm": "equilibrium_bridge_inputs",
                        "seed": "",
                        "kind": kind,
                        "path": str(record["path"]),
                        "size_bytes": int(record["size_bytes"]),
                        "sha256": str(record["sha256"]),
                        "copied_into_release": False,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            str(row["system"]),
            str(row["arm"]),
            str(row["seed"]),
            str(row["kind"]),
        ),
    )


def _preflight(
    inputs: Sequence[ArmInput],
) -> dict[str, tuple[CheckpointProvenance, CheckpointProvenance, PFProvenance]]:
    """Validate every large input before creating the staging directory."""

    result: dict[
        str,
        tuple[CheckpointProvenance, CheckpointProvenance, PFProvenance],
    ] = {}
    identity_cache: dict[str, tuple[dict[str, Any], str]] = {}
    for arm in inputs:
        for candidate in (
            arm.forward_checkpoint,
            arm.backward_checkpoint,
            arm.pf_archive,
            arm.config_path,
        ):
            if any(marker in str(candidate).lower() for marker in ALL_ATOM_MARKERS):
                raise ValueError(
                    f"all-atom input is outside the release scope: {candidate}"
                )
        if not arm.config_path.is_file():
            raise FileNotFoundError(f"experiment config is missing: {arm.config_path}")
        config_key = str(_absolute(arm.config_path))
        if config_key not in identity_cache:
            identity_cache[config_key] = _expected_runtime_identity(
                _absolute(arm.config_path)
            )
        expected_identity, expected_config_sha256 = identity_cache[config_key]
        forward = _checkpoint_provenance(arm.forward_checkpoint)
        backward = _checkpoint_provenance(arm.backward_checkpoint)
        pf = _pf_provenance(
            arm.pf_archive,
            expected_samples=arm.expected_samples,
            forward_sha256=forward.sha256,
            backward_sha256=backward.sha256,
            expected_event_shape=arm.expected_event_shape,
        )
        _validate_arm_identity(
            arm,
            forward=forward,
            backward=backward,
            pf=pf,
            expected_identity=expected_identity,
            expected_config_sha256=expected_config_sha256,
        )
        result[arm.key] = (forward, backward, pf)

    # A bridge-to-Energy label is only allowed when its initial controller is
    # exactly the released bridge-only controller from the same seed/data arm.
    by_identity = {(arm.system, arm.arm, arm.seed): arm for arm in inputs}
    linked_pairs = (
        ("mb2d", "biased_bridge_only", "biased_bridge_energy"),
        ("mb2d", "equilibrium_bridge_only", "equilibrium_bridge_energy"),
    )
    for system, bridge_arm, energy_arm in linked_pairs:
        for seed in sorted({arm.seed for arm in inputs if arm.system == system}):
            bridge = by_identity.get((system, bridge_arm, seed))
            energy = by_identity.get((system, energy_arm, seed))
            if bridge is None or energy is None:
                raise ValueError(
                    f"missing linked release arms for {system} seed {seed}: "
                    f"{bridge_arm}, {energy_arm}"
                )
            bridge_forward = result[bridge.key][0]
            energy_metadata = _checkpoint_metadata(result[energy.key][0])
            if (
                energy_metadata.get("initial_controller_sha256")
                != bridge_forward.sha256
            ):
                raise ValueError(
                    f"{energy.key} was not initialized from the released "
                    f"{bridge.key} controller"
                )
    return result


def _write_provenance(
    *,
    stage: Path,
    arm: ArmInput,
    forward: CheckpointProvenance,
    backward: CheckpointProvenance,
    pf: PFProvenance,
) -> None:
    destination = (
        stage
        / "provenance"
        / _release_family(arm.system)
        / arm.arm
        / f"seed_{arm.seed}"
    )
    _atomic_json(
        destination / "forward_checkpoint.json",
        {
            "source_path": forward.path,
            "sha256": forward.sha256,
            "copied": False,
            "manifest": forward.manifest,
        },
    )
    _atomic_json(
        destination / "backward_checkpoint.json",
        {
            "source_path": backward.path,
            "sha256": backward.sha256,
            "copied": False,
            "manifest": backward.manifest,
        },
    )
    _atomic_json(
        destination / "pf_archive.json",
        {
            "source_path": pf.path,
            "sha256": pf.sha256,
            "size_bytes": pf.size_bytes,
            "copied": False,
            "sample_count": pf.sample_count,
            "sampler_kind": pf.sampler_kind,
            "density_mode": pf.density_mode,
            "arrays": pf.arrays,
            "metadata": pf.metadata,
            "sampling_provenance_audit": pf.sampling_audit,
            "endpoint_metadata_audit": pf.endpoint_metadata_audit,
            "no_clip_audit": pf.no_clip_audit,
        },
    )


def _diagnostic_clip_mapping(path: Path) -> dict[str, np.ndarray]:
    """Build an explicit drop-top-1% view from the audited raw formula."""

    with np.load(path, allow_pickle=False) as archive:
        coordinates = np.asarray(archive["R"])
        energy = np.asarray(archive["U"])
        raw = (
            -np.asarray(
                archive["target_reduced_energy"],
                dtype=np.float64,
            ).reshape(-1)
            - np.asarray(archive["logq_ambient"], dtype=np.float64).reshape(-1)
        )
        included = (
            np.asarray(archive["valid_mask"], dtype=bool).reshape(-1)
            & np.asarray(archive["support_mask"], dtype=bool).reshape(-1)
            & np.isfinite(raw)
        )
    diagnostic_raw = np.full_like(raw, -np.inf)
    diagnostic_raw[included] = raw[included]
    return {"R": coordinates, "U": energy, "logw": diagnostic_raw}


def _write_weight_tail_panel(
    *,
    archive: Path,
    output_dir: Path,
    arm: ArmInput,
    audit: Mapping[str, Any],
) -> None:
    """Save a standalone, formal no-clip concentration diagnostic."""

    import matplotlib.pyplot as plt

    with np.load(archive, allow_pickle=False) as data:
        weights = np.asarray(data["weights"], dtype=np.float64).reshape(-1)
    order = np.sort(weights)[::-1]
    cumulative = np.cumsum(order)
    fraction = np.arange(1, order.size + 1, dtype=np.float64) / order.size
    # A logarithmically spaced curve is visually indistinguishable from the
    # full N-point staircase while keeping the Git payload compact.  The exact
    # scalar tail masses remain recorded in the adjacent summary JSON.
    selected = np.unique(
        np.concatenate(
            (
                np.array([0, order.size - 1], dtype=np.int64),
                np.geomspace(
                    1,
                    order.size,
                    num=min(4096, order.size),
                ).astype(np.int64)
                - 1,
            )
        )
    )
    plotted_fraction = fraction[selected]
    plotted_cumulative = cumulative[selected]

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir = output_dir / "plot_data"
    _atomic_numeric_csv(
        plot_data_dir / "weight_tail_curve.csv",
        columns={
            "top_fraction": plotted_fraction,
            "top_percent": 100.0 * plotted_fraction,
            "cumulative_weight_mass": plotted_cumulative,
        },
    )
    _atomic_json(
        plot_data_dir / "weight_tail_summary.json",
        {
            "formal": True,
            "clip": None,
            "num_generated_samples": int(order.size),
            "curve_points": int(selected.size),
            "curve_sampling": "deterministic_log_spaced_indices_plus_endpoints",
            **dict(audit),
        },
    )
    figure, axis = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    axis.plot(
        100.0 * plotted_fraction,
        plotted_cumulative,
        color="#3b6dcc",
        linewidth=2.0,
        label="Formal no-clip weights",
    )
    axis.set_xscale("log")
    axis.set_xlim(max(100.0 / order.size, 1.0e-4), 100.0)
    axis.set_ylim(0.0, 1.015)
    axis.set_xlabel("Top fraction of samples (%)")
    axis.set_ylabel("Cumulative importance-weight mass")
    axis.axvline(0.1, color="#777777", linewidth=1.0, linestyle=":")
    axis.axvline(1.0, color="#777777", linewidth=1.0, linestyle="--")
    axis.grid(alpha=0.22)
    axis.set_title(
        f"{_release_family(arm.system)} · {arm.arm} · seed {arm.seed}\n"
        "formal PF-ODE, no clipping"
    )
    annotation = "\n".join(
        (
            f"ESS/N = {float(audit['ess_fraction']):.4g}",
            f"max w = {float(audit['max_weight']):.4g}",
            f"top 0.1% mass = {float(audit['top_0p1_percent_mass']):.4g}",
            f"top 1% mass = {float(audit['top_1_percent_mass']):.4g}",
            f"Var(raw log w) = {float(audit['logw_variance']):.4g}",
            f"raw log w span = {float(audit['logw_span']):.4g}",
        )
    )
    axis.text(
        0.985,
        0.055,
        annotation,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.92,
        },
    )
    axis.legend(loc="upper left")
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"weight_tail_diagnostic.{suffix}",
            dpi=220 if suffix == "png" else None,
        )
    plt.close(figure)


def _write_mb_plot_data(
    *,
    target: Path,
    sample: Path | Mapping[str, Any],
    output_dir: Path,
    kT: float,
    clip_percentile: float | None,
    clip_mode: str,
) -> None:
    """Persist the numerical arrays behind every MB1D standalone panel."""

    from cg_bms_jax.evaluation import mb as mb_eval
    from cg_bms_jax.evaluation.metrics import resolve_weights

    target_data = mb_eval._load(target)
    sample_data = mb_eval._load(sample)
    x_reference = mb_eval._extract_mb_x(
        target_data["R"],
        full_reference=True,
    )
    x_proposal = mb_eval._extract_mb_x(
        sample_data["R"],
        full_reference=False,
    )
    weights, _, source = resolve_weights(
        sample_data,
        kT=kT,
        clip_percentile=clip_percentile,
        clip_mode=clip_mode,
    )
    if weights is None:
        raise ValueError("frozen MB plot data requires importance weights")

    plot_data = output_dir / "plot_data"
    edges = np.linspace(0.0, 50.0, 101)
    centers = 0.5 * (edges[:-1] + edges[1:])
    reference_density, _ = np.histogram(
        x_reference,
        bins=edges,
        density=True,
    )
    proposal_density, _ = np.histogram(
        x_proposal,
        bins=edges,
        density=True,
    )
    weighted_density, _ = np.histogram(
        x_proposal,
        bins=edges,
        density=True,
        weights=weights,
    )
    _atomic_numeric_csv(
        plot_data / "density_histogram.csv",
        columns={
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": centers,
            "reference_density": reference_density,
            "proposal_density": proposal_density,
            "reweighted_density": weighted_density,
        },
    )

    grid = np.linspace(0.0, 50.0, 300)
    exact_density = mb_eval.mb_exact_marginal(grid, kT=kT)
    exact_free = -kT * np.log(
        np.maximum(exact_density, np.finfo(np.float64).tiny)
    )
    exact_free -= np.min(exact_free)
    _atomic_numeric_csv(
        plot_data / "density_and_free_energy.csv",
        columns={
            "x": grid,
            "exact_density": exact_density,
            "exact_free_energy": exact_free,
            "reference_free_energy": mb_eval._kde_free_energy(
                x_reference,
                grid,
                kT,
                None,
            ),
            "proposal_free_energy": mb_eval._kde_free_energy(
                x_proposal,
                grid,
                kT,
                None,
            ),
            "reweighted_free_energy": mb_eval._kde_free_energy(
                x_proposal,
                grid,
                kT,
                weights,
            ),
        },
    )
    _atomic_json(
        plot_data / "plot_data_manifest.json",
        {
            "system": "mb_cg1d",
            "formal": clip_percentile is None,
            "clip_percentile": clip_percentile,
            "clip_mode": None if clip_percentile is None else clip_mode,
            "weight_source": source,
            "palette": RELEASE_PALETTE,
            "fill_style": RELEASE_FILL_STYLE,
            "kT": kT,
            "density_histogram_bins": 100,
            "free_energy_grid_points": 300,
            "figure_xlim": [10.0, 50.0],
            "all_derived_arrays_are_plot_data_not_raw_samples": True,
        },
    )


def _write_mb2d_plot_data(
    *,
    sample: Path | Mapping[str, Any],
    output_dir: Path,
    bins: int,
    energy_bins_count: int,
    clip_percentile: float | None,
    clip_mode: str,
    weight_view: str | None = None,
) -> None:
    """Persist exact/proposal/reweighted MB2D panel arrays."""

    from cg_bms_jax.evaluation import mb2d as mb2d_eval

    data = mb2d_eval._load(sample)
    coordinates = mb2d_eval.extract_mb2d_coordinates(data["R"])
    reference = mb2d_eval.analytic_mb2d_reference(beta=1.0, bins=bins)
    weights, _, source = mb2d_eval._resolve_mb2d_weights(
        data,
        kT=1.0,
        clip_percentile=clip_percentile,
        clip_mode=clip_mode,
    )
    if weights is None:
        raise ValueError("frozen MB2D plot data requires importance weights")
    proposal_probability, proposal_in_domain = mb2d_eval._normalized_histogram(
        coordinates,
        reference,
    )
    weighted_probability, weighted_in_domain = mb2d_eval._normalized_histogram(
        coordinates,
        reference,
        weights,
    )
    exact_fes = mb2d_eval._free_energy(reference.probability, reference.beta)
    proposal_fes = mb2d_eval._free_energy(
        proposal_probability,
        reference.beta,
    )
    weighted_fes = mb2d_eval._free_energy(
        weighted_probability,
        reference.beta,
    )
    xx, yy = np.meshgrid(
        reference.x_centers,
        reference.y_centers,
        indexing="ij",
    )
    x_left = np.broadcast_to(
        reference.x_edges[:-1, None],
        xx.shape,
    )
    x_right = np.broadcast_to(
        reference.x_edges[1:, None],
        xx.shape,
    )
    y_left = np.broadcast_to(
        reference.y_edges[None, :-1],
        yy.shape,
    )
    y_right = np.broadcast_to(
        reference.y_edges[None, 1:],
        yy.shape,
    )
    plot_data = output_dir / "plot_data"
    _atomic_numeric_csv(
        plot_data / "fes_grid.csv.gz",
        columns={
            "x_left": x_left,
            "x_right": x_right,
            "x_center": xx,
            "y_left": y_left,
            "y_right": y_right,
            "y_center": yy,
            "exact_probability": reference.probability,
            "proposal_probability": proposal_probability,
            "reweighted_probability": weighted_probability,
            "exact_free_energy": exact_fes,
            "proposal_free_energy": proposal_fes,
            "reweighted_free_energy": weighted_fes,
            "exact_display_free_energy": np.minimum(
                exact_fes,
                MB2D_FES_VMAX_KT,
            ),
            "proposal_display_free_energy": np.minimum(
                proposal_fes,
                MB2D_FES_VMAX_KT,
            ),
            "reweighted_display_free_energy": np.minimum(
                weighted_fes,
                MB2D_FES_VMAX_KT,
            ),
        },
    )
    _atomic_numeric_csv(
        plot_data / "x_marginal.csv",
        columns={
            "x": reference.x_centers,
            "exact_probability_per_bin": np.sum(reference.probability, axis=1),
            "proposal_probability_per_bin": np.sum(
                proposal_probability,
                axis=1,
            ),
            "reweighted_probability_per_bin": np.sum(
                weighted_probability,
                axis=1,
            ),
        },
    )
    _atomic_numeric_csv(
        plot_data / "y_marginal.csv",
        columns={
            "y": reference.y_centers,
            "exact_probability_per_bin": np.sum(reference.probability, axis=0),
            "proposal_probability_per_bin": np.sum(
                proposal_probability,
                axis=0,
            ),
            "reweighted_probability_per_bin": np.sum(
                weighted_probability,
                axis=0,
            ),
        },
    )

    sample_energy = mb2d_eval.muller_brown_energy_numpy(coordinates)
    exact_energy = reference.energy.reshape(-1)
    exact_weights = reference.probability.reshape(-1)
    lower, upper = mb2d_eval._energy_plot_view(
        exact_energy,
        exact_weights,
        lower_quantile=FORMAL_ENERGY_LOWER_QUANTILE,
        upper_quantile=FORMAL_ENERGY_UPPER_QUANTILE,
        padding_fraction=FORMAL_ENERGY_VIEW_PADDING,
    )
    energy_edges = np.linspace(lower, upper, energy_bins_count + 1)
    _atomic_numeric_csv(
        plot_data / "energy_distribution.csv",
        columns={
            "bin_left": energy_edges[:-1],
            "bin_right": energy_edges[1:],
            "bin_center": 0.5 * (energy_edges[:-1] + energy_edges[1:]),
            "exact_density": mb2d_eval._energy_density_in_view(
                exact_energy,
                energy_edges,
                weights=exact_weights,
            ),
            "proposal_density": mb2d_eval._energy_density_in_view(
                sample_energy,
                energy_edges,
            ),
            "reweighted_density": mb2d_eval._energy_density_in_view(
                sample_energy,
                energy_edges,
                weights=weights,
            ),
        },
    )
    resolved_view = (
        weight_view
        if weight_view is not None
        else (
            "formal_no_clip"
            if clip_percentile is None
            else "diagnostic_drop_top_1_percent"
        )
    )
    if resolved_view not in {
        "formal_no_clip",
        "diagnostic_drop_top_1_percent",
    }:
        raise ValueError(f"unknown MB2D selected weight view: {resolved_view}")
    reported_clip = (
        None
        if resolved_view == "formal_no_clip"
        else (
            float(clip_percentile)
            if clip_percentile is not None
            else 99.0
        )
    )
    _atomic_json(
        plot_data / "plot_data_manifest.json",
        {
            "system": "mb2d_analytic",
            "formal": resolved_view == "formal_no_clip",
            "weight_view": resolved_view,
            "clip_percentile": reported_clip,
            "clip_mode": (
                None if resolved_view == "formal_no_clip" else clip_mode
            ),
            "weight_source": source,
            "grid_bins": bins,
            "visual_contract": {
                "palette": RELEASE_PALETTE,
                "fill_style": RELEASE_FILL_STYLE,
                "fes_colormap": MB2D_FES_COLORMAP,
                "fes_vmin": MB2D_FES_VMIN,
                "fes_vmax_kT": MB2D_FES_VMAX_KT,
                "physical_limits": MB2D_PHYSICAL_LIMITS,
                "colorbar_label": "Free energy",
            },
            "energy_histogram_bins": energy_bins_count,
            "energy_view": {
                "display_only": True,
                "limits": [lower, upper],
                "lower_reference_quantile": FORMAL_ENERGY_LOWER_QUANTILE,
                "upper_reference_quantile": FORMAL_ENERGY_UPPER_QUANTILE,
                "padding_fraction": FORMAL_ENERGY_VIEW_PADDING,
                "exact": mb2d_eval._energy_mass_summary(
                    exact_energy,
                    lower=lower,
                    upper=upper,
                    weights=exact_weights,
                ),
                "proposal": mb2d_eval._energy_mass_summary(
                    sample_energy,
                    lower=lower,
                    upper=upper,
                ),
                "reweighted": mb2d_eval._energy_mass_summary(
                    sample_energy,
                    lower=lower,
                    upper=upper,
                    weights=weights,
                ),
            },
            "proposal_in_domain_mass": proposal_in_domain,
            "reweighted_in_domain_mass": weighted_in_domain,
            "all_derived_arrays_are_plot_data_not_raw_samples": True,
        },
    )


def _ala2_rama_grid(
    angles: np.ndarray,
    *,
    weights: np.ndarray | None,
    kT: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    density, phi_edges, psi_edges = np.histogram2d(
        angles[:, 0],
        angles[:, 1],
        bins=100,
        range=((-np.pi, np.pi), (-np.pi, np.pi)),
        density=True,
        weights=weights,
    )
    fes = -(kT / 4.184) * np.log(
        np.maximum(density, np.finfo(np.float64).tiny)
    )
    fes -= np.nanmin(fes)
    return density, fes, phi_edges, psi_edges


def _write_ala2_plot_data(
    *,
    target: Path,
    sample: Path | Mapping[str, Any],
    implicit: Path,
    output_dir: Path,
    kT: float,
    clip_percentile: float | None,
    clip_mode: str,
) -> None:
    """Persist the numerical arrays behind CG Ala2 standalone panels."""

    from cg_bms_jax.evaluation import ala2 as ala2_eval

    reference = ala2_eval.prepare_ala2_dataset(
        target,
        variant="ala2_cb",
        kT=kT,
    )
    proposal = ala2_eval.prepare_ala2_dataset(
        sample,
        variant="ala2_cb",
        kT=kT,
        clip_percentile=clip_percentile,
        clip_mode=clip_mode,
    )
    implicit_data = ala2_eval.prepare_ala2_dataset(
        implicit,
        variant="ala2_implicit",
        kT=kT,
    )
    if proposal.weights is None:
        raise ValueError("frozen CG Ala2 plot data requires importance weights")

    reference_density, reference_fes, phi_edges, psi_edges = _ala2_rama_grid(
        reference.dihedrals,
        weights=None,
        kT=kT,
    )
    proposal_density, proposal_fes, _, _ = _ala2_rama_grid(
        proposal.dihedrals,
        weights=None,
        kT=kT,
    )
    weighted_density, weighted_fes, _, _ = _ala2_rama_grid(
        proposal.dihedrals,
        weights=proposal.weights,
        kT=kT,
    )
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    psi_centers = 0.5 * (psi_edges[:-1] + psi_edges[1:])
    phi_grid, psi_grid = np.meshgrid(
        phi_centers,
        psi_centers,
        indexing="ij",
    )
    plot_data = output_dir / "plot_data"
    _atomic_numeric_csv(
        plot_data / "ramachandran_fes_grid.csv.gz",
        columns={
            "phi_left": np.broadcast_to(phi_edges[:-1, None], phi_grid.shape),
            "phi_right": np.broadcast_to(phi_edges[1:, None], phi_grid.shape),
            "phi_center": phi_grid,
            "psi_left": np.broadcast_to(psi_edges[None, :-1], psi_grid.shape),
            "psi_right": np.broadcast_to(psi_edges[None, 1:], psi_grid.shape),
            "psi_center": psi_grid,
            "reference_density": reference_density,
            "proposal_density": proposal_density,
            "reweighted_density": weighted_density,
            "reference_free_energy_kcal_mol": reference_fes,
            "proposal_free_energy_kcal_mol": proposal_fes,
            "reweighted_free_energy_kcal_mol": weighted_fes,
            "reference_display_free_energy_kcal_mol": np.minimum(
                reference_fes,
                5.25,
            ),
            "proposal_display_free_energy_kcal_mol": np.minimum(
                proposal_fes,
                5.25,
            ),
            "reweighted_display_free_energy_kcal_mol": np.minimum(
                weighted_fes,
                5.25,
            ),
        },
    )

    angle_edges = np.linspace(-np.pi, np.pi, 101)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    fes_grid = np.linspace(-np.pi, np.pi, 200)
    for index, name in enumerate(("phi", "psi")):
        def histogram(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
            return np.histogram(
                values,
                bins=angle_edges,
                density=True,
                weights=weights,
            )[0]

        _atomic_numeric_csv(
            plot_data / f"{name}_marginal_density.csv",
            columns={
                "bin_left": angle_edges[:-1],
                "bin_right": angle_edges[1:],
                "bin_center": angle_centers,
                "reference_density": histogram(
                    reference.dihedrals[:, index],
                ),
                "implicit_reference_density": histogram(
                    implicit_data.dihedrals[:, index],
                ),
                "proposal_density": histogram(
                    proposal.dihedrals[:, index],
                ),
                "reweighted_density": histogram(
                    proposal.dihedrals[:, index],
                    proposal.weights,
                ),
            },
        )
        _atomic_numeric_csv(
            plot_data / f"{name}_free_energy.csv",
            columns={
                name: fes_grid,
                "reference_free_energy_kj_mol": ala2_eval._free_energy_1d(
                    reference.dihedrals[:, index],
                    fes_grid,
                    kT,
                    None,
                ),
                "implicit_reference_free_energy_kj_mol": (
                    ala2_eval._free_energy_1d(
                        implicit_data.dihedrals[:, index],
                        fes_grid,
                        kT,
                        None,
                    )
                ),
                "proposal_free_energy_kj_mol": ala2_eval._free_energy_1d(
                    proposal.dihedrals[:, index],
                    fes_grid,
                    kT,
                    None,
                ),
                "reweighted_free_energy_kj_mol": ala2_eval._free_energy_1d(
                    proposal.dihedrals[:, index],
                    fes_grid,
                    kT,
                    proposal.weights,
                ),
            },
        )

    energy_view: dict[str, Any] | None = None
    if reference.energies is not None and proposal.energies is not None:
        lower, upper = ala2_eval._energy_plot_window(reference.energies)
        energy_edges = np.linspace(lower, upper, 121)
        energy_columns: dict[str, Any] = {
            "bin_left": energy_edges[:-1],
            "bin_right": energy_edges[1:],
            "bin_center": 0.5 * (energy_edges[:-1] + energy_edges[1:]),
            "reference_density": ala2_eval._energy_density_in_view(
                reference.energies,
                energy_edges,
            ),
            "proposal_density": ala2_eval._energy_density_in_view(
                proposal.energies,
                energy_edges,
            ),
            "reweighted_density": ala2_eval._energy_density_in_view(
                proposal.energies,
                energy_edges,
                weights=proposal.weights,
            ),
        }
        implicit_outside: float | None = None
        if implicit_data.energies is not None:
            energy_columns["implicit_reference_density"] = (
                ala2_eval._energy_density_in_view(
                    implicit_data.energies,
                    energy_edges,
                )
            )
            implicit_outside = ala2_eval._energy_outside_fraction(
                implicit_data.energies,
                lower=lower,
                upper=upper,
            )
        _atomic_numeric_csv(
            plot_data / "energy_distribution.csv",
            columns=energy_columns,
        )
        energy_view = {
            "display_only": True,
            "limits": [lower, upper],
            "lower_reference_quantile": FORMAL_ENERGY_LOWER_QUANTILE,
            "upper_reference_quantile": FORMAL_ENERGY_UPPER_QUANTILE,
            "reference_outside_fraction": ala2_eval._energy_outside_fraction(
                reference.energies,
                lower=lower,
                upper=upper,
            ),
            "implicit_reference_outside_fraction": implicit_outside,
            "proposal_outside_fraction": ala2_eval._energy_outside_fraction(
                proposal.energies,
                lower=lower,
                upper=upper,
            ),
            "reweighted_outside_fraction": ala2_eval._energy_outside_fraction(
                proposal.energies,
                lower=lower,
                upper=upper,
                weights=proposal.weights,
            ),
        }

    _atomic_json(
        plot_data / "plot_data_manifest.json",
        {
            "system": "ala2_cg",
            "variant": "ala2_cb",
            "formal": clip_percentile is None,
            "clip_percentile": clip_percentile,
            "clip_mode": None if clip_percentile is None else clip_mode,
            "weight_source": proposal.weight_source,
            "palette": RELEASE_PALETTE,
            "fill_style": RELEASE_FILL_STYLE,
            "kT_kj_mol": kT,
            "rama_bins": 100,
            "marginal_density_bins": 100,
            "marginal_free_energy_grid_points": 200,
            "energy_view": energy_view,
            "explicit_reference_role": "structural_plotting_reference",
            "all_derived_arrays_are_plot_data_not_raw_samples": True,
        },
    )


def _pooled_mb2d_plot_mapping(
    archives: Mapping[str, Path | Mapping[str, Any]],
    *,
    clip_percentile: float | None,
    clip_mode: str,
) -> dict[str, np.ndarray]:
    from cg_bms_jax.evaluation import mb2d as mb2d_eval
    from cg_bms_jax.evaluation.metrics import normalized_weights

    coordinates: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for source in archives.values():
        data = mb2d_eval._load(source)
        block = mb2d_eval.extract_mb2d_coordinates(data["R"])
        block_weights, _, _ = mb2d_eval._resolve_mb2d_weights(
            data,
            kT=1.0,
            clip_percentile=clip_percentile,
            clip_mode=clip_mode,
        )
        if block_weights is None:
            raise ValueError("pooled plot data requires selected weights")
        coordinates.append(block)
        weights.append(normalized_weights(block_weights))
    count = len(coordinates)
    return {
        "R": np.concatenate(coordinates, axis=0),
        "weights": np.concatenate(
            [block / count for block in weights],
            axis=0,
        ),
    }


def _evaluate_all(
    *,
    args: argparse.Namespace,
    stage: Path,
    inputs: Sequence[ArmInput],
    provenance: Mapping[
        str,
        tuple[CheckpointProvenance, CheckpointProvenance, PFProvenance],
    ],
) -> list[dict[str, Any]]:
    # Import only after preflight so ``--help`` and path validation remain
    # usable without importing JAX/Matplotlib.
    from cg_bms_jax.evaluation import (
        evaluate_ala2,
        evaluate_mb,
        evaluate_mb2d,
        evaluate_pooled_mb2d,
    )

    mb_reference = _absolute(args.mb1d_reference)
    ala2_reference = _absolute(args.ala2_reference)
    ala2_implicit = _absolute(args.ala2_implicit_reference)
    for reference in (mb_reference, ala2_reference, ala2_implicit):
        if not reference.is_file():
            raise FileNotFoundError(f"evaluation reference is missing: {reference}")

    rows: list[dict[str, Any]] = []
    mb2d_groups: dict[str, dict[str, Path]] = {}
    mb2d_clip_groups: dict[str, dict[str, Mapping[str, Any]]] = {}
    for arm in inputs:
        forward, backward, pf = provenance[arm.key]
        base_figure_dir = (
            stage
            / "figures"
            / _release_family(arm.system)
            / arm.arm
            / f"seed_{arm.seed}"
        )
        figure_dir = base_figure_dir / "formal_no_clip"
        clip_figure_dir = base_figure_dir / "diagnostic_drop_top_1_percent"
        diagnostic_sample = _diagnostic_clip_mapping(arm.pf_archive)
        if arm.system == "mb_cg1d":
            evaluation = evaluate_mb(
                target=mb_reference,
                sample=arm.pf_archive,
                output_dir=figure_dir,
                kT=1.0,
                clip_percentile=None,
                n_bootstraps=args.n_bootstraps,
                seed=arm.seed,
            )
            clip_evaluation = evaluate_mb(
                target=mb_reference,
                sample=diagnostic_sample,
                output_dir=clip_figure_dir,
                kT=1.0,
                clip_percentile=99.0,
                clip_mode="drop",
                n_bootstraps=args.n_bootstraps,
                seed=arm.seed,
            )
            _write_mb_plot_data(
                target=mb_reference,
                sample=arm.pf_archive,
                output_dir=figure_dir,
                kT=1.0,
                clip_percentile=None,
                clip_mode="drop",
            )
            _write_mb_plot_data(
                target=mb_reference,
                sample=diagnostic_sample,
                output_dir=clip_figure_dir,
                kT=1.0,
                clip_percentile=99.0,
                clip_mode="drop",
            )
        elif arm.system == "mb2d":
            evaluation = evaluate_mb2d(
                sample=arm.pf_archive,
                output_dir=figure_dir,
                kT=1.0,
                bins=args.mb2d_bins,
                energy_lower_quantile=FORMAL_ENERGY_LOWER_QUANTILE,
                energy_upper_quantile=FORMAL_ENERGY_UPPER_QUANTILE,
                energy_view_padding=FORMAL_ENERGY_VIEW_PADDING,
                energy_histogram_bins=args.energy_bins,
                clip_percentile=None,
            )
            mb2d_groups.setdefault(arm.arm, {})[
                f"seed{arm.seed}"
            ] = arm.pf_archive
            clip_evaluation = evaluate_mb2d(
                sample=diagnostic_sample,
                output_dir=clip_figure_dir,
                kT=1.0,
                bins=args.mb2d_bins,
                energy_lower_quantile=FORMAL_ENERGY_LOWER_QUANTILE,
                energy_upper_quantile=FORMAL_ENERGY_UPPER_QUANTILE,
                energy_view_padding=FORMAL_ENERGY_VIEW_PADDING,
                energy_histogram_bins=args.energy_bins,
                clip_percentile=99.0,
                clip_mode="drop",
                weight_view_label=(
                    "Diagnostic only: drop top 1% finite raw importance weights"
                ),
            )
            mb2d_clip_groups.setdefault(arm.arm, {})[
                f"seed{arm.seed}"
            ] = diagnostic_sample
            _write_mb2d_plot_data(
                sample=arm.pf_archive,
                output_dir=figure_dir,
                bins=args.mb2d_bins,
                energy_bins_count=args.energy_bins,
                clip_percentile=None,
                clip_mode="drop",
            )
            _write_mb2d_plot_data(
                sample=diagnostic_sample,
                output_dir=clip_figure_dir,
                bins=args.mb2d_bins,
                energy_bins_count=args.energy_bins,
                clip_percentile=99.0,
                clip_mode="drop",
            )
        elif arm.system == "ala2_cg":
            evaluation = evaluate_ala2(
                target=ala2_reference,
                sample=arm.pf_archive,
                output_dir=figure_dir,
                variant="ala2_cb",
                implicit=ala2_implicit,
                kT=args.ala2_kT,
                clip_percentile=None,
                n_bootstraps=args.n_bootstraps,
                seed=arm.seed,
            )
            _write_ala2_plot_data(
                target=ala2_reference,
                sample=arm.pf_archive,
                implicit=ala2_implicit,
                output_dir=figure_dir,
                kT=args.ala2_kT,
                clip_percentile=None,
                clip_mode="drop",
            )
            _write_ala2_plot_data(
                target=ala2_reference,
                sample=diagnostic_sample,
                implicit=ala2_implicit,
                output_dir=clip_figure_dir,
                kT=args.ala2_kT,
                clip_percentile=99.0,
                clip_mode="drop",
            )
            clip_evaluation = evaluate_ala2(
                target=ala2_reference,
                sample=diagnostic_sample,
                output_dir=clip_figure_dir,
                variant="ala2_cb",
                implicit=ala2_implicit,
                kT=args.ala2_kT,
                clip_percentile=99.0,
                clip_mode="drop",
                n_bootstraps=args.n_bootstraps,
                seed=arm.seed,
            )
        else:
            raise AssertionError(f"unknown system {arm.system!r}")
        metrics = evaluation["metrics"]
        metrics_path = (
            stage
            / "metrics"
            / _release_family(arm.system)
            / arm.arm
            / "formal_no_clip"
            / f"seed_{arm.seed}.json"
        )
        _atomic_json(
            metrics_path,
            {
                "estimator": {
                    "formal": True,
                    "clip": None,
                    "density_mode": "ambient_exact",
                    "raw_formula": (
                        "-target_reduced_energy-logq_ambient"
                    ),
                },
                "metrics": metrics,
            },
        )
        _atomic_json(
            stage
            / "metrics"
            / _release_family(arm.system)
            / arm.arm
            / "diagnostic_drop_top_1_percent"
            / f"seed_{arm.seed}.json",
            {
                "estimator": {
                    "formal": False,
                    "diagnostic_only": True,
                    "clip_percentile": 99.0,
                    "clip_mode": "drop",
                    "description": (
                        "Drop the top 1% of finite raw importance weights, then "
                        "renormalize; never used in headline tables."
                    ),
                },
                "metrics": clip_evaluation["metrics"],
            },
        )
        _write_weight_tail_panel(
            archive=arm.pf_archive,
            output_dir=figure_dir,
            arm=arm,
            audit=pf.no_clip_audit,
        )
        row = _result_row(arm, metrics, pf, forward, backward)
        rows.append(row)
        _write_provenance(
            stage=stage,
            arm=arm,
            forward=forward,
            backward=backward,
            pf=pf,
        )

    expected_mb2d_arms = {
        "cold_energy_only",
        "equilibrium_bridge_only",
        "equilibrium_bridge_energy",
        "biased_bridge_only",
        "biased_bridge_energy",
    }
    if set(mb2d_groups) != expected_mb2d_arms:
        raise ValueError(
            "MB2D release matrix is incomplete: "
            f"expected {sorted(expected_mb2d_arms)}, "
            f"found {sorted(mb2d_groups)}"
        )
    for arm, archives in sorted(mb2d_groups.items()):
        if len(archives) != 3:
            raise ValueError(f"MB2D arm {arm} does not contain three seeds")
        pooled_dir = (
            stage
            / "figures"
            / "mb2d_analytic"
            / arm
            / "pooled_3seed"
            / "formal_no_clip"
        )
        pooled = evaluate_pooled_mb2d(
            archives,
            output_dir=pooled_dir,
            kT=1.0,
            bins=args.mb2d_pooled_bins,
            energy_lower_quantile=FORMAL_ENERGY_LOWER_QUANTILE,
            energy_upper_quantile=FORMAL_ENERGY_UPPER_QUANTILE,
            energy_view_padding=FORMAL_ENERGY_VIEW_PADDING,
            energy_histogram_bins=args.energy_bins,
            clip_percentile=None,
        )
        _atomic_json(
            stage
            / "metrics"
            / "mb2d_analytic"
            / arm
            / "formal_no_clip"
            / "pooled_3seed.json",
            {
                "estimator": {
                    "formal": True,
                    "clip": None,
                    "pooling": (
                        "equal seed mass for visualization; per-seed metrics "
                        "remain authoritative"
                    ),
                },
                "metrics": pooled["metrics"],
            },
        )
        pooled_clip = evaluate_pooled_mb2d(
            mb2d_clip_groups[arm],
            output_dir=(
                stage
                / "figures"
                / "mb2d_analytic"
                / arm
                / "pooled_3seed"
                / "diagnostic_drop_top_1_percent"
            ),
            kT=1.0,
            bins=args.mb2d_pooled_bins,
            energy_lower_quantile=FORMAL_ENERGY_LOWER_QUANTILE,
            energy_upper_quantile=FORMAL_ENERGY_UPPER_QUANTILE,
            energy_view_padding=FORMAL_ENERGY_VIEW_PADDING,
            energy_histogram_bins=args.energy_bins,
            clip_percentile=99.0,
            clip_mode="drop",
        )
        _write_mb2d_plot_data(
            sample=_pooled_mb2d_plot_mapping(
                archives,
                clip_percentile=None,
                clip_mode="drop",
            ),
            output_dir=pooled_dir,
            bins=args.mb2d_pooled_bins,
            energy_bins_count=args.energy_bins,
            clip_percentile=None,
            clip_mode="drop",
        )
        clip_pooled_dir = (
            stage
            / "figures"
            / "mb2d_analytic"
            / arm
            / "pooled_3seed"
            / "diagnostic_drop_top_1_percent"
        )
        _write_mb2d_plot_data(
            sample=_pooled_mb2d_plot_mapping(
                mb2d_clip_groups[arm],
                clip_percentile=99.0,
                clip_mode="drop",
            ),
            output_dir=clip_pooled_dir,
            bins=args.mb2d_pooled_bins,
            energy_bins_count=args.energy_bins,
            clip_percentile=None,
            clip_mode="drop",
            weight_view="diagnostic_drop_top_1_percent",
        )
        _atomic_json(
            stage
            / "metrics"
            / "mb2d_analytic"
            / arm
            / "diagnostic_drop_top_1_percent"
            / "pooled_3seed.json",
            {
                "estimator": {
                    "formal": False,
                    "diagnostic_only": True,
                    "clip_percentile": 99.0,
                    "clip_mode": "drop",
                    "pooling": "equal seed mass for visualization",
                },
                "metrics": pooled_clip["metrics"],
            },
        )
    return rows


def _validate_generated_release_payload(
    *,
    stage: Path,
    inputs: Sequence[ArmInput],
) -> dict[str, Any]:
    """Verify every required standalone image and its machine-readable data."""

    common: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "mb_cg1d": (
            ("mb_plots.png", "mb_density.png", "mb_free_energy.png"),
            (
                "density_histogram.csv",
                "density_and_free_energy.csv",
                "plot_data_manifest.json",
            ),
        ),
        "mb2d": (
            (
                "mb2d_plots.png",
                "mb2d_exact_fes.png",
                "mb2d_proposal_fes.png",
                "mb2d_reweighted_fes.png",
                "mb2d_x_marginal.png",
                "mb2d_y_marginal.png",
                "mb2d_energy_distribution.png",
            ),
            (
                "fes_grid.csv.gz",
                "x_marginal.csv",
                "y_marginal.csv",
                "energy_distribution.csv",
                "plot_data_manifest.json",
            ),
        ),
        "ala2_cg": (
            (
                "ala2_cb_energy_distribution.png",
                "ala2_cb_density.png",
                "ala2_cb_phi_density.png",
                "ala2_cb_psi_density.png",
                "ala2_cb_free_energy.png",
                "ala2_cb_phi_free_energy.png",
                "ala2_cb_psi_free_energy.png",
                "ala2_cb_ramachandran_fes.png",
                "ala2_cb_reference_ramachandran_fes.png",
                "ala2_cb_proposal_ramachandran_fes.png",
                "ala2_cb_reweighted_ramachandran_fes.png",
            ),
            (
                "ramachandran_fes_grid.csv.gz",
                "phi_marginal_density.csv",
                "phi_free_energy.csv",
                "psi_marginal_density.csv",
                "psi_free_energy.csv",
                "energy_distribution.csv",
                "plot_data_manifest.json",
            ),
        ),
    }
    checked: list[str] = []

    def check(path: Path) -> None:
        if not path.is_file() or path.stat().st_size < 16:
            raise RuntimeError(f"required frozen image/data payload is missing: {path}")
        header = path.read_bytes()[:8]
        if path.suffix == ".png" and header != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"invalid PNG payload: {path}")
        if path.suffix == ".pdf" and not header.startswith(b"%PDF-"):
            raise RuntimeError(f"invalid PDF payload: {path}")
        if path.name.endswith(".csv.gz") and header[:2] != b"\x1f\x8b":
            raise RuntimeError(f"invalid gzip CSV payload: {path}")
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "plot_data_manifest.json":
                style_contract = payload.get("visual_contract", payload)
                if style_contract.get("palette") != RELEASE_PALETTE:
                    raise RuntimeError(
                        f"plot-data palette drifted from release contract: {path}"
                    )
                if style_contract.get("fill_style") != RELEASE_FILL_STYLE:
                    raise RuntimeError(
                        f"plot-data fill style drifted from release contract: {path}"
                    )
        checked.append(path.relative_to(stage).as_posix())

    for arm in inputs:
        base = (
            stage
            / "figures"
            / _release_family(arm.system)
            / arm.arm
            / f"seed_{arm.seed}"
        )
        images, plot_files = common[arm.system]
        for view in ("formal_no_clip", "diagnostic_drop_top_1_percent"):
            directory = base / view
            for name in images:
                check(directory / name)
            for name in plot_files:
                check(directory / "plot_data" / name)
        formal = base / "formal_no_clip"
        for name in (
            "weight_tail_diagnostic.png",
            "weight_tail_diagnostic.pdf",
        ):
            check(formal / name)
        for name in ("weight_tail_curve.csv", "weight_tail_summary.json"):
            check(formal / "plot_data" / name)

    mb2d_arms = sorted({arm.arm for arm in inputs if arm.system == "mb2d"})
    mb2d_images, mb2d_data = common["mb2d"]
    for arm in mb2d_arms:
        base = stage / "figures" / "mb2d_analytic" / arm / "pooled_3seed"
        for view in ("formal_no_clip", "diagnostic_drop_top_1_percent"):
            directory = base / view
            for name in mb2d_images:
                check(directory / name)
            for name in mb2d_data:
                check(directory / "plot_data" / name)

    return {
        "verification": "PASS",
        "required_files_checked": len(checked),
        "files": checked,
        "visual_contract": {
            "palette": RELEASE_PALETTE,
            "fill_style": RELEASE_FILL_STYLE,
            "mb2d_fes_colormap": MB2D_FES_COLORMAP,
            "mb2d_fes_vmin": MB2D_FES_VMIN,
            "mb2d_fes_vmax_kT": MB2D_FES_VMAX_KT,
            "mb2d_physical_limits": MB2D_PHYSICAL_LIMITS,
        },
    }


def _summary_json(
    *,
    rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
    project_root: Path,
    git_provenance: Mapping[str, Any],
    input_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        key = f"{row['system']}/{row['arm']}"
        group = groups.setdefault(
            key,
            {
                "system": row["system"],
                "arm": row["arm"],
                "release_role": row["release_role"],
                "data_identity": row["data_identity"],
                "seeds": row["seeds"],
                "metrics": {},
            },
        )
        group["metrics"][str(row["metric"])] = {
            field: row[field]
            for field in ("n", "mean", "std", "minimum", "maximum")
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "release_name": DEFAULT_RELEASE_NAME,
        "scientific_claim": (
            "This freeze evaluates formal PF-ODE importance reweighting for "
            "Energy-BMS, bridge, and bridge-to-Energy generators. Where "
            "proposal overlap and ESS permit, it corrects residual proposal "
            "bias; low-overlap/low-ESS outcomes are retained as limitations."
        ),
        "formal_weighting": {
            "no_clip": True,
            "raw_formula": "-target_reduced_energy-logq_ambient",
            "definition": (
                "softmax(recomputed raw formula) over finite valid/support "
                "samples; invalid/out-of-support samples have zero weight"
            ),
            "clip_results_in_headline_tables": False,
            "diagnostic_clip1": {
                "included": True,
                "formal": False,
                "mode": "drop",
                "percentile": 99.0,
                "description": "drop top 1% finite raw weights and renormalize",
            },
        },
        "plotting": {
            "energy_limits_are_display_only": True,
            "all_generated_samples_accounted_for": True,
            "nonzero_weight_domain": "finite valid/support samples only",
            "machine_readable_panel_data": (
                "figures/**/plot_data/*.csv and plot_data_manifest.json"
            ),
            "visual_contract": input_provenance.get("visual_contract"),
            "lower_reference_quantile": FORMAL_ENERGY_LOWER_QUANTILE,
            "upper_reference_quantile": FORMAL_ENERGY_UPPER_QUANTILE,
            "padding_fraction": FORMAL_ENERGY_VIEW_PADDING,
        },
        "scope": {
            "systems": ["mb_cg1d", "mb2d_analytic", "ala2_cg"],
            "all_atom_ala2": "explicitly_excluded",
            "mb2d_biased_endpoints": (
                "legacy deliberately biased synthetic full-support stress test; "
                "not MD and not equilibrium data"
            ),
            "ala2_target": (
                "six-bead core-beta CG-BG MACE PMF + fixed support terms, "
                "300 K, ambient18 auxiliary-COM; warm endpoints hash-pinned "
                "CG-BG explicit core-beta data"
            ),
            "ala2_reference_caveat": (
                "the explicit flow_ub asset is a structural plotting reference, "
                "not asserted to be an exact equilibrium sample of the complete "
                "PMF-plus-support formal target"
            ),
        },
        "num_result_rows": len(rows),
        "groups": groups,
        "copied_configs": list(configs),
        "input_provenance": input_provenance,
        "code": {
            "project_root": str(project_root),
            "git": dict(git_provenance),
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
            "python": sys.version,
        },
    }


def _readme(
    rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> str:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in summary_rows:
        lookup[(str(row["system"]), str(row["arm"]), str(row["metric"]))] = float(
            row["mean"]
        )
    groups = sorted({(str(row["system"]), str(row["arm"])) for row in rows})
    table = [
        "| system | arm | role | data identity | N seeds | proposal JS | reweighted JS | ESS/N |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for system, arm in groups:
        group_rows = [
            row
            for row in rows
            if row["system"] == system and row["arm"] == arm
        ]
        table.append(
            "| "
            + " | ".join(
                (
                    system,
                    arm,
                    str(group_rows[0]["release_role"]),
                    str(group_rows[0]["data_identity"]),
                    str(len(group_rows)),
                    f"{lookup[(system, arm, 'proposal_js')]:.6g}",
                    f"{lookup[(system, arm, 'weighted_js')]:.6g}",
                    f"{lookup[(system, arm, 'ess_fraction')]:.6g}",
                )
            )
            + " |"
        )
    return "\n".join(
        (
            "# cg-bms-jax frozen result bundle",
            "",
            "This is a lightweight, reproducible result freeze. Large PF archives "
            "and checkpoints are **not copied**; their independently verified "
            "SHA-256 digests, manifests, array inventories and PF metadata are "
            "stored under `provenance/`.",
            "",
            "## Scientific scope",
            "",
            "- MB CG1D: three-seed cold Energy-BMS plus formal reweighting.",
            "- MB2D: three-seed cold Energy-BMS, exact-equilibrium bridge, and "
            "exact-equilibrium bridge-to-Energy.",
            "- MB2D legacy: bridge and bridge-to-Energy on deliberately biased "
            "synthetic full-support endpoints. These are stress tests; they are "
            "**not MD** and **not equilibrium endpoint** experiments.",
            "- CG Ala2: the strict release-tolerance W10k-to-E20k-to-B20k PF10k "
            "result and the in-root W10k-to-E100k-to-B100k PF20k mixed/low-ESS "
            "result.",
            "- CG Ala2 target: six-bead core-beta CG-BG MACE PMF plus fixed "
            "support terms at 300 K in ambient18 auxiliary-COM coordinates; "
            "warm endpoints are SHA-pinned CG-BG explicit core-beta data.",
            "- The CG Ala2 explicit `flow_ub` asset is a structural plotting "
            "reference. It is not claimed to be an exact equilibrium sample "
            "from the full PMF-plus-support formal target.",
            "- All-atom Ala2 is deliberately excluded.",
            "",
            "## Formal estimator and plotting",
            "",
            "Every stored result passed an explicit audit that `logw_raw` equals "
            "`-target_reduced_energy-logq_ambient` and that `weights` equal the "
            "softmax of that recomputed formula over finite valid/support "
            "samples. Invalid/out-of-support samples retain zero formal weight. "
            "No clipping, capping or PSIS replacement is part of the formal "
            "result.",
            "",
            "Energy plots use the reference 0.5%--99.5% energy-quantile window "
            "plus 5% padding only to avoid an unreadable x axis. All generated "
            "samples are accounted for; only finite valid/support samples have "
            "nonzero formal weight. Omitted display mass is recorded in each "
            "evaluator metric JSON.",
            "",
            "Each PF arm also has a standalone formal no-clip cumulative "
            "weight-tail panel annotated with ESS/N, maximum weight, top-tail "
            "mass, and raw-log-weight variance/span. A separately labelled "
            "`diagnostic_drop_top_1_percent/` view drops the top 1% of finite "
            "raw weights and renormalizes. It is diagnostic only and is never "
            "used in the headline benchmark tables.",
            "",
            "All MB panels obey `VISUAL_CONTRACT.json`: exact/true is black, "
            "a distinct reference is green `#8FD18B`, proposal is orange "
            "`#F2A174`, and reweighted uses a light-blue `#B9CBEA` fill with "
            "a blue `#4472C4` outline. Reference/proposal/reweighted one-"
            "dimensional marginals and energy distributions are translucent "
            "filled densities with crisp outlines; exact/true remains a black "
            "line. Every MB2D FES uses "
            "the same viridis map, physical [0,50] x [0,50] extent, and fixed "
            "0--12 kT color scale labelled `Free energy`. CG Ala2 reference, "
            "proposal and reweighted Ramachandran maps are saved as separate "
            "standalone panels without overlays.",
            "",
            "## Benchmark",
            "",
            *table,
            "",
            "Per-run authoritative values are in `per_seed.csv` and "
            "`per_seed.json`; aggregate mean/std/min/max values are in "
            "`summary.csv` and `summary.json`. Standalone panels are below "
            "`figures/`; formal and diagnostic estimators are in separate "
            "subdirectories. Every panel directory contains `plot_data/` CSV/JSON "
            "with the exact derived grid, histogram, marginal, energy-window or "
            "weight-tail values used by that panel. `SOURCE_RESULTS_INDEX.csv` "
            "and `.json` identify every NAS checkpoint, PF archive, reference, "
            "PMF and endpoint dataset by path, byte size and SHA-256; those "
            "large sources are not copied into Git. `SHA256SUMS` covers every "
            "payload file except itself "
            "and the subsequently generated self-describing `MANIFEST.json`.",
            "",
        )
    )


def _write_sha256sums(stage: Path) -> None:
    rows: list[str] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "MANIFEST.json"}:
            continue
        rows.append(f"{_sha256_file(path)}  {path.relative_to(stage).as_posix()}")
    _atomic_text(stage / "SHA256SUMS", "\n".join(rows) + "\n")


def _semantic_role(relative: Path) -> str:
    if relative.name in {"SOURCE_RESULTS_INDEX.csv", "SOURCE_RESULTS_INDEX.json"}:
        return "external_large_source_results_index"
    if relative.name == "VISUAL_CONTRACT.json":
        return "cross_arm_palette_and_scale_contract"
    if relative.name == "IMAGE_DATA_AUDIT.json":
        return "standalone_image_and_plot_data_completeness_audit"
    if relative.parts[0] == "figures":
        if "plot_data" in relative.parts:
            if "diagnostic_drop_top_1_percent" in relative.parts:
                return "machine_readable_nonformal_clip1_panel_data"
            return "machine_readable_standalone_panel_data"
        if "diagnostic_drop_top_1_percent" in relative.parts:
            return "explicit_nonformal_clip1_diagnostic_figure"
        if relative.name.startswith("weight_tail_diagnostic"):
            return "formal_no_clip_weight_concentration_figure"
        return "standalone_or_combined_figure"
    if relative.parts[0] == "metrics":
        if "diagnostic_drop_top_1_percent" in relative.parts:
            return "explicit_nonformal_clip1_diagnostic_metrics"
        return "formal_no_clip_metrics"
    if relative.parts[0] == "provenance":
        return "external_input_provenance"
    if relative.parts[0] == "configs":
        return "frozen_configuration"
    if relative.name == "per_seed.csv" or relative.name == "per_seed.json":
        return "per_seed_benchmark"
    if relative.name == "summary.csv" or relative.name == "summary.json":
        return "aggregate_benchmark"
    if relative.name == "SHA256SUMS":
        return "payload_checksum_index"
    if relative.name == "README.md":
        return "release_documentation"
    if relative.name == "release_validation.json":
        return "release_scope_validation_contract"
    return "release_payload"


def _write_manifest(stage: Path) -> None:
    files = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        relative = path.relative_to(stage)
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
                "semantic_role": _semantic_role(relative),
            }
        )
    _atomic_json(
        stage / "MANIFEST.json",
        {
            "schema_version": SCHEMA_VERSION,
            "release_name": DEFAULT_RELEASE_NAME,
            "release_scope": {
                "included_benchmark_families": [
                    "mb_cg1d",
                    "mb2d_analytic",
                    "ala2_cg",
                ],
                "excluded_benchmark_families": ["ala2_all_atom"],
            },
            "files": files,
        },
    )


def _validate_stage_scope(*, project_root: Path, stage: Path) -> None:
    validator = project_root / "scripts" / "validate_release_scope.py"
    completed = subprocess.run(
        (sys.executable, str(validator), str(stage)),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "frozen bundle failed release-scope validation:\n"
            + completed.stdout
            + completed.stderr
        )


def build_release(args: argparse.Namespace) -> Path:
    canonical_mismatches = _canonical_release_mismatches(args)
    if canonical_mismatches:
        raise ValueError(
            "v0.1 is an immutable canonical benchmark:\n  "
            + "\n  ".join(canonical_mismatches)
        )
    project_root = _absolute(args.project_root)
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root is missing: {project_root}")
    git_provenance = _git_provenance(project_root)
    if not git_provenance.get("commit"):
        raise RuntimeError(
            "canonical v0.1 must be built from a committed git revision"
        )
    if git_provenance.get("dirty") is not False:
        raise RuntimeError(
            "canonical v0.1 must start from a clean git worktree; commit the "
            "release code/config snapshot before building"
        )
    inputs = _arm_inputs(args)
    # This intentionally performs all expensive integrity/no-clip checks before
    # the first release output is created.
    provenance = _preflight(inputs)
    input_provenance, input_snapshots = _capture_input_provenance(
        args=args,
        project_root=project_root,
        inputs=inputs,
    )
    visual_contract = _validate_visual_contract_sources(project_root)
    input_provenance["visual_contract"] = visual_contract

    output = _absolute(args.output_dir)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing frozen release: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.tmp-{secrets.token_hex(8)}"
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir()
    try:
        configs = _copy_release_configs(
            stage=stage,
            inputs=inputs,
            project_root=project_root,
        )
        rows = _evaluate_all(
            args=args,
            stage=stage,
            inputs=inputs,
            provenance=provenance,
        )
        image_data_audit = _validate_generated_release_payload(
            stage=stage,
            inputs=inputs,
        )
        _atomic_json(stage / "IMAGE_DATA_AUDIT.json", image_data_audit)
        _verify_input_snapshots(input_snapshots)
        _verify_large_inputs_unchanged(
            inputs=inputs,
            provenance=provenance,
        )
        _atomic_json(
            stage / "provenance" / "input_assets_and_code.json",
            input_provenance,
        )
        _atomic_json(stage / "VISUAL_CONTRACT.json", visual_contract)
        source_index = _source_results_index(
            inputs=inputs,
            provenance=provenance,
            input_provenance=input_provenance,
        )
        _atomic_csv(stage / "SOURCE_RESULTS_INDEX.csv", source_index)
        _atomic_json(
            stage / "SOURCE_RESULTS_INDEX.json",
            {
                "schema_version": SCHEMA_VERSION,
                "large_sources_copied": False,
                "rows": source_index,
            },
        )
        rows = sorted(
            rows,
            key=lambda row: (
                str(row["system"]),
                str(row["arm"]),
                int(row["seed"]),
            ),
        )
        summary_rows = _summary_rows(rows)
        _atomic_csv(stage / "per_seed.csv", rows)
        _atomic_json(
            stage / "per_seed.json",
            {
                "schema_version": SCHEMA_VERSION,
                "formal_no_clip": True,
                "rows": rows,
            },
        )
        _atomic_csv(stage / "summary.csv", summary_rows)
        _atomic_json(
            stage / "summary.json",
            _summary_json(
                rows=rows,
                summary_rows=summary_rows,
                configs=configs,
                project_root=project_root,
                git_provenance=git_provenance,
                input_provenance=input_provenance,
            ),
        )
        _atomic_text(stage / "README.md", _readme(rows, summary_rows))
        _atomic_json(
            stage / "release_validation.json",
            {
                "validator": "scripts/validate_release_scope.py",
                "validator_sha256": _sha256_file(
                    project_root / "scripts" / "validate_release_scope.py"
                ),
                "validated_before_atomic_publish": True,
                "required_families": [
                    "mb_cg1d",
                    "mb2d_analytic",
                    "ala2_cg",
                ],
                "required_exclusion": "ala2_all_atom",
            },
        )
        _write_sha256sums(stage)
        _write_manifest(stage)
        _validate_stage_scope(project_root=project_root, stage=stage)
        os.replace(stage, output)
        if os.name == "posix":
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return output
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and freeze the lightweight, CG-only cg-bms-jax result release."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mb1d-root", type=Path)
    parser.add_argument("--mb2d-legacy-root", type=Path)
    parser.add_argument("--mb2d-equilibrium-root", type=Path)
    parser.add_argument("--ala2-root", type=Path)
    parser.add_argument("--ala2-positive-root", type=Path)
    parser.add_argument("--ala2-positive-training-root", type=Path)
    parser.add_argument("--ala2-mixed-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--mb-samples", type=int, default=100_000)
    parser.add_argument("--ala2-positive-samples", type=int, default=10_000)
    parser.add_argument("--ala2-mixed-samples", type=int, default=20_000)
    parser.add_argument("--n-bootstraps", type=int, default=500)
    parser.add_argument("--mb2d-bins", type=int, default=160)
    parser.add_argument("--mb2d-pooled-bins", type=int, default=160)
    parser.add_argument("--energy-bins", type=int, default=240)
    parser.add_argument(
        "--mb1d-reference",
        type=Path,
    )
    parser.add_argument(
        "--ala2-reference",
        type=Path,
    )
    parser.add_argument(
        "--ala2-implicit-reference",
        type=Path,
    )
    parser.add_argument(
        "--ala2-kT",
        type=float,
        default=2.494338785445972,
    )
    return parser


def _canonical_release_mismatches(
    args: argparse.Namespace,
) -> list[str]:
    """Prevent a v0.1 path from silently naming a noncanonical benchmark."""

    exact = {
        "seeds": [0, 1, 2],
        "mb_samples": 100_000,
        "ala2_positive_samples": 10_000,
        "ala2_mixed_samples": 20_000,
        "n_bootstraps": 500,
        "mb2d_bins": 160,
        "mb2d_pooled_bins": 160,
        "energy_bins": 240,
    }
    mismatches: list[str] = [
        f"{name}={getattr(args, name)!r} (required {expected!r})"
        for name, expected in exact.items()
        if getattr(args, name) != expected
    ]
    if not math.isclose(
        float(args.ala2_kT),
        2.494338785445972,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        mismatches.append(
            f"ala2_kT={args.ala2_kT!r} (required 2.494338785445972)"
        )
    return mismatches


def _validate_canonical_release_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    mismatches = _canonical_release_mismatches(args)
    if mismatches:
        parser.error(
            "v0.1 is an immutable canonical benchmark; noncanonical settings "
            "must use a different release builder/name:\n  "
            + "\n  ".join(mismatches)
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = _absolute(args.project_root)
    if args.output_dir is None:
        args.output_dir = root / "artifacts" / DEFAULT_RELEASE_NAME
    if args.mb1d_reference is None:
        args.mb1d_reference = (
            root / "assets" / "cache" / "MB" / "pmf_b" / "flow_ub" / "data.npz"
        )
    if args.ala2_reference is None:
        args.ala2_reference = (
            root
            / "assets"
            / "cache"
            / "Ac-Ala-NHMe"
            / "explicit"
            / "core_beta"
            / "pmf_b"
            / "flow_ub"
            / "data.npz"
        )
    if args.ala2_implicit_reference is None:
        args.ala2_implicit_reference = (
            root
            / "assets"
            / "cache"
            / "Ac-Ala-NHMe"
            / "implicit"
            / "data.npz"
        )
    for field in (
        "mb_samples",
        "ala2_positive_samples",
        "ala2_mixed_samples",
        "n_bootstraps",
        "mb2d_bins",
        "mb2d_pooled_bins",
        "energy_bins",
    ):
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    _validate_canonical_release_arguments(parser, args)
    output = build_release(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
