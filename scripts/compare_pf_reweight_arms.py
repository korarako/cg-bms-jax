#!/usr/bin/env python3
"""Compare paired Warm and Warm->Energy PF/reweight archives."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cg_bms_jax.reweight.io import validate_reweight_payload

_SHARED_METADATA_KEYS = (
    "experiment_family",
    "event_shape",
    "coordinate_mode",
    "model_signature",
    "sde_signature",
    "pmf_revision",
    "pmf_sha256",
    "training_data_sha256",
    "coordinate_signature",
    "species_signature",
    "standardization_std_nm",
    "density_mode",
    "target_mode",
    "target_implementation_abi",
    "pmf_scale",
    "target_terms",
    "target_signature",
    "formal_target_signature",
    "training_target_signature",
    "topology_signature",
    "domain",
)

_SHARED_LIKELIHOOD_KEYS = (
    "solver",
    "t0",
    "t1",
    "dt0",
    "rtol",
    "atol",
    "max_steps",
    "divergence",
    "dtype",
    "target_batch_size",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _logmeanexp(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    maximum = float(np.max(finite))
    return maximum + math.log(float(np.mean(np.exp(finite - maximum))))


def _weight_mass(weights: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(weights.size * fraction)))
    return float(np.sort(weights)[-count:].sum())


def _summary(sample: dict[str, np.ndarray]) -> dict[str, Any]:
    weights = np.asarray(sample["weights"], dtype=np.float64).reshape(-1)
    if weights.size == 0 or not np.isfinite(weights).all():
        raise ValueError("weights must be a non-empty finite vector")
    total = float(weights.sum())
    if not np.isclose(total, 1.0, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError(f"weights are not normalized: sum={total}")
    ess = float(1.0 / np.sum(np.square(weights)))
    logw_raw = np.asarray(sample["logw_raw"], dtype=np.float64).reshape(-1)
    finite_logw = logw_raw[np.isfinite(logw_raw)]
    valid = np.asarray(sample.get("valid_mask", np.ones(weights.size)), dtype=bool)
    support = np.asarray(sample.get("support_mask", valid), dtype=bool)
    steps = np.asarray(sample.get("ode_steps_per_batch", []), dtype=np.int64)
    return {
        "num_samples": int(weights.size),
        "valid_fraction": float(valid.mean()),
        "support_fraction": float(support.mean()),
        "ess": ess,
        "ess_fraction": ess / weights.size,
        "max_weight": float(weights.max()),
        "top_0.1_percent_weight_mass": _weight_mass(weights, 0.001),
        "top_1_percent_weight_mass": _weight_mass(weights, 0.01),
        "finite_logw_fraction": float(np.isfinite(logw_raw).mean()),
        "logw_raw_variance_finite": (
            float(np.var(finite_logw)) if finite_logw.size else float("nan")
        ),
        "logmeanexp_logw_raw_finite": _logmeanexp(logw_raw),
        "ode_steps": (
            {
                "min": int(steps.min()),
                "median": float(np.median(steps)),
                "p95": float(np.quantile(steps, 0.95)),
                "max": int(steps.max()),
            }
            if steps.size
            else None
        ),
    }


def _validated_metadata(sample: dict[str, np.ndarray]) -> dict[str, Any]:
    metadata = validate_reweight_payload(
        sample,
        expected_density_mode="ambient_exact",
    )
    for field in ("initial_X", "initial_logq", "ode_steps_per_batch"):
        if field not in sample:
            raise KeyError(f"PF archive is missing comparison field {field!r}")
    count = int(np.asarray(sample["weights"]).size)
    if np.asarray(sample["initial_X"]).shape[0] != count:
        raise ValueError("initial_X does not match the number of PF samples")
    if np.asarray(sample["initial_logq"]).reshape(-1).size != count:
        raise ValueError("initial_logq does not match the number of PF samples")
    if metadata.get("sampler_kind") != "pf_ode":
        raise ValueError("metadata_json must identify sampler_kind=pf_ode")
    if metadata.get("density_mode") != "ambient_exact":
        raise ValueError("metadata_json must identify density_mode=ambient_exact")
    return metadata


def _require_same_provenance(
    warm: dict[str, Any], warm_energy: dict[str, Any]
) -> None:
    for key in _SHARED_METADATA_KEYS:
        if key not in warm or key not in warm_energy:
            raise ValueError(f"Both PF archives must record metadata key {key!r}")
        if warm[key] != warm_energy[key]:
            raise ValueError(f"PF archive provenance differs for {key!r}")
    warm_likelihood = warm.get("likelihood")
    warm_energy_likelihood = warm_energy.get("likelihood")
    if not isinstance(warm_likelihood, dict) or not isinstance(
        warm_energy_likelihood, dict
    ):
        raise ValueError("Both PF archives must record likelihood metadata")
    for key in _SHARED_LIKELIHOOD_KEYS:
        if warm_likelihood.get(key) != warm_energy_likelihood.get(key):
            raise ValueError(f"PF likelihood settings differ for {key!r}")
    if warm.get("seed") != warm_energy.get("seed"):
        raise ValueError("PF archives must use the same seed for paired comparison")


def _metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--warm-energy", type=Path, required=True)
    parser.add_argument("--warm-metrics", type=Path)
    parser.add_argument("--warm-energy-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    warm = _load(args.warm)
    warm_energy = _load(args.warm_energy)
    warm_metadata = _validated_metadata(warm)
    warm_energy_metadata = _validated_metadata(warm_energy)
    _require_same_provenance(warm_metadata, warm_energy_metadata)

    warm_count = int(np.asarray(warm["weights"]).size)
    warm_energy_count = int(np.asarray(warm_energy["weights"]).size)
    if warm_count != warm_energy_count:
        raise ValueError(
            "PF archives must have the same number of samples for paired comparison"
        )
    paired_fields = ("initial_X", "initial_logq")
    paired = {
        name: bool(
            name in warm
            and name in warm_energy
            and np.array_equal(warm[name], warm_energy[name])
        )
        for name in paired_fields
    }
    failed_pairing = [name for name, matches in paired.items() if not matches]
    if failed_pairing:
        raise ValueError(
            "PF archives are not paired: bitwise mismatch in "
            + ", ".join(failed_pairing)
        )
    result = {
        "warm": _summary(warm),
        "warm_energy": _summary(warm_energy),
        "paired_initial_conditions": paired,
        "same_num_samples": True,
        "evaluation": {
            "warm": _metrics(args.warm_metrics),
            "warm_energy": _metrics(args.warm_energy_metrics),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
