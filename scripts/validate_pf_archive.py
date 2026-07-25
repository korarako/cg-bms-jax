#!/usr/bin/env python
"""Validate the lightweight provenance contract of a PF/reweight archive.

This command deliberately inspects only scalar metadata and array shapes.  It
is therefore cheap enough to run before reusing a large archive in a resumed
formal matrix, while the full numerical payload is still validated by the
normal evaluation path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _scalar_string(value: Any, *, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{field} must be a scalar string")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def validate_pf_archive(
    archive_path: str | Path,
    *,
    forward_sha256: str,
    backward_sha256: str,
    forward_controller_kind: str,
    seed: int,
    num_samples: int,
    density_mode: str,
    endpoint_distribution: str,
    likelihood_dt0: float,
    likelihood_rtol: float,
    likelihood_atol: float,
    likelihood_max_steps: int,
) -> dict[str, Any]:
    """Fail unless ``archive_path`` belongs to the requested formal run."""

    archive_path = Path(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {
            "R",
            "U",
            "logp",
            "logw",
            "weights",
            "sampler_kind",
            "density_mode",
            "metadata_json",
        }
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"PF archive is missing fields: {sorted(missing)}")

        sampler_kind = _scalar_string(archive["sampler_kind"], field="sampler_kind")
        actual_density_mode = _scalar_string(
            archive["density_mode"],
            field="density_mode",
        )
        metadata = json.loads(
            _scalar_string(archive["metadata_json"], field="metadata_json")
        )
        if not isinstance(metadata, dict):
            raise TypeError("metadata_json must decode to an object")

        sample_count = int(np.asarray(archive["U"]).shape[0])
        if int(np.asarray(archive["R"]).shape[0]) != sample_count:
            raise ValueError("R and U do not contain the same number of samples")
        for field in ("logp", "logw", "weights"):
            if int(np.asarray(archive[field]).shape[0]) != sample_count:
                raise ValueError(f"{field} and U do not contain the same number of samples")

    sampling = metadata.get("sampling")
    if sampling is None:
        # Archives written before the sampling sub-record was introduced still
        # carry a top-level seed, and their true size is pinned by every
        # numerical array checked above.
        sampling = {}
    elif not isinstance(sampling, dict):
        raise ValueError("metadata_json sampling provenance must be an object")
    likelihood = metadata.get("likelihood")
    if not isinstance(likelihood, dict):
        raise ValueError("metadata_json is missing likelihood provenance")
    forward_metadata = metadata.get("forward_metadata")
    backward_metadata = metadata.get("backward_metadata")
    if not isinstance(forward_metadata, dict) or not isinstance(
        backward_metadata,
        dict,
    ):
        raise ValueError("metadata_json is missing checkpoint metadata")
    endpoint_spec_sha256 = metadata.get("equilibrium_endpoint_spec_sha256")
    expected = {
        "sampler_kind": ("pf_ode", sampler_kind),
        "density_mode": (density_mode, actual_density_mode),
        "forward_checkpoint_sha256": (
            forward_sha256,
            metadata.get("forward_checkpoint_sha256"),
        ),
        "backward_checkpoint_sha256": (
            backward_sha256,
            metadata.get("backward_checkpoint_sha256"),
        ),
        "forward_controller_kind": (
            forward_controller_kind,
            metadata.get("forward_controller_kind"),
        ),
        "endpoint_distribution": (
            endpoint_distribution,
            metadata.get("endpoint_distribution"),
        ),
        "equilibrium endpoint identity": (
            endpoint_spec_sha256,
            metadata.get("configured_endpoint_spec_sha256"),
        ),
        "training endpoint identity": (
            endpoint_spec_sha256,
            metadata.get("training_data_sha256"),
        ),
        "forward warm-start identity": (
            endpoint_spec_sha256,
            forward_metadata.get("warmstart_data_sha256"),
        ),
        "backward parent forward": (
            forward_sha256,
            backward_metadata.get("parent_forward_sha256"),
        ),
        "seed": (seed, metadata.get("seed")),
        "archive sample count": (num_samples, sample_count),
        "likelihood.solver": ("dopri5", likelihood.get("solver")),
        "likelihood.divergence": ("exact", likelihood.get("divergence")),
        "likelihood.max_steps": (
            likelihood_max_steps,
            likelihood.get("max_steps"),
        ),
    }
    if sampling:
        expected["sampling.global_num_samples"] = (
            num_samples,
            sampling.get("global_num_samples"),
        )
        expected["sampling.local_num_samples"] = (
            num_samples,
            sampling.get("local_num_samples"),
        )
    for signature in (
        "formal_target_signature",
        "training_target_signature",
        "coordinate_signature",
        "model_signature",
        "sde_signature",
        "topology_signature",
    ):
        expected[f"forward_metadata.{signature}"] = (
            metadata.get(signature),
            forward_metadata.get(signature),
        )
        expected[f"backward_metadata.{signature}"] = (
            metadata.get(signature),
            backward_metadata.get(signature),
        )
    mismatches = [
        f"{field}: expected {wanted!r}, found {actual!r}"
        for field, (wanted, actual) in expected.items()
        if actual != wanted
    ]
    expected_likelihood_floats = {
        "likelihood.dt0": likelihood_dt0,
        "likelihood.rtol": likelihood_rtol,
        "likelihood.atol": likelihood_atol,
    }
    for field, wanted in expected_likelihood_floats.items():
        actual = likelihood.get(field.rsplit(".", maxsplit=1)[1])
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual),
            wanted,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            mismatches.append(
                f"{field}: expected {wanted!r}, found {actual!r}"
            )
    if not isinstance(endpoint_spec_sha256, str) or len(endpoint_spec_sha256) != 64:
        mismatches.append(
            "equilibrium_endpoint_spec_sha256: expected a 64-character digest, "
            f"found {endpoint_spec_sha256!r}"
        )
    if mismatches:
        raise ValueError(
            "Refusing to reuse a PF archive with mismatched provenance:\n  "
            + "\n  ".join(mismatches)
        )
    return {
        "archive": str(archive_path),
        "samples": sample_count,
        "forward_checkpoint_sha256": forward_sha256,
        "backward_checkpoint_sha256": backward_sha256,
        "forward_controller_kind": forward_controller_kind,
        "seed": seed,
        "density_mode": density_mode,
        "endpoint_distribution": endpoint_distribution,
        "sampling_provenance_present": bool(sampling),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--forward-sha256", required=True)
    parser.add_argument("--backward-sha256", required=True)
    parser.add_argument("--forward-controller-kind", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--density-mode", default="ambient_exact")
    parser.add_argument(
        "--endpoint-distribution",
        default="equilibrium_endpoint_npz_v1",
    )
    parser.add_argument("--likelihood-dt0", default=1.0e-3, type=float)
    parser.add_argument("--likelihood-rtol", default=1.0e-5, type=float)
    parser.add_argument("--likelihood-atol", default=1.0e-6, type=float)
    parser.add_argument("--likelihood-max-steps", default=16384, type=int)
    args = parser.parse_args()
    result = validate_pf_archive(
        args.archive,
        forward_sha256=args.forward_sha256,
        backward_sha256=args.backward_sha256,
        forward_controller_kind=args.forward_controller_kind,
        seed=args.seed,
        num_samples=args.num_samples,
        density_mode=args.density_mode,
        endpoint_distribution=args.endpoint_distribution,
        likelihood_dt0=args.likelihood_dt0,
        likelihood_rtol=args.likelihood_rtol,
        likelihood_atol=args.likelihood_atol,
        likelihood_max_steps=args.likelihood_max_steps,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
