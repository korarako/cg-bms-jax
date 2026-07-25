#!/usr/bin/env python3
"""Merge exact PF-ODE reweighting shards and normalize weights globally."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = {
    "R",
    "U",
    "logp",
    "logw",
    "logw_raw",
    "weights",
    "logq_ambient",
    "valid_mask",
    "support_mask",
    "sampler_kind",
    "density_mode",
    "metadata_json",
    "target_reduced_energy",
}
SCALAR_FIELDS = {"sampler_kind", "density_mode", "metadata_json"}
NON_SAMPLE_FIELDS = SCALAR_FIELDS | {"ode_steps_per_batch"}


def _scalar_string(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected a scalar string")
    item = array.reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    missing = REQUIRED_FIELDS.difference(payload)
    if missing:
        raise KeyError(f"{path} is missing fields: {sorted(missing)}")
    return payload


def _comparable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(metadata)
    value.pop("seed", None)
    value.pop("sampling", None)
    likelihood = value.get("likelihood")
    if isinstance(likelihood, dict):
        likelihood.pop("pf_stage", None)
    return value


def _normalize(raw: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe = np.where(valid & np.isfinite(raw), raw, -np.inf)
    if not np.isfinite(safe).any():
        raise ValueError("Merged archive has no finite valid importance weight")
    maximum = float(np.max(safe))
    mass = np.where(np.isfinite(safe), np.exp(safe - maximum), 0.0)
    total = float(np.sum(mass, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Merged importance weights cannot be normalized")
    weights = mass / total
    log_normalizer = maximum + np.log(total)
    logw = np.where(weights > 0.0, safe - log_normalizer, -np.inf)
    return logw, weights


def merge(
    output: Path,
    shard_paths: list[Path],
    *,
    verify_standard_normal_source: bool = False,
) -> dict[str, Any]:
    if len(shard_paths) < 2:
        raise ValueError("At least two shards are required")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    payloads = [_load(path) for path in shard_paths]
    field_set = set(payloads[0])
    if any(set(payload) != field_set for payload in payloads[1:]):
        raise ValueError("Shard archives do not have identical field sets")

    sampler_kind = _scalar_string(payloads[0]["sampler_kind"])
    density_mode = _scalar_string(payloads[0]["density_mode"])
    if sampler_kind != "pf_ode":
        raise ValueError("Only PF-ODE archives may be merged")
    if any(_scalar_string(p["sampler_kind"]) != sampler_kind for p in payloads):
        raise ValueError("sampler_kind differs between shards")
    if any(_scalar_string(p["density_mode"]) != density_mode for p in payloads):
        raise ValueError("density_mode differs between shards")

    metadata = [
        json.loads(_scalar_string(payload["metadata_json"])) for payload in payloads
    ]
    seeds = [int(value["seed"]) for value in metadata]
    if len(set(seeds)) != 1:
        raise ValueError("Deterministic shards must use the same global seed")
    comparable = _comparable_metadata(metadata[0])
    if any(_comparable_metadata(value) != comparable for value in metadata[1:]):
        raise ValueError("Checkpoint, target, domain, or likelihood metadata differs")

    counts = [int(np.asarray(payload["U"]).shape[0]) for payload in payloads]
    if any(count <= 0 for count in counts):
        raise ValueError("Every shard must contain at least one sample")
    for path, payload, count in zip(shard_paths, payloads, counts, strict=True):
        for field, value in payload.items():
            array = np.asarray(value)
            if field in NON_SAMPLE_FIELDS:
                continue
            if array.ndim == 0 or array.shape[0] != count:
                raise ValueError(
                    f"{path}: field {field!r} does not have sample dimension {count}"
                )
        archived_raw = np.asarray(payload["logw_raw"], dtype=np.float64)
        recomputed_raw = -np.asarray(
            payload["target_reduced_energy"], dtype=np.float64
        ) - np.asarray(payload["logq_ambient"], dtype=np.float64)
        finite = np.isfinite(archived_raw) & np.isfinite(recomputed_raw)
        if finite.any() and not np.allclose(
            archived_raw[finite],
            recomputed_raw[finite],
            rtol=2.0e-5,
            atol=2.0e-4,
        ):
            raise ValueError(f"{path}: archived logw_raw is inconsistent with U and logq")

    merged: dict[str, np.ndarray] = {}
    for field in sorted(field_set - NON_SAMPLE_FIELDS):
        if field in {"logw", "logw_raw", "weights", "logp"}:
            continue
        merged[field] = np.concatenate(
            [np.asarray(payload[field]) for payload in payloads],
            axis=0,
        )
    merged["ode_steps_per_batch"] = np.concatenate(
        [np.asarray(payload["ode_steps_per_batch"], dtype=np.int64) for payload in payloads]
    )
    merged["shard_id"] = np.concatenate(
        [
            np.full(count, shard_id, dtype=np.int32)
            for shard_id, count in enumerate(counts)
        ]
    )
    merged["index_within_shard"] = np.concatenate(
        [np.arange(count, dtype=np.int32) for count in counts]
    )

    logq = np.asarray(merged["logq_ambient"], dtype=np.float64)
    reduced = np.asarray(merged["target_reduced_energy"], dtype=np.float64)
    valid = (
        np.asarray(merged["valid_mask"], dtype=bool)
        & np.asarray(merged["support_mask"], dtype=bool)
        & np.isfinite(logq)
        & np.isfinite(reduced)
    )
    raw = -reduced - logq
    logw, weights = _normalize(raw, valid)
    merged["valid_mask"] = valid
    merged["logp"] = logq
    merged["logw_raw"] = raw
    merged["logw"] = logw
    merged["weights"] = weights
    merged["sampler_kind"] = np.asarray(sampler_kind)
    merged["density_mode"] = np.asarray(density_mode)

    initial = np.asarray(merged["initial_X"])
    initial_logq = np.asarray(merged["initial_logq"], dtype=np.float64)
    flattened_initial = np.ascontiguousarray(initial.reshape(initial.shape[0], -1))
    unique_initial = np.unique(
        flattened_initial.view(
            np.dtype((np.void, flattened_initial.dtype.itemsize * flattened_initial.shape[1]))
        )
    )
    if unique_initial.size != flattened_initial.shape[0]:
        raise ValueError("Merged deterministic shards contain duplicate initial samples")
    if verify_standard_normal_source:
        dimension = flattened_initial.shape[1]
        analytic_logq = -0.5 * (
            np.sum(np.square(flattened_initial, dtype=np.float64), axis=1)
            + dimension * np.log(2.0 * np.pi)
        )
        if not np.allclose(
            initial_logq,
            analytic_logq,
            rtol=2.0e-5,
            atol=2.0e-4,
        ):
            maximum_error = float(np.max(np.abs(initial_logq - analytic_logq)))
            raise ValueError(
                "initial_logq is inconsistent with the standard-normal source "
                f"(max abs error {maximum_error:.3e})"
            )

    shard_records = []
    sampling_ranges = []
    for path, value, count in zip(shard_paths, metadata, counts, strict=True):
        sampling = value.get("sampling")
        if not isinstance(sampling, dict):
            raise ValueError(f"{path}: deterministic sampling metadata is missing")
        sampling_ranges.append(
            (
                int(sampling["batch_offset"]),
                int(sampling["local_num_batches"]),
                int(sampling["local_num_samples"]),
            )
        )
        shard_records.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "num_samples": count,
                "batch_offset": int(sampling["batch_offset"]),
                "local_num_batches": int(sampling["local_num_batches"]),
                "pf_stage": str(value.get("likelihood", {}).get("pf_stage", "")),
            }
        )
        pf_stage = Path(shard_records[-1]["pf_stage"])
        if not pf_stage.is_file():
            raise FileNotFoundError(f"PF stage recorded by {path} does not exist: {pf_stage}")
        shard_records[-1]["pf_stage_sha256"] = _sha256(pf_stage)
    sampling_ranges.sort()
    expected_offset = 0
    for offset, batches, local_samples in sampling_ranges:
        if offset != expected_offset:
            raise ValueError("Shard batch ranges are not contiguous from offset zero")
        if local_samples <= 0:
            raise ValueError("Shard sampling metadata has an invalid local sample count")
        expected_offset += batches
    first_sampling = metadata[0]["sampling"]
    sampling_identity_fields = (
        "global_num_samples",
        "batch_size",
        "global_num_batches",
        "key_schedule",
    )
    for value in metadata[1:]:
        sampling = value["sampling"]
        for field in sampling_identity_fields:
            if sampling[field] != first_sampling[field]:
                raise ValueError(
                    f"Deterministic sampling metadata differs for {field!r}"
                )
    if expected_offset != int(first_sampling["global_num_batches"]):
        raise ValueError("Shard ranges do not cover the complete global key schedule")
    if sum(counts) != int(first_sampling["global_num_samples"]):
        raise ValueError("Shard samples do not cover the requested global sample count")

    merged_metadata = copy.deepcopy(metadata[0])
    merged_metadata["seed"] = int(metadata[0]["seed"])
    merged_metadata["sampling"] = {
        "global_num_samples": int(first_sampling["global_num_samples"]),
        "batch_size": int(first_sampling["batch_size"]),
        "global_num_batches": int(first_sampling["global_num_batches"]),
        "batch_offset": 0,
        "local_num_batches": int(first_sampling["global_num_batches"]),
        "local_num_samples": int(first_sampling["global_num_samples"]),
        "key_schedule": str(first_sampling["key_schedule"]),
        "merged_from_shards": shard_records,
    }
    likelihood = merged_metadata.get("likelihood")
    if isinstance(likelihood, dict):
        likelihood["pf_stage"] = [record["path"] for record in shard_records]
    merged_metadata["merge_schema_version"] = 1
    merged["metadata_json"] = np.asarray(
        json.dumps(merged_metadata, sort_keys=True, separators=(",", ":"))
    )

    if not np.isclose(np.sum(weights), 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise AssertionError("Merged normalized weights do not sum to one")
    if np.any(weights[~valid] != 0.0):
        raise AssertionError("Invalid merged samples carry non-zero weight")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **merged)
    temporary.replace(output)

    ess = float(1.0 / np.sum(np.square(weights), dtype=np.float64))
    summary = {
        "output": str(output.resolve()),
        "sha256": _sha256(output),
        "num_samples": int(sum(counts)),
        "num_shards": len(shard_paths),
        "density_mode": density_mode,
        "valid_fraction": float(np.mean(valid)),
        "ess": ess,
        "ess_fraction": ess / float(sum(counts)),
        "max_weight": float(np.max(weights)),
        "weight_sum": float(np.sum(weights)),
        "seed": seeds[0],
        "standard_normal_source_verified": bool(verify_standard_normal_source),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--verify-standard-normal-source", action="store_true")
    arguments = parser.parse_args()
    summary = merge(
        arguments.output,
        arguments.shards,
        verify_standard_normal_source=arguments.verify_standard_normal_source,
    )
    if arguments.summary is not None:
        arguments.summary.parent.mkdir(parents=True, exist_ok=True)
        arguments.summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
