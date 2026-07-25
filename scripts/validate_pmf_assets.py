#!/usr/bin/env python3
# ruff: noqa: E402
"""Validate the pinned CG-BG PMF assets against the pure-JAX adapters.

This is an integration check, not a unit test: it loads trusted pickle files
only after checking the SHA-256 values recorded in ``assets/manifest.yaml``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

_deterministic_flag = "--xla_gpu_deterministic_ops=true"
_xla_flags = os.environ.get("XLA_FLAGS", "").split()
if _deterministic_flag not in _xla_flags:
    os.environ["XLA_FLAGS"] = " ".join((*_xla_flags, _deterministic_flag))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from cg_bms_jax.potential.ala2 import (
    AmbientAla2Potential,
    CGBGAla2PMF,
    build_cgbg_mace_energy_fn,
)
from cg_bms_jax.potential.bundle import CGBGAla2Bundle, MBPMFBundle
from cg_bms_jax.potential.mb import CGBGMBPotential


def _shape_report(path: Path) -> dict[str, list[int]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: list(archive[name].shape) for name in archive.files}


def _manifest_entry(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return manifest["files"][name]
    except KeyError as error:
        raise KeyError(f"Asset manifest does not define {name!r}") from error


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=project / "assets" / "cache")
    parser.add_argument("--manifest", type=Path, default=project / "assets" / "manifest.yaml")
    parser.add_argument("--skip-ala2", action="store_true")
    parser.add_argument(
        "--compare-cgbg",
        action="store_true",
        help="Also import CG-BG's RBFMLP and compare the restored MB energy.",
    )
    parser.add_argument(
        "--ala2-cgbg-golden",
        type=Path,
        help="NPZ exported by export_cgbg_pmf_golden.py for energy parity.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    mb_data_spec = _manifest_entry(manifest, "mb_train")
    mb_pmf_spec = _manifest_entry(manifest, "mb_pmf")
    mb_data = args.assets / mb_data_spec["path"]
    mb_checkpoint = args.assets / mb_pmf_spec["path"]
    mb_bundle = MBPMFBundle(
        checkpoint_path=str(mb_checkpoint),
        checkpoint_sha256=mb_pmf_spec["sha256"],
        data_path=str(mb_data),
        data_sha256=mb_data_spec["sha256"],
    )
    mb = CGBGMBPotential.from_bundle(mb_bundle, trusted=True)
    mb_points = jnp.asarray([10.0, 25.0, 40.0])
    mb_result = mb.evaluate(mb_points)
    report: dict[str, Any] = {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "mb": {
            "data_shapes": _shape_report(mb_data),
            "points": np.asarray(mb_points).tolist(),
            "energy": np.asarray(mb_result.energy).tolist(),
            "gradient": np.asarray(mb_result.gradient).tolist(),
            "finite": bool(np.asarray(mb_result.valid_mask).all()),
        },
    }
    if args.compare_cgbg:
        # This optional path is deliberately absent from the production
        # adapter.  It is used in a source-checkout validation environment to
        # prove that the dependency-free manual apply has not drifted.
        from cg_bg.models.rbf_mlp import RBFMLP

        cfg = dict(mb_bundle.model_config)
        upstream_model = RBFMLP(
            hidden_dim=int(cfg["hidden_dim"]),
            n_layers=int(cfg["n_layers"]),
            num_rbf_centers=int(cfg["num_rbf_centers"]),
            sigma=float(cfg["sigma"]),
        )
        upstream = jnp.squeeze(upstream_model.apply({"params": mb.params}, mb_points))
        difference = np.asarray(upstream - mb_result.energy)
        report["mb"]["cgbg_energy"] = np.asarray(upstream).tolist()
        report["mb"]["cgbg_max_abs_error"] = float(np.max(np.abs(difference)))
        if not np.allclose(upstream, mb_result.energy, rtol=1e-6, atol=1e-6):
            raise AssertionError(
                "Manual MB PMF adapter does not match CG-BG RBFMLP: "
                f"max abs error {np.max(np.abs(difference))}"
            )

    if not args.skip_ala2:
        ala_data_spec = _manifest_entry(manifest, "ala2_train")
        ala_pmf_spec = _manifest_entry(manifest, "ala2_pmf")
        ala_data = args.assets / ala_data_spec["path"]
        ala_checkpoint = args.assets / ala_pmf_spec["path"]
        bundle = CGBGAla2Bundle.from_data_file(ala_data, ala_checkpoint)
        energy_fn = build_cgbg_mace_energy_fn(bundle, trusted_checkpoint=True)
        pmf = CGBGAla2PMF(bundle, energy_fn)
        ambient = AmbientAla2Potential(pmf, bundle.standardization_std_nm)
        with np.load(ala_data, allow_pickle=False) as archive:
            physical = jnp.asarray(archive["R"][:8])
        result = pmf.evaluate(physical)
        result_valid = np.asarray(result.valid_mask)
        finite_lanes = np.flatnonzero(result_valid)
        if not result_valid.all():
            raise AssertionError(
                "Ala2 PMF sacrificial-lane backend left a real input non-finite: "
                f"energy={np.asarray(result.energy).tolist()}, "
                f"valid={np.asarray(result.valid_mask).tolist()}, "
                f"gradient_finite={np.isfinite(np.asarray(result.gradient)).reshape(8, -1).all(axis=1).tolist()}"
            )

        # MACE is a stiff float32 model.  Use one fixed frame, direction, batch
        # shape and fourth-order centered step; do not select the best result
        # from a scan during validation.
        validation_lane = 3
        validation_epsilon = 3.875e-3
        point = np.asarray(physical[validation_lane])
        analytical_gradient = np.asarray(
            pmf.energy_and_grad(physical)[1][validation_lane], dtype=np.float32
        )
        direction = analytical_gradient.copy()
        direction -= direction.mean(axis=0, keepdims=True)
        direction /= np.linalg.norm(direction)
        analytical_directional = float(np.sum(analytical_gradient * direction))

        def energy_at(scale: float) -> float:
            displaced = np.asarray(physical).copy()
            displaced[validation_lane] = (
                point + scale * validation_epsilon * direction
            )
            return float(np.asarray(pmf.energy(jnp.asarray(displaced)))[validation_lane])

        numerical_directional = (
            -energy_at(2.0)
            + 8.0 * energy_at(1.0)
            - 8.0 * energy_at(-1.0)
            + energy_at(-2.0)
        ) / (12.0 * validation_epsilon)
        gradient_relative_error = abs(
            analytical_directional - numerical_directional
        ) / max(abs(analytical_directional), abs(numerical_directional), 1.0e-8)
        if gradient_relative_error >= 1.0e-3:
            raise AssertionError(
                "Ala2 PMF gradient finite-difference relative error is "
                f"{gradient_relative_error}, expected <1e-3"
            )
        report["ala2"] = {
            "data_shapes": _shape_report(ala_data),
            "standardization_std_nm": bundle.standardization_std_nm,
            "box_nm": np.asarray(bundle.box_nm).tolist(),
            "species": list(bundle.species),
            "mask": list(bundle.mask),
            "energy_kj_mol": np.asarray(result.energy).tolist(),
            "reduced_energy_pmf": np.asarray(result.reduced_energy).tolist(),
            "gradient_norm": np.linalg.norm(
                np.asarray(result.reduced_gradient).reshape(result_valid.size, -1), axis=1
            ).tolist(),
            "gradient_validation_lane": validation_lane,
            "gradient_validation_epsilon": validation_epsilon,
            "gradient_finite_difference_relative_error": gradient_relative_error,
            "gradient_analytical_directional": analytical_directional,
            "gradient_numerical_directional": numerical_directional,
            "finite_validation_lane_count": int(finite_lanes.size),
            "ambient_adapter_constructed": isinstance(ambient, AmbientAla2Potential),
        }
        if args.ala2_cgbg_golden is not None:
            with np.load(args.ala2_cgbg_golden, allow_pickle=False) as golden:
                golden_coordinates = jnp.asarray(golden["R"])
                golden_energy = np.asarray(golden["energy_cgbg"]).reshape(-1)
            # CG-BG's ForceMatching.predict obtains U from a joint
            # value-and-gradient call (coordinates and box).  The production
            # adapter likewise uses energy_and_grad for terminal scores, so
            # compare that primal rather than a separately compiled U-only
            # graph.
            adapter_energy = np.asarray(
                pmf.energy_and_grad(golden_coordinates)[0]
            ).reshape(-1)
            if adapter_energy.shape != golden_energy.shape:
                raise AssertionError(
                    f"Ala2 parity shapes differ: {adapter_energy.shape} and {golden_energy.shape}"
                )
            finite = np.isfinite(adapter_energy) & np.isfinite(golden_energy)
            error = np.abs(adapter_energy[finite] - golden_energy[finite])
            if not finite.any():
                raise AssertionError("Ala2 PMF parity produced no jointly finite energies")
            report["ala2"]["cgbg_max_abs_error_kj_mol"] = float(error.max())
            report["ala2"]["cgbg_mean_abs_error_kj_mol"] = float(error.mean())
            report["ala2"]["cgbg_parity_count"] = int(error.size)
            report["ala2"]["adapter_nonfinite_indices"] = np.flatnonzero(
                ~np.isfinite(adapter_energy)
            ).tolist()
            report["ala2"]["cgbg_nonfinite_indices"] = np.flatnonzero(
                ~np.isfinite(golden_energy)
            ).tolist()
            report["ala2"]["adapter_all_real_inputs_finite"] = bool(
                np.isfinite(adapter_energy).all()
            )
            if not np.isfinite(adapter_energy).all() or not np.allclose(
                adapter_energy[finite], golden_energy[finite], rtol=0.0, atol=1.0e-5
            ):
                raise AssertionError(
                    "Ala2 PMF adapter does not match CG-BG energy_evaluate: "
                    f"max finite abs error {error.max()} kJ/mol; "
                    f"adapter nonfinite {np.flatnonzero(~np.isfinite(adapter_energy)).tolist()}; "
                    f"CG-BG nonfinite {np.flatnonzero(~np.isfinite(golden_energy)).tolist()}"
                )

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
