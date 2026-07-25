#!/usr/bin/env python3
"""Audit a proposal-only Ala2 SDE archive before any backward/PF-ODE run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cg_bms_jax.evaluation.proposal_geometry import ProposalGeometryThresholds, audit_npz


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="cg-bms-sample-sde NPZ archive")
    parser.add_argument("--reference", type=Path, default=None, help="optional implicit-AA or six-bead reference NPZ")
    parser.add_argument("--reference-variant", choices=("auto", "cg", "implicit"), default="auto")
    parser.add_argument("--reference-units", choices=("auto", "nm", "angstrom"), default="auto")
    parser.add_argument("--reference-max-frames", type=int, default=100_000)
    parser.add_argument("--kT", type=float, default=2.494338785445972, dest="kT_kj_mol")
    parser.add_argument("--output", type=Path, default=None, help="JSON report path")
    parser.add_argument("--energy-tail-max-kT", type=float, default=50.0)
    parser.add_argument("--no-fail", action="store_true", help="return exit code zero even when a gate fails")
    return parser


def _print_report(report: dict) -> None:
    state = "PASS" if report["overall_pass"] else "FAIL"
    metrics = report["metrics"]
    print(f"Ala2 proposal geometry audit: {state}")
    print(
        f"frames={metrics['num_frames']} finite={metrics['finite_coordinate_fraction']:.6f} "
        f"Rg_median={metrics['rg_nm']['q50']:.6f} nm "
        f"chirality={metrics['chirality_match_fraction']:.6f} "
        f"collision={metrics['collision_fraction']:.6f} "
        f"connected@{metrics['connectivity_cutoff_nm']:.3g}nm="
        f"{metrics['connectivity_fraction']:.6f}"
    )
    improper = metrics["bms_cb_improper"]
    angle = improper["angle_degrees"]
    flat_low, flat_high = improper["flat_bottom_degrees"]
    print(
        "BMS CB improper [2,1,4,3]: "
        f"positive={improper['positive_fraction']:.6f} "
        f"flat[{flat_low:.0f},{flat_high:.0f}]deg={improper['flat_bottom_fraction']:.6f} "
        f"angle_deg(q01={angle['q01']:.3f}, q50={angle['q50']:.3f}, q99={angle['q99']:.3f})"
    )
    print("bond medians [nm]:")
    for name, values in metrics["bond_lengths_nm"].items():
        print(f"  {name}: {values['q50']:.6f} (q01={values['q01']:.6f}, q99={values['q99']:.6f})")
    print("neighbor graphs:")
    for name, graph in metrics["neighbor_graphs"].items():
        print(
            f"  {name}: connected={graph['connected_frame_fraction']:.6f} "
            f"isolated_beads={graph['isolated_bead_fraction']:.6f} "
            f"frames_with_isolated_bead={graph['frames_with_isolated_bead_fraction']:.6f}"
        )
    permutation = metrics["carbon_permutation_diagnostics"]
    print(
        "carbon permutations (diagnostic only): "
        f"all_bonds<0.20nm={permutation['all_bonds_below_0p20_nm_fraction']:.6f} "
        f"all_bonds<0.25nm={permutation['all_bonds_below_0p25_nm_fraction']:.6f}"
    )
    if "best_max_relative_error" in permutation:
        max_error = permutation["best_max_relative_error"]
        mean_error = permutation["best_mean_relative_error"]
        print(
            "  best reference-relative errors: "
            f"max(q50={max_error['q50']:.6f}, mean={max_error['mean']:.6f}) "
            f"mean(q50={mean_error['q50']:.6f}, mean={mean_error['mean']:.6f})"
        )
    if "energy_components" in metrics:
        print(f"energy components (primary={metrics['energy_primary_component']}):")
        preferred = (
            "U_pmf",
            "U_pmf_raw",
            "U_pmf_effective",
            "U_pmf_scaled",
            "U_pmf_target_contribution",
            "U_pmf_floor_delta",
            "U_pmf_transform_delta",
            "U_pmf_gated",
            "U_bond",
            "U_angle",
            "U_repulsion",
            "U_support",
            "U_cb",
            "U_target",
        )
        available = metrics["energy_components"]
        ordered = list(dict.fromkeys((*preferred, *available.keys())))
        for name in ordered:
            if name not in available:
                continue
            component = available[name]
            line = f"  {name}: finite={component['finite_fraction']:.6f}"
            if "q99_minus_median_kT" in component:
                line += f" q99-median={component['q99_minus_median_kT']:.6f} kT"
            if "flat_zero_fraction" in component:
                line += (
                    f" flat/zero={component['flat_zero_fraction']:.6f} "
                    f"(|U|<={component['flat_zero_threshold_kj_mol']:.3g} kJ/mol)"
                )
            print(line)
    elif "energy_q99_minus_median_kT" in metrics:
        # Schema-v2 reports loaded by external callers remain printable.
        print(f"energy q99-median={metrics['energy_q99_minus_median_kT']:.6f} kT")
    print("gates:")
    for name, gate in report["gates"].items():
        mark = "PASS" if gate["passed"] else "FAIL"
        print(
            f"  [{mark}] {name}: {gate['value']:.6g} "
            f"{gate['criterion']} {gate['threshold']:.6g}"
        )


def main() -> int:
    args = _parser().parse_args()
    thresholds = ProposalGeometryThresholds(energy_tail_max_kT=args.energy_tail_max_kT)
    report = audit_npz(
        args.sample,
        reference_path=args.reference,
        reference_variant=args.reference_variant,
        reference_units=args.reference_units,
        reference_max_frames=args.reference_max_frames,
        kT_kj_mol=args.kT_kj_mol,
        thresholds=thresholds,
    )
    output = args.output or args.sample.with_suffix(".geometry_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    _print_report(report)
    print(output)
    return 0 if report["overall_pass"] or args.no_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
