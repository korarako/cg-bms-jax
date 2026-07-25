"""Command-line entry point for analytic MB2D reweighting studies."""

from __future__ import annotations

import argparse
from pathlib import Path

from cg_bms_jax.evaluation.mb2d import (
    compare_likelihood_archives,
    compare_mb2d_archives,
    compare_sde_pf_archives,
    evaluate_mb2d,
    evaluate_pooled_mb2d,
    sample_size_convergence,
    temperature_reweighting_sweep,
)
from cg_bms_jax.runtime import resolve_project_path


def _comma_numbers(value: str, *, kind: type[int] | type[float]) -> tuple[int, ...] | tuple[float, ...]:
    try:
        return tuple(kind(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated numeric list: {value!r}") from error


def _labelled_archives(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Archive must use LABEL=PATH syntax, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in result:
            raise ValueError(f"Archive label must be unique and non-empty: {label!r}")
        path = resolve_project_path(raw_path.strip())
        if not path.is_file():
            raise FileNotFoundError(path)
        result[label] = path
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analytic Muller--Brown 2D evaluation and no-clip sweeps."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate one MB2D archive.")
    evaluate.add_argument("--sample", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--kt", type=float, default=1.0)
    evaluate.add_argument("--bins", type=int, default=120)
    evaluate.add_argument("--energy-lower-quantile", type=float, default=0.005)
    evaluate.add_argument("--energy-upper-quantile", type=float, default=0.995)
    evaluate.add_argument("--energy-view-padding", type=float, default=0.05)
    evaluate.add_argument("--energy-bins", type=int, default=160)
    evaluate.add_argument("--clip-percentile", type=float)
    evaluate.add_argument("--clip-mode", choices=("drop", "cap"), default="drop")

    pool = subparsers.add_parser(
        "pool",
        help="Make an equal-seed all-sample visualization from formal archives.",
    )
    pool.add_argument(
        "--archive",
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    pool.add_argument("--output-dir", required=True)
    pool.add_argument("--kt", type=float, default=1.0)
    pool.add_argument("--bins", type=int, default=160)
    pool.add_argument("--energy-lower-quantile", type=float, default=0.005)
    pool.add_argument("--energy-upper-quantile", type=float, default=0.995)
    pool.add_argument("--energy-view-padding", type=float, default=0.05)
    pool.add_argument("--energy-bins", type=int, default=200)
    pool.add_argument("--clip-percentile", type=float)
    pool.add_argument("--clip-mode", choices=("drop", "cap"), default="drop")

    convergence = subparsers.add_parser(
        "sample-size",
        help="Nested-prefix sample-size convergence with repeated permutations.",
    )
    convergence.add_argument("--sample", required=True)
    convergence.add_argument("--output-dir", required=True)
    convergence.add_argument(
        "--sizes",
        default="1000,2000,5000,10000,20000,50000,100000",
    )
    convergence.add_argument("--repeats", type=int, default=5)
    convergence.add_argument("--seed", type=int, default=0)
    convergence.add_argument("--kt", type=float, default=1.0)
    convergence.add_argument("--bins", type=int, default=100)

    temperature = subparsers.add_parser(
        "temperature",
        help="Reweight one fixed PF proposal to several inverse temperatures.",
    )
    temperature.add_argument("--sample", required=True)
    temperature.add_argument("--output-dir", required=True)
    temperature.add_argument("--betas", required=True)
    temperature.add_argument("--bins", type=int, default=100)

    compare = subparsers.add_parser(
        "compare",
        help="Compare arms, forward checkpoints, or backward budgets.",
    )
    compare.add_argument(
        "--archive",
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--kt", type=float, default=1.0)
    compare.add_argument("--bins", type=int, default=100)

    tolerance = subparsers.add_parser(
        "tolerance",
        help="Compare coupled PF/logq archives against a strict reference.",
    )
    tolerance.add_argument(
        "--archive",
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    tolerance.add_argument("--reference", required=True)
    tolerance.add_argument("--output", required=True)

    dynamics = subparsers.add_parser(
        "dynamics",
        help="Compare forward-SDE and PF terminal sample laws.",
    )
    dynamics.add_argument("--sde", required=True)
    dynamics.add_argument("--pf", required=True)
    dynamics.add_argument("--output-dir", required=True)
    dynamics.add_argument("--bins", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_mb2d(
            sample=resolve_project_path(args.sample),
            output_dir=resolve_project_path(args.output_dir),
            kT=args.kt,
            bins=args.bins,
            energy_lower_quantile=args.energy_lower_quantile,
            energy_upper_quantile=args.energy_upper_quantile,
            energy_view_padding=args.energy_view_padding,
            energy_histogram_bins=args.energy_bins,
            clip_percentile=args.clip_percentile,
            clip_mode=args.clip_mode,
        )
        print(result["metrics_path"])
        return
    if args.command == "pool":
        archives = _labelled_archives(args.archive)
        result = evaluate_pooled_mb2d(
            archives,
            output_dir=resolve_project_path(args.output_dir),
            kT=args.kt,
            bins=args.bins,
            energy_lower_quantile=args.energy_lower_quantile,
            energy_upper_quantile=args.energy_upper_quantile,
            energy_view_padding=args.energy_view_padding,
            energy_histogram_bins=args.energy_bins,
            clip_percentile=args.clip_percentile,
            clip_mode=args.clip_mode,
        )
        print(result["metrics_path"])
        return
    if args.command == "sample-size":
        sample_size_convergence(
            sample=resolve_project_path(args.sample),
            output_dir=resolve_project_path(args.output_dir),
            sizes=_comma_numbers(args.sizes, kind=int),
            repeats=args.repeats,
            seed=args.seed,
            kT=args.kt,
            bins=args.bins,
        )
        print(resolve_project_path(args.output_dir) / "sample_size_convergence.json")
        return
    if args.command == "temperature":
        temperature_reweighting_sweep(
            sample=resolve_project_path(args.sample),
            output_dir=resolve_project_path(args.output_dir),
            betas=_comma_numbers(args.betas, kind=float),
            bins=args.bins,
        )
        print(resolve_project_path(args.output_dir) / "temperature_reweighting.json")
        return
    if args.command == "dynamics":
        compare_sde_pf_archives(
            sde_sample=resolve_project_path(args.sde),
            pf_sample=resolve_project_path(args.pf),
            output_dir=resolve_project_path(args.output_dir),
            bins=args.bins,
        )
        print(resolve_project_path(args.output_dir) / "sde_pf_alignment.json")
        return
    archives = _labelled_archives(args.archive)
    if args.command == "compare":
        compare_mb2d_archives(
            archives,
            output_dir=resolve_project_path(args.output_dir),
            kT=args.kt,
            bins=args.bins,
        )
        print(resolve_project_path(args.output_dir) / "comparison.json")
        return
    if args.command == "tolerance":
        compare_likelihood_archives(
            archives,
            reference_label=args.reference,
            output_path=resolve_project_path(args.output),
        )
        print(resolve_project_path(args.output))
        return
    raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    main()
