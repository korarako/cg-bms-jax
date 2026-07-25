from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from cg_bms_jax.evaluation.metrics import (
    importance_weight_diagnostics,
    log_weights_to_weights,
)


def _summary(rows: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    fields = tuple(field for field in rows[0] if field != "seed")
    return {
        field: {
            "mean": float(np.mean([row[field] for row in rows])),
            "std": float(np.std([row[field] for row in rows])),
        }
        for field in fields
    }


def _row(
    *,
    seed: int,
    weighted: dict[str, float],
    diagnostics: dict[str, float],
) -> dict[str, float | int]:
    return {
        "seed": seed,
        "js_1d": float(weighted["JS_Divergence"]),
        "pmf_error": float(weighted["PMF_Error"]),
        "ess_fraction": float(diagnostics["ess_fraction"]),
        "max_weight": float(diagnostics["max_weight"]),
        "top_0p1_percent_mass": float(diagnostics["top_0p1_percent_mass"]),
        "top_1_percent_mass": float(diagnostics["top_1_percent_mass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize formal MB CG1D seed runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-percentile", type=float, default=99.0)
    args = parser.parse_args()

    rows: list[dict[str, float | int]] = []
    clipped_rows: list[dict[str, float | int]] = []
    clip_label = f"{args.clip_percentile:g}"
    for seed in args.seeds:
        root = args.run_root / f"seed_{seed}"
        metrics_path = root / "evaluation" / "formal" / "mb_metrics.json"
        clipped_metrics_path = (
            root
            / "evaluation"
            / f"cgbg_clip_{clip_label}"
            / "mb_metrics.json"
        )
        sample_path = root / "samples_and_weights_100000.npz"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        weighted = metrics["weighted_direct"]
        with np.load(sample_path, allow_pickle=False) as archive:
            diagnostics = importance_weight_diagnostics(
                archive["weights"],
                logw_raw=archive["logw_raw"],
            )
            if clipped_metrics_path.is_file():
                clipped_weights = log_weights_to_weights(
                    archive["logw_raw"],
                    clip_percentile=args.clip_percentile,
                    clip_mode="drop",
                )
                clipped_diagnostics = importance_weight_diagnostics(
                    clipped_weights,
                    logw_raw=archive["logw_raw"],
                )
            else:
                clipped_diagnostics = None
        rows.append(
            _row(seed=seed, weighted=weighted, diagnostics=diagnostics)
        )
        if clipped_diagnostics is not None:
            clipped_metrics = json.loads(
                clipped_metrics_path.read_text(encoding="utf-8")
            )
            clipped_rows.append(
                _row(
                    seed=seed,
                    weighted=clipped_metrics["weighted_direct"],
                    diagnostics=clipped_diagnostics,
                )
            )
    if clipped_rows and len(clipped_rows) != len(rows):
        raise RuntimeError(
            "Clip diagnostic is present for only a subset of requested seeds"
        )
    result = {
        "formal_no_clip": True,
        "rows": rows,
        "summary": _summary(rows),
    }
    if clipped_rows:
        result["clip_1_percent_diagnostic"] = {
            "formal": False,
            "clip_percentile": float(args.clip_percentile),
            "removed_upper_percent": float(100.0 - args.clip_percentile),
            "clip_mode": "drop",
            "rows": clipped_rows,
            "summary": _summary(clipped_rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    figure = args.output.with_suffix(".png")
    labels = [f"seed {row['seed']}" for row in rows]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), constrained_layout=True)
    for axis, field, title in zip(
        axes,
        ("js_1d", "pmf_error", "ess_fraction"),
        ("Weighted JS", "Weighted PMF error", "ESS / N"),
        strict=True,
    ):
        if clipped_rows:
            width = 0.36
            axis.bar(
                positions - width / 2.0,
                [row[field] for row in rows],
                width=width,
                label="no clip",
            )
            axis.bar(
                positions + width / 2.0,
                [row[field] for row in clipped_rows],
                width=width,
                label="drop top 1%",
            )
        else:
            axis.bar(positions, [row[field] for row in rows])
        axis.set(title=title, xticks=positions, xticklabels=labels)
        axis.grid(axis="y", alpha=0.25)
    if clipped_rows:
        axes[0].legend(fontsize=8)
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
