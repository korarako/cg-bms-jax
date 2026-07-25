from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


ARMS = ("equilibrium_bridge_only", "equilibrium_bridge_energy")


def _mean_std(rows: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    fields = tuple(name for name in rows[0] if name != "seed")
    return {
        name: {
            "mean": float(np.mean([float(row[name]) for row in rows])),
            "std": float(np.std([float(row[name]) for row in rows])),
        }
        for name in fields
    }


def _row(seed: int, metrics: dict[str, Any]) -> dict[str, float | int]:
    unweighted = metrics["unweighted"]
    weighted = metrics["weighted"]
    diagnostics = metrics["weight_diagnostics"]
    return {
        "seed": seed,
        "proposal_js_2d": float(unweighted["js_2d"]),
        "weighted_js_2d": float(weighted["js_2d"]),
        "proposal_pmf_rmse": float(unweighted["pmf_rmse"]),
        "weighted_pmf_rmse": float(weighted["pmf_rmse"]),
        "proposal_basin_l1": float(unweighted["basin_l1_error"]),
        "weighted_basin_l1": float(weighted["basin_l1_error"]),
        "proposal_energy_mean_error": float(unweighted["energy_mean_error"]),
        "weighted_energy_mean_error": float(weighted["energy_mean_error"]),
        "ess_fraction": float(diagnostics["ess_fraction"]),
        "max_weight": float(diagnostics["max_weight"]),
        "top_0p1_percent_mass": float(
            diagnostics["top_0p1_percent_mass"]
        ),
        "top_1_percent_mass": float(diagnostics["top_1_percent_mass"]),
        "logw_variance": float(diagnostics["logw_variance"]),
        "finite_logw_fraction": float(diagnostics["finite_logw_fraction"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the formal equilibrium-endpoint MB2D seed matrix."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    result: dict[str, Any] = {
        "formal_no_clip": True,
        "dataset_identity": "equilibrium_exact_v1",
        "num_samples_per_seed": arguments.num_samples,
        "seeds": arguments.seeds,
        "arms": {},
    }
    for arm in ARMS:
        rows: list[dict[str, float | int]] = []
        for seed in arguments.seeds:
            metrics_path = (
                arguments.run_root
                / f"seed_{seed}"
                / f"{arm}_evaluation"
                / "formal"
                / "mb2d_metrics.json"
            )
            if not metrics_path.is_file():
                raise FileNotFoundError(metrics_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(_row(seed, metrics))
        result["arms"][arm] = {
            "rows": rows,
            "summary": _mean_std(rows),
        }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    labels = ("Bridge", "Bridge to Energy-BMS")
    proposal_js = [
        result["arms"][arm]["summary"]["proposal_js_2d"]["mean"]
        for arm in ARMS
    ]
    weighted_js = [
        result["arms"][arm]["summary"]["weighted_js_2d"]["mean"]
        for arm in ARMS
    ]
    ess = [
        result["arms"][arm]["summary"]["ess_fraction"]["mean"]
        for arm in ARMS
    ]
    positions = np.arange(len(ARMS))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(8.4, 3.6),
        constrained_layout=True,
    )
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        proposal_js,
        width=width,
        label="Proposal",
    )
    axes[0].bar(
        positions + width / 2,
        weighted_js,
        width=width,
        label="Reweighted (no clip)",
    )
    axes[0].set(
        ylabel="2D JS divergence",
        xticks=positions,
        xticklabels=labels,
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(positions, ess)
    axes[1].set(
        ylabel="ESS / N",
        xticks=positions,
        xticklabels=labels,
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=12)
    figure.savefig(arguments.output.with_suffix(".png"), dpi=220)
    plt.close(figure)
    print(arguments.output)


if __name__ == "__main__":
    main()
