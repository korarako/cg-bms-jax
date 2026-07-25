#!/usr/bin/env python3
"""Compare the neighbor statistics that parameterize the CG-BG MACE PMF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from chemtrain.data import preprocessing
from jax_md import partition, space


def _stats(dataset: dict[str, np.ndarray], batch_size: int = 256) -> list[float]:
    box = jnp.asarray(dataset["box"][0])
    displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)
    _neighbors, stats = preprocessing.allocate_neighborlist(
        {key: jnp.asarray(value) for key, value in dataset.items()},
        displacement,
        box,
        r_cutoff=0.5,
        mask_key="mask",
        box_key="box",
        format=partition.Sparse,
        batch_size=batch_size,
    )
    return [float(np.asarray(value)) for value in stats]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode", choices=("replica", "upstream"), required=True)
    args = parser.parse_args()
    if args.mode == "upstream":
        from cg_bg.data.ala.data import get_ala_pmf_dataset

        dataset = get_ala_pmf_dataset(
            str(args.data),
            scale_R=1.0,
            scale_U=1.0,
            fractional=True,
            train_frac=1.0,
            seed=0,
        )["training"]
        selected = {key: np.asarray(dataset[key]) for key in ("R", "box", "mask")}
    else:
        with np.load(args.data, allow_pickle=False) as archive:
            raw = {key: np.asarray(archive[key]) for key in ("R", "box", "mask")}
        selected, _validation, _test = preprocessing.train_val_test_split(
            raw,
            train_ratio=0.9,
            val_ratio=0.1,
            shuffle=True,
            shuffle_seed=0,
        )
        length = float(np.asarray(selected["box"])[0, 0, 0])
        selected["R"] = np.asarray(selected["R"]) / length
    report = {
        "mode": args.mode,
        "count": int(selected["R"].shape[0]),
        "position_dtype": str(selected["R"].dtype),
        "position_range": [float(np.min(selected["R"])), float(np.max(selected["R"]))],
        "stats": _stats(selected),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
