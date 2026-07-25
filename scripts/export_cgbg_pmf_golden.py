#!/usr/bin/env python3
"""Export CG-BG oracle energies for the pinned Ala2 PMF.

Run this script with the legacy validation environment and put the pinned
``cg-bms/src`` and ``cg-bg/src`` directories on ``PYTHONPATH``.  It is not a
runtime dependency of :mod:`cg_bms_jax`; it exists solely for cross-repository
acceptance testing.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_torch_type_stubs() -> None:
    """Let CG-BG import type-only DataLoader names without installing Torch."""

    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    class DataLoader:
        pass

    class SubsetRandomSampler:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("The validation stub cannot construct a Torch sampler")

    data.Dataset = Dataset
    data.DataLoader = DataLoader
    data.SubsetRandomSampler = SubsetRandomSampler
    utils.data = data
    torch.utils = utils
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.utils", utils)
    sys.modules.setdefault("torch.utils.data", data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument(
        "--stub-torch",
        action="store_true",
        help="Stub CG-BG's type-only torch DataLoader imports in a pure-JAX env.",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    if args.stub_torch:
        _install_torch_type_stubs()

    # These imports intentionally resolve from the pinned CG-BG source tree
    # supplied by the caller's PYTHONPATH.
    import jax
    from cg_bg.data.ala.data import get_ala_pmf_dataset
    from cg_bg.flow.cgbg import energy_evaluate
    from cg_bg.pmf.trainers import get_ala_trainer

    dataset = get_ala_pmf_dataset(
        str(args.data),
        scale_R=1.0,
        scale_U=1.0,
        fractional=True,
        train_frac=1.0,
        seed=0,
    )
    trainer = get_ala_trainer(
        dataset=dataset,
        batch_size=256,
        epochs=100,
        init_lr=1.0e-3,
        decay_rate=0.9,
        r_cutoff=0.5,
        hidden_irreps="32x0e+32x1o",
        readout_irreps="16x0e",
        output_irreps="1x0e",
        max_ell=3,
        num_interactions=2,
        correlation=3,
        rng=jax.random.PRNGKey(0),
    )
    with args.checkpoint.open("rb") as handle:
        params = pickle.load(handle)
    if isinstance(params, Mapping) and "params" in params:
        params = params["params"]
    with np.load(args.data, allow_pickle=False) as archive:
        total = int(archive["R"].shape[0])
        raw = {
            name: np.asarray(value[: args.count] if value.ndim and len(value) == total else value)
            for name in archive.files
            for value in (archive[name],)
        }
    oracle = np.asarray(
        energy_evaluate(data=raw, trainer=trainer, params=params)
    ).reshape(-1)
    if oracle.shape != (args.count,):
        raise RuntimeError(f"Unexpected CG-BG oracle shape {oracle.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        R=np.asarray(raw["R"]),
        box=np.asarray(raw["box"]),
        energy_cgbg=oracle,
        data_sha256=np.asarray(_sha256(args.data)),
        checkpoint_sha256=np.asarray(_sha256(args.checkpoint)),
        cgbg_revision=np.asarray("948aaeff8a6b25de38b6e7b1112041c1cfd40573"),
    )
    print(f"wrote {args.count} CG-BG oracle energies to {args.output}")


if __name__ == "__main__":
    main()
