#!/usr/bin/env python3
"""Diagnose same-environment Ala2 PMF parity at every inference layer."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np


def _install_torch_type_stubs() -> None:
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    class DataLoader:
        pass

    class SubsetRandomSampler:
        pass

    data.Dataset = Dataset
    data.DataLoader = DataLoader
    data.SubsetRandomSampler = SubsetRandomSampler
    utils.data = data
    torch.utils = utils
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.utils", utils)
    sys.modules.setdefault("torch.utils.data", data)


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    difference = np.abs(left[finite] - right[finite])
    return {
        "left_first": left[:5].tolist(),
        "right_first": right[:5].tolist(),
        "left_nonfinite": np.flatnonzero(~np.isfinite(left)).tolist(),
        "right_nonfinite": np.flatnonzero(~np.isfinite(right)).tolist(),
        "max_abs_error": float(difference.max()) if difference.size else None,
        "mean_abs_error": float(difference.mean()) if difference.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--backend-first", action="store_true")
    args = parser.parse_args()
    _install_torch_type_stubs()

    import jax
    import jax.numpy as jnp
    from cg_bg.data.ala.data import get_ala_pmf_dataset
    from cg_bg.flow.cgbg import energy_evaluate
    from cg_bg.pmf.trainers import get_ala_trainer
    from chemtrain.data import preprocessing
    from chemtrain.learn import force_matching, max_likelihood
    from chemutils.models import mace
    from jax_md import partition, space
    from jax_md_mod import custom_quantity

    from cg_bms_jax.potential.ala2 import (
        CGBGAla2PMF,
        build_cgbg_mace_energy_fn,
    )
    from cg_bms_jax.potential.bundle import CGBGAla2Bundle

    factory_calls: list[dict[str, object]] = []
    original_mace_factory = mace.mace_neighborlist_pp

    def recording_mace_factory(*factory_args, **factory_kwargs):
        init_fn, apply_fn = original_mace_factory(*factory_args, **factory_kwargs)
        record: dict[str, object] = {
            "args": factory_args,
            "kwargs": dict(factory_kwargs),
            "apply": apply_fn,
        }

        def recording_init(*init_args, **init_kwargs):
            value = init_fn(*init_args, **init_kwargs)
            record["init_args"] = init_args
            record["init_kwargs"] = dict(init_kwargs)
            record["init_params"] = value
            return value

        record["init"] = recording_init
        factory_calls.append(record)
        return recording_init, apply_fn

    mace.mace_neighborlist_pp = recording_mace_factory

    bundle = CGBGAla2Bundle.from_data_file(args.data, args.checkpoint)
    backend = None
    if args.backend_first:
        backend = build_cgbg_mace_energy_fn(bundle, trusted_checkpoint=True)

    dataset = get_ala_pmf_dataset(
        str(args.data), fractional=True, train_frac=1.0, seed=0
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
    if isinstance(params, dict) and "params" in params:
        params = params["params"]

    with np.load(args.data, allow_pickle=False) as archive:
        total = int(archive["R"].shape[0])
        raw = {
            key: np.asarray(value[: args.count] if value.ndim and len(value) == total else value)
            for key in archive.files
            for value in (archive[key],)
        }
        allocation_raw = {
            "R": np.asarray(archive["R"]),
            "box": np.asarray(archive["box"]),
            "mask": np.asarray(archive["mask"]),
        }

    oracle = np.asarray(energy_evaluate(raw, trainer, params)).reshape(-1)
    fractional = jnp.asarray(raw["R"]) / jnp.asarray(raw["box"])[0, 0, 0]
    observations = {key: jnp.asarray(value) for key, value in raw.items()}
    observations["R"] = fractional
    observations["U"] = jnp.zeros((args.count,), dtype=fractional.dtype)
    direct = np.asarray(trainer.batched_model(params, observations)["U"]).reshape(-1)
    shmapped = np.asarray(
        max_likelihood.shmap_model(trainer.batched_model)(params, observations)["U"]
    ).reshape(-1)

    # Reconstruct the ForceMatching model from the trainer's own static
    # neighbor list and energy closure.  This isolates trainer.predict from
    # model reconstruction.
    trainer_features = {
        "energy_and_force": custom_quantity.energy_force_wrapper(
            trainer.reference_energy_fn_template
        )
    }
    quantities = {
        "F": custom_quantity.force_wrapper(None),
        "U": custom_quantity.energy_wrapper(None),
    }
    trainer_clone = force_matching.init_model(
        trainer._nbrs_init,
        quantities,
        feature_extract_fns=trainer_features,
    )
    cloned = np.asarray(trainer_clone(params, observations)["U"]).reshape(-1)

    if backend is None:
        backend = build_cgbg_mace_energy_fn(bundle, trusted_checkpoint=True)
    backend_energy = np.asarray(backend.predict_fractional(fractional)[0]).reshape(-1)
    pmf_energy = np.asarray(
        CGBGAla2PMF(bundle, backend).energy_and_grad(jnp.asarray(raw["R"]))[0]
    ).reshape(-1)

    if len(factory_calls) != 2:
        raise RuntimeError(f"Expected two MACE factory calls, got {len(factory_calls)}")
    call_a, call_b = factory_calls
    if args.backend_first:
        backend_call, trainer_call = call_a, call_b
    else:
        trainer_call, backend_call = call_a, call_b

    def _tree_max_abs(left, right) -> float:
        differences = [
            np.max(np.abs(np.asarray(a) - np.asarray(b)))
            for a, b in zip(
                jax.tree.leaves(left),
                jax.tree.leaves(right),
                strict=True,
            )
        ]
        return float(max(differences, default=0.0))

    def _factory_summary(call: dict[str, object]) -> dict[str, object]:
        kwargs = call["kwargs"]
        return {
            key: float(np.asarray(value))
            if key in {"r_cutoff", "avg_num_neighbors"}
            else int(np.asarray(value))
            if key in {"n_species", "max_edges", "max_ell", "num_interactions", "correlation"}
            else value
            for key, value in kwargs.items()
            if key != "positions_test" and key != "neighbor_test"
        }

    same_neighbor = trainer._nbrs_init
    same_position = fractional[1]
    same_kwargs = {
        "species": jnp.asarray(raw["species"])[1],
        "mask": jnp.asarray(raw["mask"])[1],
        "box": jnp.asarray(raw["box"])[1],
    }
    trainer_apply = trainer_call["apply"]
    backend_apply = backend_call["apply"]
    trainer_apply_energy = np.asarray(
        trainer_apply(params, same_position, same_neighbor, **same_kwargs)
    ).reshape(-1)
    backend_apply_energy = np.asarray(
        backend_apply(params, same_position, same_neighbor, **same_kwargs)
    ).reshape(-1)

    # Compare original and manually reconstructed allocation metadata.
    manual_training, _, _ = preprocessing.train_val_test_split(
        allocation_raw,
        train_ratio=0.9,
        val_ratio=0.1,
        shuffle=True,
        shuffle_seed=0,
    )
    manual_training["R"] = manual_training["R"] / manual_training["box"][0, 0, 0]
    original_training = jax.tree.map(jnp.asarray, dataset["training"])
    manual_training = jax.tree.map(jnp.asarray, manual_training)
    box = original_training["box"][0]
    displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)
    manual_neighbor, manual_stats = preprocessing.allocate_neighborlist(
        manual_training,
        displacement,
        box,
        r_cutoff=0.5,
        mask_key="mask",
        box_key="box",
        format=partition.Sparse,
        batch_size=256,
    )
    bundle_box = jnp.asarray(bundle.box_nm)
    bundle_displacement, _ = space.periodic_general(
        box=bundle_box, fractional_coordinates=True
    )
    bundle_neighbor, bundle_stats = preprocessing.allocate_neighborlist(
        manual_training,
        bundle_displacement,
        bundle_box,
        r_cutoff=0.5,
        mask_key="mask",
        box_key="box",
        format=partition.Sparse,
        batch_size=256,
    )

    report = {
        "backend": jax.default_backend(),
        "backend_first": args.backend_first,
        "devices": [str(device) for device in jax.devices()],
        "box_first": np.asarray(raw["box"])[0].tolist(),
        "species_first": np.asarray(raw["species"])[0].tolist(),
        "mask_first": np.asarray(raw["mask"])[0].tolist(),
        "trainer_direct_vs_oracle": _comparison(direct, oracle),
        "trainer_shmapped_vs_oracle": _comparison(shmapped, oracle),
        "trainer_clone_vs_direct": _comparison(cloned, direct),
        "backend_vs_direct": _comparison(backend_energy, direct),
        "backend_vs_oracle": _comparison(backend_energy, oracle),
        "pmf_vs_backend": _comparison(pmf_energy, backend_energy),
        "factory": {
            "trainer": _factory_summary(trainer_call),
            "backend": _factory_summary(backend_call),
            "init_params_max_abs_error": _tree_max_abs(
                trainer_call["init_params"], backend_call["init_params"]
            ),
            "same_apply": _comparison(trainer_apply_energy, backend_apply_energy),
        },
        "allocation": {
            "manual_stats": [float(np.asarray(value)) for value in manual_stats],
            "bundle_stats": [float(np.asarray(value)) for value in bundle_stats],
            "base_box_original": np.asarray(box).tolist(),
            "base_box_bundle": np.asarray(bundle_box).tolist(),
            "bundle_idx_equal": bool(
                np.array_equal(
                    np.asarray(bundle_neighbor.idx),
                    np.asarray(trainer._nbrs_init.idx),
                )
            ),
            "idx_equal": bool(
                np.array_equal(
                    np.asarray(manual_neighbor.idx),
                    np.asarray(trainer._nbrs_init.idx),
                )
            ),
            "reference_max_abs_error": float(
                np.max(
                    np.abs(
                        np.asarray(manual_neighbor.reference_position)
                        - np.asarray(trainer._nbrs_init.reference_position)
                    )
                )
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
