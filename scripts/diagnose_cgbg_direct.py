#!/usr/bin/env python3
"""Compare direct MACE calls with CG-BG ForceMatching.predict.

This acceptance diagnostic runs only in the pinned legacy reference
environment; it is not imported by the pure-JAX package.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from chemtrain.data import preprocessing
from chemutils.models import mace
from jax_md import partition, space
from jax_md_mod import custom_partition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()

    from cg_bg.data.ala.data import get_ala_pmf_dataset
    from cg_bg.flow.cgbg import energy_evaluate
    from cg_bg.pmf.trainers import get_ala_trainer

    dataset = get_ala_pmf_dataset(
        str(args.data), fractional=True, train_frac=1.0, seed=0
    )
    training = jax.tree.map(jnp.asarray, dataset["training"])
    box = training["box"][0]
    species = training["species"][0]
    mask = training["mask"][0]
    displacement, _ = space.periodic_general(box=box, fractional_coordinates=True)
    neighbor_template, stats = preprocessing.allocate_neighborlist(
        training,
        displacement,
        box,
        r_cutoff=0.5,
        mask_key="mask",
        box_key="box",
        format=partition.Sparse,
        batch_size=256,
    )
    init_fn, gnn = mace.mace_neighborlist_pp(
        displacement,
        r_cutoff=0.5,
        n_species=len(species),
        max_edges=stats[1],
        per_particle=False,
        avg_num_neighbors=stats[2],
        mode="energy",
        hidden_irreps="32x0e+32x1o",
        max_ell=3,
        num_interactions=2,
        correlation=3,
        readout_mlp_irreps="16x0e",
        output_irreps="1x0e",
    )
    # Exercise init exactly as get_ala_trainer does before replacing params.
    _ = init_fn(jax.random.PRNGKey(0), training["R"][0], neighbor_template, species=species, mask=mask)
    with args.checkpoint.open("rb") as handle:
        params = pickle.load(handle)
    if isinstance(params, dict) and "params" in params:
        params = params["params"]

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
    with np.load(args.data, allow_pickle=False) as archive:
        total = archive["R"].shape[0]
        raw = {
            key: np.asarray(value[: args.count] if value.ndim and len(value) == total else value)
            for key in archive.files
            for value in (archive[key],)
        }
    oracle = np.asarray(energy_evaluate(raw, trainer, params)).reshape(-1)
    fractional = jnp.squeeze(jnp.asarray(raw["R"])) / raw["box"][0, 0, 0]
    trainer_energy = trainer.reference_energy_fn_template(params)
    direct_batch = {name: jnp.asarray(value) for name, value in raw.items()}
    direct_batch["R"] = fractional
    direct_batch["U"] = jnp.zeros((args.count,), dtype=fractional.dtype)
    trainer_batched = trainer.batched_model(params, direct_batch)

    def one(pos, *, second_mask: bool, all_update_kwargs: bool, with_grad: bool):
        kwargs = {"box": box, "mask": mask}
        if all_update_kwargs:
            kwargs.update(species=species, energy_params=params)
        neighbor = neighbor_template.update(pos, **kwargs)
        if second_mask:
            neighbor = custom_partition.mask_neighbor_list(neighbor, mask)

        def energy_fn(position, dynamic_box):
            return jnp.asarray(
                gnn(
                    params,
                    position,
                    neighbor,
                    species=species,
                    mask=mask,
                    box=dynamic_box,
                )
            ).reshape(())

        if with_grad:
            return jax.value_and_grad(energy_fn, argnums=(0, 1))(pos, box)[0]
        return energy_fn(pos, box)

    variants = {}
    variants["trainer_batched_model"] = np.asarray(trainer_batched["U"]).reshape(-1)
    for second_mask in (False, True):
        for all_kwargs in (False, True):
            for with_grad in (False, True):
                name = f"mask{int(second_mask)}_allkw{int(all_kwargs)}_grad{int(with_grad)}"
                def fn(pos, sm=second_mask, ak=all_kwargs, wg=with_grad):
                    return one(pos, second_mask=sm, all_update_kwargs=ak, with_grad=wg)

                variants[name] = np.asarray(jax.vmap(fn)(fractional)).reshape(-1)

    def trainer_one(pos, *, with_grad: bool):
        neighbor = trainer._nbrs_init.update(
            pos,
            box=box,
            mask=mask,
            species=species,
            energy_params=params,
        )
        neighbor = custom_partition.mask_neighbor_list(neighbor, mask)

        def energy_fn(position, dynamic_box):
            return jnp.asarray(
                trainer_energy(
                    position,
                    neighbor,
                    species=species,
                    mask=mask,
                    box=dynamic_box,
                )
            ).reshape(())

        if with_grad:
            return jax.value_and_grad(energy_fn, argnums=(0, 1))(pos, box)[0]
        return energy_fn(pos, box)

    for with_grad in (False, True):
        name = f"trainer_closure_grad{int(with_grad)}"
        variants[name] = np.asarray(
            jax.vmap(lambda pos, wg=with_grad: trainer_one(pos, with_grad=wg))(fractional)
        ).reshape(-1)
        name = f"trainer_lax_vmap_grad{int(with_grad)}"
        variants[name] = np.asarray(
            jax.lax.map(
                lambda batch, wg=with_grad: jax.vmap(
                    lambda pos: trainer_one(pos, with_grad=wg)
                )(batch),
                fractional[None, ...],
            )[0]
        ).reshape(-1)

    report = {
        "oracle": oracle.tolist(),
        "oracle_nonfinite": np.flatnonzero(~np.isfinite(oracle)).tolist(),
        "stats": [float(np.asarray(value)) for value in stats],
        "neighbor_templates": {
            "idx_equal": bool(
                np.array_equal(
                    np.asarray(neighbor_template.idx),
                    np.asarray(trainer._nbrs_init.idx),
                )
            ),
            "idx_direct": np.asarray(neighbor_template.idx).tolist(),
            "idx_trainer": np.asarray(trainer._nbrs_init.idx).tolist(),
            "reference_max_abs_error": float(
                np.max(
                    np.abs(
                        np.asarray(neighbor_template.reference_position)
                        - np.asarray(trainer._nbrs_init.reference_position)
                    )
                )
            ),
        },
        "variants": {},
    }
    for name, values in variants.items():
        finite = np.isfinite(values) & np.isfinite(oracle)
        diff = np.abs(values[finite] - oracle[finite])
        report["variants"][name] = {
            "nonfinite": np.flatnonzero(~np.isfinite(values)).tolist(),
            "max_abs_error": float(diff.max()) if diff.size else None,
            "first_values": values[:5].tolist(),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
