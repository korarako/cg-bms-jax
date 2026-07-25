"""Full-output parity against a fixture exported from pinned Torch BMS."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze, unfreeze

from cg_bms_jax.model.painn import PaiNN

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "painn_tiny_torch_golden.npz"
UPSTREAM_REVISION = "d19a27b854fd77387c43109e37b597bc35252e7d"


def _encoded_state_key(key: str) -> str:
    return "state__" + key.replace(".", "__")


def _torch_array(fixture: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    encoded = _encoded_state_key(key)
    if encoded not in fixture.files:
        raise KeyError(f"Golden fixture lacks Torch state tensor {key!r} ({encoded!r}).")
    return np.asarray(fixture[encoded])


def _assign(tree: dict, path: tuple[str, ...], value: np.ndarray) -> None:
    cursor = tree
    for component in path[:-1]:
        if component not in cursor:
            raise KeyError(f"JAX variable path does not exist: {'/'.join(path)}")
        cursor = cursor[component]
    if path[-1] not in cursor:
        raise KeyError(f"JAX variable leaf does not exist: {'/'.join(path)}")
    cursor[path[-1]] = jnp.asarray(value, dtype=cursor[path[-1]].dtype)


def _copy_linear(
    tree: dict,
    fixture: np.lib.npyio.NpzFile,
    torch_prefix: str,
    jax_path: tuple[str, ...],
    *,
    bias: bool,
) -> None:
    # Torch weight is (out, in); Flax kernel is (in, out).
    _assign(
        tree,
        (*jax_path, "kernel"),
        _torch_array(fixture, f"{torch_prefix}.weight").T,
    )
    if bias:
        _assign(
            tree,
            (*jax_path, "bias"),
            _torch_array(fixture, f"{torch_prefix}.bias"),
        )


def _variables_from_torch_fixture(
    model: PaiNN,
    fixture: np.lib.npyio.NpzFile,
    time: jax.Array,
    positions: jax.Array,
    num_layers: int,
):
    variables = unfreeze(
        model.init(
            {
                "params": jax.random.PRNGKey(101),
                "constants": jax.random.PRNGKey(102),
            },
            time,
            positions,
        )
    )
    _assign(
        variables,
        ("params", "atom_embedding", "embedding"),
        _torch_array(fixture, "atom_embedding.embedding.weight"),
    )
    _assign(
        variables,
        ("constants", "time_embedding", "freqs"),
        _torch_array(fixture, "time_embedding.freqs"),
    )

    for index in range(num_layers):
        message = ("params", f"message_{index}")
        _copy_linear(
            variables,
            fixture,
            f"messages.{index}.mlp_phi.0",
            (*message, "mlp_phi", "linear_0"),
            bias=True,
        )
        _copy_linear(
            variables,
            fixture,
            f"messages.{index}.mlp_phi.2",
            (*message, "mlp_phi", "linear_1"),
            bias=True,
        )
        _copy_linear(
            variables,
            fixture,
            f"messages.{index}.linear_W",
            (*message, "linear_W"),
            bias=True,
        )

        update = ("params", f"update_{index}")
        _copy_linear(
            variables,
            fixture,
            f"updates.{index}.linear_UV",
            (*update, "linear_UV"),
            bias=False,
        )
        _copy_linear(
            variables,
            fixture,
            f"updates.{index}.mlp_a.0",
            (*update, "mlp_a", "linear_0"),
            bias=True,
        )
        _copy_linear(
            variables,
            fixture,
            f"updates.{index}.mlp_a.2",
            (*update, "mlp_a", "linear_1"),
            bias=True,
        )

        for norm_group, torch_group in (("norm_1", "norms_1"), ("norm_2", "norms_2")):
            jax_norm = ("params", f"{norm_group}_{index}")
            for leaf in ("affine_s_weight", "affine_s_bias", "affine_v_weight"):
                _assign(
                    variables,
                    (*jax_norm, leaf),
                    _torch_array(fixture, f"{torch_group}.{index}.{leaf}"),
                )

    for index in range(2):
        torch_block = f"output_block.out_vector.{index}"
        jax_block = ("params", "output_block", f"block_{index}")
        _copy_linear(
            variables,
            fixture,
            f"{torch_block}.linear_v1",
            (*jax_block, "linear_v1"),
            bias=False,
        )
        _copy_linear(
            variables,
            fixture,
            f"{torch_block}.linear_v2",
            (*jax_block, "linear_v2"),
            bias=False,
        )
        _copy_linear(
            variables,
            fixture,
            f"{torch_block}.mlp_s.0",
            (*jax_block, "mlp_s", "linear_0"),
            bias=True,
        )
        _copy_linear(
            variables,
            fixture,
            f"{torch_block}.mlp_s.2",
            (*jax_block, "mlp_s", "linear_1"),
            bias=True,
        )
    return freeze(variables)


def test_tiny_painn_matches_pinned_torch_full_output():
    if not FIXTURE.is_file():
        pytest.skip(
            "Torch PaiNN golden fixture is absent. Generate it in the old cg-bms "
            "environment with: python scripts/export_painn_golden.py "
            "--bms-root /ds/project/weilong/ke/cg-bms"
        )

    with np.load(FIXTURE, allow_pickle=False) as fixture:
        metadata = json.loads(str(fixture["metadata_json"].item()))
        assert metadata["schema_version"] == 1
        assert metadata["upstream_revision"] == UPSTREAM_REVISION
        assert metadata["dense_layout"] == "torch_out_in"
        assert metadata["cross_product"] == "sender_vector_cross_unit_ij"
        config = metadata["model"]
        positions = jnp.asarray(fixture["positions"], dtype=jnp.float32)
        time = jnp.asarray(fixture["time"], dtype=jnp.float32)
        expected = np.asarray(fixture["output"], dtype=np.float32)

        model = PaiNN(
            num_features=int(config["num_features"]),
            num_radial_basis=int(config["num_radial_basis"]),
            num_layers=int(config["num_layers"]),
            num_elements=int(config["num_elements"]),
            r_max=float(config["r_max"]),
            r_offset=float(config["r_offset"]),
            time_init_mode=str(config["time_init_mode"]),
            parity_breaking=bool(config["parity_breaking"]),
            conservative=bool(config["conservative"]),
            unique_atom_indices=bool(config["unique_atom_indices"]),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
        variables = _variables_from_torch_fixture(
            model,
            fixture,
            time,
            positions,
            num_layers=int(config["num_layers"]),
        )
        actual = np.asarray(model.apply(variables, time, positions))

    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-6)

