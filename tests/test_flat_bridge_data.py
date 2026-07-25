from __future__ import annotations

import numpy as np
import pytest

from cg_bms_jax.data import (
    bridge_data_sha256,
    bridge_endpoint_bank_from_config,
    full_support_mixture_from_config,
)

BRIDGE_DATA = {
    "schema_version": 1,
    "distribution": "full_support_diagonal_gaussian_mixture_v1",
    "coordinate_space": "physical",
    "seed": 17,
    "num_endpoints": 20_000,
    "components": [
        {
            "label": "basin_right",
            "mean": [48.0, 8.0],
            "scale": [2.5, 2.5],
            "weight": 0.65,
        },
        {
            "label": "basin_central",
            "mean": [32.0, 16.0],
            "scale": [2.5, 2.5],
            "weight": 0.20,
        },
        {
            "label": "basin_upper",
            "mean": [24.0, 32.0],
            "scale": [2.5, 2.5],
            "weight": 0.13,
        },
        {
            "label": "broad_tail",
            "mean": [25.0, 25.0],
            "scale": [15.0, 15.0],
            "weight": 0.02,
        },
    ],
}


def test_bridge_data_digest_is_canonical_and_sensitive() -> None:
    reordered = {
        "components": BRIDGE_DATA["components"],
        "num_endpoints": BRIDGE_DATA["num_endpoints"],
        "seed": BRIDGE_DATA["seed"],
        "coordinate_space": BRIDGE_DATA["coordinate_space"],
        "distribution": BRIDGE_DATA["distribution"],
        "schema_version": BRIDGE_DATA["schema_version"],
    }
    assert bridge_data_sha256(reordered) == bridge_data_sha256(BRIDGE_DATA)

    changed = dict(BRIDGE_DATA)
    changed["seed"] = 18
    assert bridge_data_sha256(changed) != bridge_data_sha256(BRIDGE_DATA)


def test_mb2d_endpoint_bank_is_deterministic_biased_and_affine_consistent() -> None:
    mixture = full_support_mixture_from_config(
        BRIDGE_DATA,
        affine_offset=[25.0, 25.0],
        affine_scale=[10.0, 10.0],
    )
    first = mixture.generate()
    second = mixture.generate()

    np.testing.assert_array_equal(first.endpoint_1, second.endpoint_1)
    np.testing.assert_array_equal(first.physical, second.physical)
    np.testing.assert_array_equal(first.component_index, second.component_index)
    assert first.bridge_data_sha256 == bridge_data_sha256(BRIDGE_DATA)
    assert first.endpoint_1.shape == (20_000, 2)
    assert first.endpoint_1.dtype == np.float32
    np.testing.assert_allclose(
        mixture.to_physical(first.endpoint_1),
        first.physical,
        rtol=0.0,
        atol=4.0e-6,
    )

    observed = np.asarray(first.component_counts) / first.num_endpoints
    expected = np.asarray([0.65, 0.20, 0.13, 0.02])
    np.testing.assert_allclose(observed, expected, atol=0.01, rtol=0.0)
    assert observed[0] > observed[1] > observed[2] > observed[3]


def test_unified_bridge_dispatch_preserves_the_legacy_synthetic_abi(
    tmp_path,
) -> None:
    mixture = full_support_mixture_from_config(
        BRIDGE_DATA,
        affine_offset=[25.0, 25.0],
        affine_scale=[10.0, 10.0],
    )
    legacy = mixture.generate()
    dispatched = bridge_endpoint_bank_from_config(
        BRIDGE_DATA,
        affine_offset=[25.0, 25.0],
        affine_scale=[10.0, 10.0],
        physical_box=[[0.0, 50.0], [0.0, 50.0]],
        base_dir=tmp_path,
    )

    np.testing.assert_array_equal(dispatched.endpoint_1, legacy.endpoint_1)
    np.testing.assert_array_equal(dispatched.physical, legacy.physical)
    np.testing.assert_array_equal(
        dispatched.component_index,
        legacy.component_index,
    )
    assert dispatched.component_labels == legacy.component_labels
    assert dispatched.component_counts == legacy.component_counts
    assert dispatched.bridge_data_sha256 == legacy.bridge_data_sha256
    assert (
        dispatched.distribution
        == "full_support_diagonal_gaussian_mixture_v1"
    )
    assert dispatched.dataset_sha256 is None
    assert dispatched.split_name is None


def test_non_degenerate_gaussian_mixture_has_full_support() -> None:
    mixture = full_support_mixture_from_config(
        BRIDGE_DATA,
        affine_offset=[25.0, 25.0],
        affine_scale=[10.0, 10.0],
    )
    probes = np.asarray(
        [
            [-1_000.0, -1_000.0],
            [0.0, 0.0],
            [24.0, 32.0],
            [1_000.0, 1_000.0],
        ]
    )
    log_density = mixture.log_prob_physical(probes)
    assert np.all(np.isfinite(log_density))


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda data: data.update(distribution="not-versioned"),
            "distribution",
        ),
        (
            lambda data: data["components"][0].update(scale=[0.0, 2.5]),
            "strictly positive",
        ),
        (
            lambda data: data["components"][0].update(weight=0.60),
            "sum to one",
        ),
    ],
)
def test_invalid_full_support_spec_is_rejected(mutator, match: str) -> None:
    copied = {
        **BRIDGE_DATA,
        "components": [dict(value) for value in BRIDGE_DATA["components"]],
    }
    mutator(copied)
    with pytest.raises(ValueError, match=match):
        full_support_mixture_from_config(
            copied,
            affine_offset=[25.0, 25.0],
            affine_scale=[10.0, 10.0],
        )
