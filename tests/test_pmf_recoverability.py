from __future__ import annotations

import numpy as np

from cg_bms_jax.evaluation.pmf_recoverability import (
    ala2_neighbor_graph_metrics,
    bond_score_direction_metrics,
    radial_scale_coordinates,
    radial_score_direction_metrics,
)

REFERENCE = np.asarray(
    [
        [0.00, 0.00, 0.00],
        [0.15, 0.00, 0.00],
        [0.30, 0.00, 0.00],
        [0.30, 0.15, 0.00],
        [0.30, 0.00, 0.15],
        [0.45, 0.00, 0.15],
    ],
    dtype=np.float64,
)
BONDS = np.asarray(((0, 1), (1, 2), (2, 3), (2, 4), (4, 5)), dtype=np.int64)
REFERENCE_BONDS = np.linalg.norm(
    REFERENCE[BONDS[:, 0]] - REFERENCE[BONDS[:, 1]],
    axis=-1,
)


def test_neighbor_graphs_distinguish_mace_and_painn_cutoffs_and_pbc() -> None:
    chain = np.zeros((2, 6, 3), dtype=np.float64)
    chain[:, :, 0] = np.arange(6)[None, :] * 0.60
    metrics = ala2_neighbor_graph_metrics(chain, mace_box_nm=None)
    assert metrics["mace_0p5_nm"]["connected_frame_fraction"] == 0.0
    assert metrics["mace_0p5_nm"]["isolated_bead_fraction"] == 1.0
    assert metrics["painn_0p8_nm"]["connected_frame_fraction"] == 1.0
    assert metrics["painn_0p8_nm"]["isolated_bead_fraction"] == 0.0

    boundary_pair = np.zeros((1, 6, 3), dtype=np.float64)
    boundary_pair[0, 0, 0] = 0.05
    boundary_pair[0, 1, 0] = 1.95
    # The first two beads are neighbors only after a 2 nm minimum image.
    periodic = ala2_neighbor_graph_metrics(
        boundary_pair,
        mace_box_nm=np.asarray((2.0, 2.0, 2.0)),
    )["mace_0p5_nm"]
    assert periodic["isolated_fraction_by_bead"]["0"] == 0.0
    assert periodic["isolated_fraction_by_bead"]["1"] == 0.0


def test_bond_score_direction_identifies_restorative_long_bond_score() -> None:
    coordinates = np.broadcast_to(REFERENCE, (8, 6, 3)).copy()
    # Lengthen the terminal 4--5 bond from 0.15 to 0.30 nm.
    coordinates[:, 5] = coordinates[:, 4] + 2.0 * (
        REFERENCE[5] - REFERENCE[4]
    )
    score = np.zeros_like(coordinates)
    displacement = coordinates[:, 5] - coordinates[:, 4]
    unit = displacement / np.linalg.norm(displacement, axis=-1, keepdims=True)
    score[:, 5] = -unit
    score[:, 4] = unit

    report = bond_score_direction_metrics(
        coordinates,
        score,
        reference_bond_lengths_nm=REFERENCE_BONDS,
        support_lower_nm=np.full(5, 0.12),
        support_upper_nm=np.full(5, 0.18),
    )
    terminal = report["bonds"]["ALA-C--NME-N"]
    assert terminal["dr_dtau_nm"]["q50"] < 0.0
    assert terminal["long_bond_shortening_fraction"] == 1.0
    assert terminal["restorative_fraction_active"] == 1.0
    assert terminal["above_support_fraction"] == 1.0
    assert terminal["above_support_shortening_fraction"] == 1.0


def test_radial_scan_projection_recovers_injected_score_direction() -> None:
    reference = np.broadcast_to(REFERENCE, (4, 6, 3)).copy()
    scale = 2.0
    scaled = radial_scale_coordinates(reference, scale)
    centre = np.mean(reference, axis=1, keepdims=True)
    # An injected score proportional to -(alpha-1)*reference direction points
    # exactly back toward alpha=1; no PMF/MACE dependency is involved.
    score = -(scale - 1.0) * (reference - centre)
    report = radial_score_direction_metrics(reference, scale, score)
    assert report["dalpha_dtau"]["q50"] < 0.0
    assert report["toward_reference_fraction"] == 1.0
    np.testing.assert_allclose(
        scaled,
        centre + scale * (reference - centre),
    )
