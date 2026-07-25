from __future__ import annotations

import math

import numpy as np

from cg_bms_jax.evaluation.proposal_geometry import (
    CARBON_BEAD_INDICES,
    IMPLICIT_OPENMM_CORE_BETA_INDICES,
    audit_ala2_proposal,
    audit_npz,
    carbon_permutation_metrics,
    geometry_metrics,
    map_core_beta,
    prepare_core_beta_coordinates,
)
from cg_bms_jax.runtime import compose_config

REFERENCE_STANDARDIZED = np.asarray(
    [
        [-1.0597709615, 2.2057466599, 0.0629004387],
        [-0.8727528773, 0.9908810532, -0.4739329682],
        [-0.0203553100, -0.0437054559, 0.2357073928],
        [-1.0266516483, -0.9203141742, 1.0651516885],
        [0.9161808241, -0.8909127554, -0.7068272997],
        [2.0633499731, -1.3416953277, -0.1829992521],
    ],
    dtype=np.float64,
)
REFERENCE_NM = REFERENCE_STANDARDIZED * 0.0983713


def _core_beta_with_torsion(angle: float) -> np.ndarray:
    coordinates = np.zeros((6, 3), dtype=np.float64)
    coordinates[2] = (0.0, 1.0, 0.0)
    coordinates[1] = (0.0, 0.0, 0.0)
    coordinates[4] = (1.0, 0.0, 0.0)
    coordinates[3] = (1.0, math.cos(angle), math.sin(angle))
    coordinates[0] = (-0.3, 0.2, 0.1)
    coordinates[5] = (1.3, -0.2, 0.1)
    return coordinates


def test_cold_clip_experiments_change_only_the_clip_cap() -> None:
    c100 = compose_config("train_forward", ["experiment=ala2_ambient18_300k_cold_clip_c100_2k"])
    c150 = compose_config("train_forward", ["experiment=ala2_ambient18_300k_cold_clip_c150_2k"])
    for config in (c100, c150):
        experiment = config.experiment
        assert experiment.source.mean == 0.0
        assert experiment.source.sigma == 1.0
        assert experiment.sde.sigma_min == 0.01
        assert experiment.sde.sigma_max == 1.0
        assert experiment.sde.rho == 7.0
        assert experiment.sde.steps == 100
        assert experiment.training.outer_iterations == 20
        assert experiment.training.rollout_samples == 2048
        assert experiment.training.buffer_capacity == 16384
        assert experiment.training.batch_size == 64
        assert experiment.training.gradient_steps == 100
        assert experiment.training.learning_rate == 2.0e-5
        assert experiment.training.damping == 50.0
        assert experiment.training.gradient_clip_norm == 1.0
        assert experiment.training.weight_decay == 1.0e-3
    assert c100.experiment.training.terminal_score_clip_norm == 100.0
    assert c150.experiment.training.terminal_score_clip_norm == 150.0


def test_implicit_mapping_and_angstrom_unit_inference() -> None:
    full = np.zeros((3, 22, 3), dtype=np.float64)
    full[:, IMPLICIT_OPENMM_CORE_BETA_INDICES, :] = REFERENCE_NM[None, ...] * 10.0
    mapped = map_core_beta(full, variant="implicit")
    np.testing.assert_allclose(mapped[0], REFERENCE_NM * 10.0)
    converted = prepare_core_beta_coordinates(full, variant="implicit", units="auto")
    np.testing.assert_allclose(converted, np.broadcast_to(REFERENCE_NM, (3, 6, 3)))


def test_reference_like_proposal_passes_physical_gates() -> None:
    rng = np.random.default_rng(9)
    reference = np.broadcast_to(REFERENCE_NM, (4096, 6, 3)).copy()
    proposal = reference + rng.normal(scale=2.0e-4, size=reference.shape)
    energies = rng.normal(loc=-10.0, scale=1.0, size=proposal.shape[0])
    report = audit_ala2_proposal(
        proposal,
        energies_kj_mol=energies,
        archive_valid_mask=np.ones(proposal.shape[0], dtype=bool),
        reference_nm=reference,
    )
    assert report["overall_pass"]
    assert all(gate["passed"] for gate in report["gates"].values())


def test_dissociated_proposal_fails_geometry_even_when_finite() -> None:
    reference = np.broadcast_to(REFERENCE_NM, (128, 6, 3)).copy()
    proposal = reference * 6.0
    report = audit_ala2_proposal(
        proposal,
        energies_kj_mol=np.zeros(proposal.shape[0]),
        reference_nm=reference,
    )
    assert report["metrics"]["finite_coordinate_fraction"] == 1.0
    assert report["gates"]["finite_coordinates"]["passed"]
    assert not report["gates"]["bond_q99_upper"]["passed"]
    assert not report["gates"]["connectivity"]["passed"]
    assert not report["overall_pass"]


def test_carbon_permutation_diagnostic_recovers_relabelled_reference() -> None:
    reference = np.broadcast_to(REFERENCE_NM, (32, 6, 3)).copy()
    relabelled = reference.copy()
    relabelled[:, CARBON_BEAD_INDICES, :] = reference[:, CARBON_BEAD_INDICES[[2, 3, 0, 1]], :]
    reference_bonds = np.linalg.norm(
        REFERENCE_NM[np.asarray((0, 1, 2, 2, 4))]
        - REFERENCE_NM[np.asarray((1, 2, 3, 4, 5))],
        axis=-1,
    )

    diagnostics = carbon_permutation_metrics(
        relabelled,
        reference_bond_lengths_nm=reference_bonds,
    )

    assert diagnostics["num_permutations"] == 24
    assert diagnostics["all_bonds_below_0p20_nm_fraction"] == 1.0
    assert diagnostics["all_bonds_below_0p25_nm_fraction"] == 1.0
    assert diagnostics["best_max_relative_error"]["q99"] < 1.0e-12
    assert diagnostics["best_mean_relative_error"]["q99"] < 1.0e-12


def test_mace_and_painn_graph_diagnostics_use_distinct_cutoffs() -> None:
    chain = np.zeros((4, 6, 3), dtype=np.float64)
    chain[:, :, 0] = np.arange(6, dtype=np.float64)[None, :] * 0.60

    metrics = geometry_metrics(chain)
    mace = metrics["neighbor_graphs"]["mace_0p5_nm"]
    painn = metrics["neighbor_graphs"]["painn_0p8_nm"]

    assert mace["connected_frame_fraction"] == 0.0
    assert mace["isolated_bead_fraction"] == 1.0
    assert mace["frames_with_isolated_bead_fraction"] == 1.0
    assert painn["connected_frame_fraction"] == 1.0
    assert painn["isolated_bead_fraction"] == 0.0
    assert painn["frames_with_isolated_bead_fraction"] == 0.0


def test_permutation_and_graph_metrics_are_not_hard_gates() -> None:
    reference = np.broadcast_to(REFERENCE_NM, (16, 6, 3)).copy()
    report = audit_ala2_proposal(reference, reference_nm=reference)

    assert "carbon_permutation_diagnostics" in report["metrics"]
    assert "neighbor_graphs" in report["metrics"]
    assert "carbon_permutation_diagnostics" not in report["gates"]
    assert "neighbor_graphs" not in report["gates"]


def test_bms_cb_improper_reports_positive_and_flat_bottom_fractions() -> None:
    coordinates = np.stack(
        [
            _core_beta_with_torsion(-0.5),
            _core_beta_with_torsion(math.radians(20.0)),
            _core_beta_with_torsion(math.radians(35.0)),
            _core_beta_with_torsion(math.radians(70.0)),
        ]
    )

    improper = geometry_metrics(coordinates)["bms_cb_improper"]

    assert improper["indices"] == [2, 1, 4, 3]
    assert improper["finite_fraction"] == 1.0
    assert improper["positive_fraction"] == 0.75
    assert improper["flat_bottom_degrees"] == [10.0, 60.0]
    assert improper["flat_bottom_fraction"] == 0.5
    np.testing.assert_allclose(improper["angle_degrees"]["q50"], 27.5, atol=1.0e-6)


def test_npz_ucb_components_use_pmf_as_primary_energy(tmp_path) -> None:
    coordinates = np.broadcast_to(REFERENCE_NM, (8, 6, 3)).copy()
    pmf = np.linspace(0.0, 7.0, num=8)
    cb = np.asarray([0.0, 1.0e-8, -1.0e-8, 1.0e-6, 0.0, 0.0, 1000.0, 2000.0])
    target = pmf + cb
    archive = tmp_path / "ucb.npz"
    np.savez(archive, R=coordinates, U=target, U_target=target, U_pmf=pmf, U_cb=cb)

    report = audit_npz(archive)
    metrics = report["metrics"]
    components = metrics["energy_components"]

    assert metrics["energy_primary_component"] == "U_pmf"
    assert metrics["energy_finite_fraction"] == 1.0
    assert components["U_pmf"]["finite_fraction"] == 1.0
    assert components["U_cb"]["finite_fraction"] == 1.0
    assert components["U_target"]["finite_fraction"] == 1.0
    assert components["U_cb"]["flat_zero_fraction"] == 0.75
    assert metrics["energy_q99_minus_median_kT"] == components["U_pmf"][
        "q99_minus_median_kT"
    ]
    assert components["U_target"]["q99_minus_median_kT"] > metrics[
        "energy_q99_minus_median_kT"
    ]


def test_npz_legacy_u_only_archive_remains_supported(tmp_path) -> None:
    coordinates = np.broadcast_to(REFERENCE_NM, (8, 6, 3)).copy()
    energy = np.linspace(-2.0, 2.0, num=8)
    archive = tmp_path / "legacy.npz"
    np.savez(archive, R=coordinates, U=energy)

    report = audit_npz(archive)

    assert report["metrics"]["energy_primary_component"] == "U_target"
    assert set(report["metrics"]["energy_components"]) == {"U_target"}
    assert report["metrics"]["energy_finite_fraction"] == 1.0
