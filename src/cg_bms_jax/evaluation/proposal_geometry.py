"""Physical stability gates for six-bead Ala2 proposal-only samples.

The stochastic SDE sampler deliberately does not attach a likelihood.  This
module therefore audits only coordinate support and PMF-energy tails; it never
interprets an SDE archive as a reweighted sample.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Literal

import numpy as np

CORE_BETA_BONDS = np.asarray(((0, 1), (1, 2), (2, 3), (2, 4), (4, 5)), dtype=np.int64)
CORE_BETA_BOND_NAMES = ("ACE-C--ALA-N", "ALA-N--ALA-CA", "ALA-CA--ALA-CB", "ALA-CA--ALA-C", "ALA-C--NME-N")
IMPLICIT_OPENMM_CORE_BETA_INDICES = np.asarray((4, 6, 7, 12, 8, 16), dtype=np.int64)
PAIR_INDICES = np.asarray(tuple((i, j) for i in range(6) for j in range(i + 1, 6)), dtype=np.int64)
CARBON_BEAD_INDICES = np.asarray((0, 2, 3, 4), dtype=np.int64)
CARBON_BEAD_PERMUTATIONS = np.asarray(
    tuple(permutations(CARBON_BEAD_INDICES.tolist())), dtype=np.int64
)
MACE_GRAPH_CUTOFF_NM = 0.50
PAINN_GRAPH_CUTOFF_NM = 0.80
BMS_CB_IMPROPER_INDICES = (2, 1, 4, 3)
BMS_CB_FLAT_BOTTOM_DEGREES = (10.0, 60.0)
DEFAULT_UCB_ZERO_THRESHOLD_KJ_MOL = 1.0e-6


@dataclass(frozen=True)
class ProposalGeometryThresholds:
    """Hard gates for a forward-only cold-start pilot.

    Distances are nanometres.  ``energy_tail_max_kT`` applies to
    ``q99(U/kT) - median(U/kT)`` and is therefore invariant to the arbitrary
    additive zero of a learned PMF.
    """

    bond_quantile_low_nm: float = 0.09
    bond_quantile_high_nm: float = 0.21
    bond_all_max_nm: float = 0.20
    bond_all_min_fraction: float = 0.99
    bond_median_max_relative_error: float = 0.10
    rg_median_min_nm: float = 0.15
    rg_median_max_nm: float = 0.20
    rg_median_max_relative_error: float = 0.10
    chirality_match_min_fraction: float = 0.99
    collision_distance_nm: float = 0.08
    collision_max_fraction: float = 0.001
    connectivity_cutoff_nm: float = 0.50
    connectivity_min_fraction: float = 0.999
    archive_valid_min_fraction: float = 0.999
    energy_tail_max_kT: float = 50.0

    def __post_init__(self) -> None:
        fractions = (
            self.bond_all_min_fraction,
            self.chirality_match_min_fraction,
            self.collision_max_fraction,
            self.connectivity_min_fraction,
            self.archive_valid_min_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("fraction thresholds must lie in [0,1]")
        if not 0.0 < self.bond_quantile_low_nm < self.bond_quantile_high_nm:
            raise ValueError("bond quantile limits must be positive and ordered")
        if not self.rg_median_min_nm < self.rg_median_max_nm:
            raise ValueError("Rg limits must be ordered")
        if self.connectivity_cutoff_nm <= 0.0 or self.collision_distance_nm <= 0.0:
            raise ValueError("distance thresholds must be positive")


def _normalise_coordinate_shape(value: Any, *, label: str) -> np.ndarray:
    coordinates = np.asarray(value)
    if coordinates.ndim == 2 and coordinates.shape[-1] == 3:
        coordinates = coordinates[None, ...]
    elif coordinates.ndim == 2 and coordinates.shape[-1] % 3 == 0:
        coordinates = coordinates.reshape((coordinates.shape[0], -1, 3))
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"{label} coordinates must have shape (B,N,3), got {coordinates.shape}")
    if coordinates.shape[0] == 0:
        raise ValueError(f"{label} coordinates are empty")
    return np.asarray(coordinates, dtype=np.float64)


def map_core_beta(
    coordinates: Any,
    *,
    variant: Literal["auto", "cg", "implicit"] = "auto",
) -> np.ndarray:
    """Return core-beta coordinates in ``[ACE-C,N,CA,CB,C,NME-N]`` order."""

    value = _normalise_coordinate_shape(coordinates, label="Ala2")
    resolved = "cg" if variant == "auto" and value.shape[1] == 6 else variant
    if variant == "auto" and value.shape[1] != 6:
        resolved = "implicit"
    if resolved == "cg":
        if value.shape[1] != 6:
            raise ValueError(f"CG core-beta input must contain 6 beads, got {value.shape[1]}")
        return value
    if resolved == "implicit":
        required = int(IMPLICIT_OPENMM_CORE_BETA_INDICES.max()) + 1
        if value.shape[1] < required:
            raise ValueError(f"Implicit OpenMM input needs at least {required} atoms, got {value.shape[1]}")
        return value[:, IMPLICIT_OPENMM_CORE_BETA_INDICES, :]
    raise ValueError(f"Unknown Ala2 coordinate variant: {variant!r}")


def _unit_scale(coordinates: np.ndarray, units: Literal["auto", "nm", "angstrom"]) -> float:
    if units == "nm":
        return 1.0
    if units == "angstrom":
        return 0.1
    if units != "auto":
        raise ValueError(f"Unknown coordinate units: {units!r}")
    bonds = np.linalg.norm(
        coordinates[:, CORE_BETA_BONDS[:, 0], :] - coordinates[:, CORE_BETA_BONDS[:, 1], :],
        axis=-1,
    )
    finite = bonds[np.isfinite(bonds)]
    if finite.size == 0:
        raise ValueError("Cannot infer coordinate units from non-finite bonds")
    # Core-beta covalent bonds are about 0.13--0.16 nm or 1.3--1.6 Angstrom.
    return 0.1 if float(np.median(finite)) > 0.5 else 1.0


def prepare_core_beta_coordinates(
    coordinates: Any,
    *,
    variant: Literal["auto", "cg", "implicit"] = "auto",
    units: Literal["auto", "nm", "angstrom"] = "auto",
) -> np.ndarray:
    mapped = map_core_beta(coordinates, variant=variant)
    return mapped * _unit_scale(mapped, units)


def deterministic_subsample(coordinates: np.ndarray, maximum: int | None) -> np.ndarray:
    if maximum is None or coordinates.shape[0] <= maximum:
        return coordinates
    if maximum <= 0:
        raise ValueError("maximum sample count must be positive")
    indices = np.linspace(0, coordinates.shape[0] - 1, num=maximum, dtype=np.int64)
    return coordinates[np.unique(indices)]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {name: float("nan") for name in ("q01", "q50", "q99")}
    q01, q50, q99 = np.quantile(finite, (0.01, 0.50, 0.99))
    return {"q01": float(q01), "q50": float(q50), "q99": float(q99)}


def _bms_cb_torsion_metrics(coordinates_nm: np.ndarray) -> dict[str, Any]:
    """Return the exact upstream-BMS CA--N--C--CB improper diagnostics.

    The operation order, the ``1e-10`` norm regularizer, and the index order
    match :func:`cg_bms_jax.potential.restraint.torsion_angle`.  Keeping this
    NumPy implementation local avoids importing JAX in the lightweight audit
    command.
    """

    coordinates = prepare_core_beta_coordinates(coordinates_nm, variant="cg", units="nm")
    selected = coordinates[:, np.asarray(BMS_CB_IMPROPER_INDICES), :]
    p0, p1, p2, p3 = (selected[:, index, :] for index in range(4))
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.sqrt(np.sum(np.square(b1), axis=-1, keepdims=True) + 1.0e-10)
    v = b0 - b1 * np.sum(b0 * b1, axis=-1, keepdims=True)
    w = b2 - b1 * np.sum(b2 * b1, axis=-1, keepdims=True)
    angle = np.arctan2(
        np.sum(np.cross(b1, v) * w, axis=-1),
        np.sum(v * w, axis=-1),
    )
    finite = np.isfinite(angle) & np.all(np.isfinite(coordinates), axis=(1, 2))
    finite_angle = angle[finite]
    low, high = BMS_CB_FLAT_BOTTOM_DEGREES
    angle_degrees = np.rad2deg(finite_angle)
    if finite_angle.size:
        positive_fraction = float(np.mean(finite_angle > 0.0))
        flat_fraction = float(np.mean((angle_degrees >= low) & (angle_degrees <= high)))
    else:
        positive_fraction = float("nan")
        flat_fraction = float("nan")
    return {
        "indices": list(BMS_CB_IMPROPER_INDICES),
        "finite_fraction": float(np.mean(finite)),
        "angle_degrees": _quantiles(np.rad2deg(angle)),
        "positive_fraction": positive_fraction,
        "flat_bottom_degrees": [low, high],
        "flat_bottom_fraction": flat_fraction,
    }


def _energy_component_metrics(
    values_kj_mol: Any,
    *,
    expected_length: int,
    kT_kj_mol: float,
    zero_threshold_kj_mol: float | None = None,
) -> dict[str, Any]:
    energy = np.asarray(values_kj_mol, dtype=np.float64).reshape(-1)
    if energy.shape[0] != expected_length:
        raise ValueError("energy length does not match proposal coordinates")
    finite = np.isfinite(energy)
    result: dict[str, Any] = {"finite_fraction": float(np.mean(finite))}
    if np.any(finite):
        if not kT_kj_mol > 0.0:
            raise ValueError("kT_kj_mol must be positive")
        reduced_stats = _quantiles(energy[finite] / float(kT_kj_mol))
        result["reduced_energy"] = reduced_stats
        result["q99_minus_median_kT"] = float(
            reduced_stats["q99"] - reduced_stats["q50"]
        )
        if zero_threshold_kj_mol is not None:
            result["flat_zero_threshold_kj_mol"] = float(zero_threshold_kj_mol)
            result["flat_zero_fraction"] = float(
                np.mean(np.abs(energy[finite]) <= zero_threshold_kj_mol)
            )
    elif zero_threshold_kj_mol is not None:
        result["flat_zero_threshold_kj_mol"] = float(zero_threshold_kj_mol)
    return result


def _graph_metrics(pair_distances: np.ndarray, cutoff: float) -> dict[str, float]:
    batch = pair_distances.shape[0]
    adjacency = np.zeros((batch, 6, 6), dtype=bool)
    adjacency[:, PAIR_INDICES[:, 0], PAIR_INDICES[:, 1]] = pair_distances < cutoff
    adjacency[:, PAIR_INDICES[:, 1], PAIR_INDICES[:, 0]] = pair_distances < cutoff
    isolated = np.sum(adjacency, axis=-1) == 0
    diagonal = np.arange(6)
    adjacency[:, diagonal, diagonal] = True
    reach = adjacency
    for node in range(6):
        reach = reach | (reach[:, :, node, None] & reach[:, None, node, :])
    return {
        "cutoff_nm": float(cutoff),
        "connected_frame_fraction": float(np.mean(np.all(reach, axis=(1, 2)))),
        "isolated_bead_fraction": float(np.mean(isolated)),
        "frames_with_isolated_bead_fraction": float(np.mean(np.any(isolated, axis=1))),
    }


def _connected_fraction(pair_distances: np.ndarray, cutoff: float) -> float:
    return _graph_metrics(pair_distances, cutoff)["connected_frame_fraction"]


def carbon_permutation_metrics(
    coordinates_nm: Any,
    *,
    reference_bond_lengths_nm: Any | None = None,
) -> dict[str, Any]:
    """Diagnose whether carbon-label permutations explain bad fixed-topology bonds.

    Nitrogen bead labels remain fixed.  For each frame, all ``4!`` assignments
    of the four carbon coordinates to the fixed core-beta topology are tested.
    These are diagnostic metrics only: relabelling never changes the proposal
    archive and is deliberately not used by any physical-stability gate.

    When reference bond lengths are supplied, the best per-frame maximum and
    mean relative errors are also reported.  Otherwise the two absolute bond
    cutoff fractions remain well-defined and the reference-error fields are
    omitted.
    """

    coordinates = prepare_core_beta_coordinates(coordinates_nm, variant="cg", units="nm")
    finite = coordinates[np.all(np.isfinite(coordinates), axis=(1, 2))]
    if finite.shape[0] == 0:
        raise ValueError("Proposal has no finite coordinate frame")

    reference = None
    if reference_bond_lengths_nm is not None:
        reference = np.asarray(reference_bond_lengths_nm, dtype=np.float64).reshape(-1)
        if reference.shape != (len(CORE_BETA_BONDS),):
            raise ValueError(
                "reference_bond_lengths_nm must contain one value for each of the five bonds"
            )
        if not np.all(np.isfinite(reference)) or np.any(reference <= 0.0):
            raise ValueError("reference bond lengths must be finite and positive")

    below_020 = np.zeros(finite.shape[0], dtype=bool)
    below_025 = np.zeros(finite.shape[0], dtype=bool)
    best_max_error = np.full(finite.shape[0], np.inf, dtype=np.float64)
    best_mean_error = np.full(finite.shape[0], np.inf, dtype=np.float64)

    identity = np.arange(6, dtype=np.int64)
    for permutation in CARBON_BEAD_PERMUTATIONS:
        assignment = identity.copy()
        assignment[CARBON_BEAD_INDICES] = permutation
        candidate = finite[:, assignment, :]
        lengths = np.linalg.norm(
            candidate[:, CORE_BETA_BONDS[:, 0], :]
            - candidate[:, CORE_BETA_BONDS[:, 1], :],
            axis=-1,
        )
        below_020 |= np.all(lengths < 0.20, axis=1)
        below_025 |= np.all(lengths < 0.25, axis=1)
        if reference is not None:
            relative_error = np.abs(lengths / reference[None, :] - 1.0)
            best_max_error = np.minimum(best_max_error, np.max(relative_error, axis=1))
            best_mean_error = np.minimum(best_mean_error, np.mean(relative_error, axis=1))

    result: dict[str, Any] = {
        "carbon_bead_indices": CARBON_BEAD_INDICES.tolist(),
        "num_permutations": int(CARBON_BEAD_PERMUTATIONS.shape[0]),
        "all_bonds_below_0p20_nm_fraction": float(np.mean(below_020)),
        "all_bonds_below_0p25_nm_fraction": float(np.mean(below_025)),
    }
    if reference is not None:
        result["reference_bond_lengths_nm"] = {
            name: float(value) for name, value in zip(CORE_BETA_BOND_NAMES, reference, strict=True)
        }
        result["best_max_relative_error"] = {
            **_quantiles(best_max_error),
            "mean": float(np.mean(best_max_error)),
        }
        result["best_mean_relative_error"] = {
            **_quantiles(best_mean_error),
            "mean": float(np.mean(best_mean_error)),
        }
    return result


def geometry_metrics(
    coordinates_nm: Any,
    *,
    expected_chirality_sign: int = 1,
    bond_all_max_nm: float = 0.20,
    collision_distance_nm: float = 0.08,
    connectivity_cutoff_nm: float = 0.50,
) -> dict[str, Any]:
    coordinates = prepare_core_beta_coordinates(coordinates_nm, variant="cg", units="nm")
    finite_mask = np.all(np.isfinite(coordinates), axis=(1, 2))
    finite = coordinates[finite_mask]
    if finite.shape[0] == 0:
        raise ValueError("Proposal has no finite coordinate frame")

    bond_lengths = np.linalg.norm(
        finite[:, CORE_BETA_BONDS[:, 0], :] - finite[:, CORE_BETA_BONDS[:, 1], :],
        axis=-1,
    )
    pair_distances = np.linalg.norm(
        finite[:, PAIR_INDICES[:, 0], :] - finite[:, PAIR_INDICES[:, 1], :],
        axis=-1,
    )
    centred = finite - np.mean(finite, axis=1, keepdims=True)
    rg = np.sqrt(np.mean(np.sum(np.square(centred), axis=-1), axis=-1))

    ca = finite[:, 2, :]
    n_vector = finite[:, 1, :] - ca
    c_vector = finite[:, 4, :] - ca
    cb_vector = finite[:, 3, :] - ca
    chiral_volume = np.einsum("bi,bi->b", np.cross(n_vector, c_vector), cb_vector)
    sign = 1 if expected_chirality_sign >= 0 else -1

    bond_statistics = {
        name: _quantiles(bond_lengths[:, index])
        for index, name in enumerate(CORE_BETA_BOND_NAMES)
    }
    graph_metrics = {
        "mace_0p5_nm": _graph_metrics(pair_distances, MACE_GRAPH_CUTOFF_NM),
        "painn_0p8_nm": _graph_metrics(pair_distances, PAINN_GRAPH_CUTOFF_NM),
    }
    return {
        "num_frames": int(coordinates.shape[0]),
        "num_finite_frames": int(finite.shape[0]),
        "finite_coordinate_fraction": float(np.mean(finite_mask)),
        "bond_lengths_nm": bond_statistics,
        "bond_q01_min_nm": float(np.min(np.quantile(bond_lengths, 0.01, axis=0))),
        "bond_q99_max_nm": float(np.max(np.quantile(bond_lengths, 0.99, axis=0))),
        "all_bonds_below_limit_fraction": float(
            np.mean(np.all(bond_lengths < bond_all_max_nm, axis=1))
        ),
        "bond_all_limit_nm": float(bond_all_max_nm),
        "rg_nm": _quantiles(rg),
        "chiral_volume_nm3": _quantiles(chiral_volume),
        "expected_chirality_sign": sign,
        "chirality_match_fraction": float(np.mean(sign * chiral_volume > 0.0)),
        "chirality_degenerate_fraction": float(np.mean(np.abs(chiral_volume) <= 1.0e-12)),
        "collision_fraction": float(
            np.mean(np.any(pair_distances < collision_distance_nm, axis=1))
        ),
        "pair_collision_fraction": float(np.mean(pair_distances < collision_distance_nm)),
        "collision_distance_nm": float(collision_distance_nm),
        "connectivity_fraction": _connected_fraction(pair_distances, connectivity_cutoff_nm),
        "connectivity_cutoff_nm": float(connectivity_cutoff_nm),
        "neighbor_graphs": graph_metrics,
        "bms_cb_improper": _bms_cb_torsion_metrics(coordinates),
    }


def _gate(value: float, criterion: str, threshold: float, passed: bool) -> dict[str, Any]:
    return {
        "value": float(value),
        "criterion": criterion,
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def _reference_sign(reference_nm: np.ndarray) -> int:
    ca = reference_nm[:, 2, :]
    volume = np.einsum(
        "bi,bi->b",
        np.cross(reference_nm[:, 1, :] - ca, reference_nm[:, 4, :] - ca),
        reference_nm[:, 3, :] - ca,
    )
    finite = volume[np.isfinite(volume)]
    if finite.size == 0 or float(np.median(finite)) == 0.0:
        raise ValueError("Reference chirality is non-finite or degenerate")
    return 1 if float(np.median(finite)) > 0.0 else -1


def audit_ala2_proposal(
    coordinates_nm: Any,
    *,
    energies_kj_mol: Any | None = None,
    pmf_energies_kj_mol: Any | None = None,
    cb_energies_kj_mol: Any | None = None,
    target_energies_kj_mol: Any | None = None,
    component_energies_kj_mol: Mapping[str, Any] | None = None,
    archive_valid_mask: Any | None = None,
    reference_nm: Any | None = None,
    kT_kj_mol: float = 2.494338785445972,
    ucb_zero_threshold_kj_mol: float = DEFAULT_UCB_ZERO_THRESHOLD_KJ_MOL,
    thresholds: ProposalGeometryThresholds | None = None,
) -> dict[str, Any]:
    """Audit one physical-coordinate proposal archive and return JSON data."""

    limits = ProposalGeometryThresholds() if thresholds is None else thresholds
    proposal = prepare_core_beta_coordinates(coordinates_nm, variant="cg", units="nm")
    reference = None
    if reference_nm is not None:
        reference = prepare_core_beta_coordinates(reference_nm, variant="cg", units="nm")
    expected_sign = 1 if reference is None else _reference_sign(reference)
    metric_options = {
        "expected_chirality_sign": expected_sign,
        "bond_all_max_nm": limits.bond_all_max_nm,
        "collision_distance_nm": limits.collision_distance_nm,
        "connectivity_cutoff_nm": limits.connectivity_cutoff_nm,
    }
    metrics = geometry_metrics(proposal, **metric_options)
    reference_metrics = None if reference is None else geometry_metrics(reference, **metric_options)
    reference_bond_lengths = None
    if reference_metrics is not None:
        reference_bond_lengths = np.asarray(
            [reference_metrics["bond_lengths_nm"][name]["q50"] for name in CORE_BETA_BOND_NAMES],
            dtype=np.float64,
        )
    metrics["carbon_permutation_diagnostics"] = carbon_permutation_metrics(
        proposal,
        reference_bond_lengths_nm=reference_bond_lengths,
    )

    gates: dict[str, dict[str, Any]] = {}
    finite_fraction = float(metrics["finite_coordinate_fraction"])
    gates["finite_coordinates"] = _gate(finite_fraction, ">=", 1.0, finite_fraction == 1.0)
    bond_low = float(metrics["bond_q01_min_nm"])
    bond_high = float(metrics["bond_q99_max_nm"])
    gates["bond_q01_lower"] = _gate(
        bond_low, ">=", limits.bond_quantile_low_nm, bond_low >= limits.bond_quantile_low_nm
    )
    gates["bond_q99_upper"] = _gate(
        bond_high, "<=", limits.bond_quantile_high_nm, bond_high <= limits.bond_quantile_high_nm
    )
    all_bonds = float(metrics["all_bonds_below_limit_fraction"])
    gates["all_bonds_below_limit"] = _gate(
        all_bonds, ">=", limits.bond_all_min_fraction, all_bonds >= limits.bond_all_min_fraction
    )
    rg_median = float(metrics["rg_nm"]["q50"])
    gates["rg_median_lower"] = _gate(
        rg_median, ">=", limits.rg_median_min_nm, rg_median >= limits.rg_median_min_nm
    )
    gates["rg_median_upper"] = _gate(
        rg_median, "<=", limits.rg_median_max_nm, rg_median <= limits.rg_median_max_nm
    )
    chirality = float(metrics["chirality_match_fraction"])
    gates["chirality_match"] = _gate(
        chirality,
        ">=",
        limits.chirality_match_min_fraction,
        chirality >= limits.chirality_match_min_fraction,
    )
    collision = float(metrics["collision_fraction"])
    gates["collision"] = _gate(
        collision, "<=", limits.collision_max_fraction, collision <= limits.collision_max_fraction
    )
    connectivity = float(metrics["connectivity_fraction"])
    gates["connectivity"] = _gate(
        connectivity,
        ">=",
        limits.connectivity_min_fraction,
        connectivity >= limits.connectivity_min_fraction,
    )

    if archive_valid_mask is not None:
        valid = np.asarray(archive_valid_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != proposal.shape[0]:
            raise ValueError("valid_mask length does not match proposal coordinates")
        valid_fraction = float(np.mean(valid))
        metrics["archive_valid_fraction"] = valid_fraction
        gates["archive_valid"] = _gate(
            valid_fraction,
            ">=",
            limits.archive_valid_min_fraction,
            valid_fraction >= limits.archive_valid_min_fraction,
        )

    if reference_metrics is not None:
        proposal_medians = np.asarray(
            [metrics["bond_lengths_nm"][name]["q50"] for name in CORE_BETA_BOND_NAMES]
        )
        reference_medians = np.asarray(
            [reference_metrics["bond_lengths_nm"][name]["q50"] for name in CORE_BETA_BOND_NAMES]
        )
        bond_error = float(np.max(np.abs(proposal_medians / reference_medians - 1.0)))
        rg_reference = float(reference_metrics["rg_nm"]["q50"])
        rg_error = float(abs(rg_median / rg_reference - 1.0))
        metrics["bond_median_max_relative_error"] = bond_error
        metrics["rg_median_relative_error"] = rg_error
        gates["bond_median_reference_error"] = _gate(
            bond_error,
            "<=",
            limits.bond_median_max_relative_error,
            bond_error <= limits.bond_median_max_relative_error,
        )
        gates["rg_median_reference_error"] = _gate(
            rg_error,
            "<=",
            limits.rg_median_max_relative_error,
            rg_error <= limits.rg_median_max_relative_error,
        )

    # ``energies_kj_mol`` is the legacy API and represents the archive's U.
    # New archives expose all three components explicitly.  PMF energy remains
    # the primary stability diagnostic because a large, deliberately stiff
    # U_CB restraint tail must not obscure whether the learned PMF support is
    # improving.  U_target and U_CB are still reported independently.
    if target_energies_kj_mol is None:
        target_energies_kj_mol = energies_kj_mol
    component_values: dict[str, Any] = (
        {} if component_energies_kj_mol is None else dict(component_energies_kj_mol)
    )
    if pmf_energies_kj_mol is not None:
        component_values.setdefault("U_pmf", pmf_energies_kj_mol)
    if cb_energies_kj_mol is not None:
        component_values.setdefault("U_cb", cb_energies_kj_mol)
    if target_energies_kj_mol is not None:
        component_values.setdefault("U_target", target_energies_kj_mol)
    zero_energy_components = {
        "U_bond",
        "U_angle",
        "U_repulsion",
        "U_support",
        "U_cb",
    }
    components = {
        name: _energy_component_metrics(
            values,
            expected_length=proposal.shape[0],
            kT_kj_mol=kT_kj_mol,
            zero_threshold_kj_mol=(
                ucb_zero_threshold_kj_mol if name in zero_energy_components else None
            ),
        )
        for name, values in component_values.items()
    }
    if components:
        metrics["energy_components"] = components
        primary_name = "U_pmf" if "U_pmf" in components else "U_target"
        primary = components[primary_name]
        metrics["energy_primary_component"] = primary_name
        energy_finite_fraction = float(primary["finite_fraction"])
        metrics["energy_finite_fraction"] = energy_finite_fraction
        gates["finite_energy"] = _gate(
            energy_finite_fraction, ">=", 1.0, energy_finite_fraction == 1.0
        )
        if "reduced_energy" in primary:
            energy_stats = primary["reduced_energy"]
            tail = float(primary["q99_minus_median_kT"])
            metrics["reduced_energy"] = energy_stats
            metrics["energy_q99_minus_median_kT"] = tail
            gates["energy_tail"] = _gate(
                tail, "<=", limits.energy_tail_max_kT, tail <= limits.energy_tail_max_kT
            )

    overall = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "schema_version": 3,
        "overall_pass": overall,
        "thresholds": asdict(limits),
        "metrics": metrics,
        "reference_metrics": reference_metrics,
        "gates": gates,
    }


def load_npz_field(path: str | Path, names: tuple[str, ...], *, required: bool) -> np.ndarray | None:
    with np.load(Path(path), allow_pickle=False) as archive:
        for name in names:
            if name in archive.files:
                return np.asarray(archive[name])
    if required:
        raise KeyError(f"{path} contains none of the required fields {names}")
    return None


def audit_npz(
    sample_path: str | Path,
    *,
    reference_path: str | Path | None = None,
    reference_variant: Literal["auto", "cg", "implicit"] = "auto",
    reference_units: Literal["auto", "nm", "angstrom"] = "auto",
    reference_max_frames: int | None = 100_000,
    kT_kj_mol: float = 2.494338785445972,
    thresholds: ProposalGeometryThresholds | None = None,
) -> dict[str, Any]:
    proposal_raw = load_npz_field(sample_path, ("R", "positions", "coordinates"), required=True)
    proposal = prepare_core_beta_coordinates(proposal_raw, variant="cg", units="nm")
    pmf_energy = load_npz_field(sample_path, ("U_pmf",), required=False)
    cb_energy = load_npz_field(sample_path, ("U_cb",), required=False)
    component_fields = (
        "U_pmf_raw",
        "U_pmf_effective",
        "U_pmf_scaled",
        "U_pmf_target_contribution",
        "U_pmf_floor_delta",
        "U_pmf_transform_delta",
        "U_pmf_gated",
        "U_bond",
        "U_angle",
        "U_repulsion",
        "U_support",
    )
    component_energies = {
        name: value
        for name in component_fields
        if (value := load_npz_field(sample_path, (name,), required=False)) is not None
    }
    target_energy = load_npz_field(
        sample_path,
        ("U_target", "U", "energy", "energies"),
        required=False,
    )
    valid = load_npz_field(sample_path, ("valid_mask", "valid"), required=False)
    reference = None
    if reference_path is not None:
        reference_raw = load_npz_field(reference_path, ("R", "positions", "coordinates"), required=True)
        reference = prepare_core_beta_coordinates(
            reference_raw,
            variant=reference_variant,
            units=reference_units,
        )
        reference = deterministic_subsample(reference, reference_max_frames)
    return audit_ala2_proposal(
        proposal,
        energies_kj_mol=target_energy,
        pmf_energies_kj_mol=pmf_energy,
        cb_energies_kj_mol=cb_energy,
        target_energies_kj_mol=target_energy,
        component_energies_kj_mol=component_energies,
        archive_valid_mask=valid,
        reference_nm=reference,
        kT_kj_mol=kT_kj_mol,
        thresholds=thresholds,
    )


__all__ = [
    "BMS_CB_FLAT_BOTTOM_DEGREES",
    "BMS_CB_IMPROPER_INDICES",
    "CARBON_BEAD_INDICES",
    "CARBON_BEAD_PERMUTATIONS",
    "CORE_BETA_BONDS",
    "CORE_BETA_BOND_NAMES",
    "DEFAULT_UCB_ZERO_THRESHOLD_KJ_MOL",
    "IMPLICIT_OPENMM_CORE_BETA_INDICES",
    "MACE_GRAPH_CUTOFF_NM",
    "PAINN_GRAPH_CUTOFF_NM",
    "ProposalGeometryThresholds",
    "audit_ala2_proposal",
    "audit_npz",
    "carbon_permutation_metrics",
    "deterministic_subsample",
    "geometry_metrics",
    "map_core_beta",
    "prepare_core_beta_coordinates",
]
