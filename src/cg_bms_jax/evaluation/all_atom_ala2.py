"""Full-atom Ala2 diagnostics in the public BMS 22-atom ordering.

The public BridgeMatchingSampler Ala2 experiment uses Ac-Ala-NHMe with the
following zero-based backbone indices::

    ACE-C=4, ALA-N=6, ALA-CA=8, ALA-C=14, NME-N=16

The module is intentionally NumPy-only.  It evaluates proposal-only SDE
archives as well as PF-ODE archives carrying formal importance weights, but it
does not manufacture a likelihood for stochastic SDE samples.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .ala2 import compute_dihedral
from .metrics import (
    distribution_metrics,
    importance_weight_diagnostics,
    resolve_weights,
)

AA_ALA2_NUM_ATOMS = 22
AA_ALA2_PHI_INDICES = (4, 6, 8, 14)
AA_ALA2_PSI_INDICES = (6, 8, 14, 16)
AA_ALA2_CB_IMPROPER_INDICES = (8, 6, 14, 10)
AA_ALA2_HA_IMPROPER_INDICES = (8, 6, 14, 9)

# Covalent graph of the upstream ``md_data/ala2/ala2_ref.pdb``.  Explicitly
# listing the bonds makes the geometry diagnostics independent of MDTraj and
# keeps atom-order mistakes visible in the JSON contract.
AA_ALA2_BONDS = np.asarray(
    (
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (4, 5),
        (4, 6),
        (6, 7),
        (6, 8),
        (8, 9),
        (8, 10),
        (8, 14),
        (10, 11),
        (10, 12),
        (10, 13),
        (14, 15),
        (14, 16),
        (16, 17),
        (16, 18),
        (18, 19),
        (18, 20),
        (18, 21),
    ),
    dtype=np.int64,
)

AA_ALA2_BOND_NAMES = (
    "ACE-H1--ACE-CH3",
    "ACE-CH3--ACE-H2",
    "ACE-CH3--ACE-H3",
    "ACE-CH3--ACE-C",
    "ACE-C--ACE-O",
    "ACE-C--ALA-N",
    "ALA-N--ALA-H",
    "ALA-N--ALA-CA",
    "ALA-CA--ALA-HA",
    "ALA-CA--ALA-CB",
    "ALA-CA--ALA-C",
    "ALA-CB--ALA-HB1",
    "ALA-CB--ALA-HB2",
    "ALA-CB--ALA-HB3",
    "ALA-C--ALA-O",
    "ALA-C--NME-N",
    "NME-N--NME-H",
    "NME-N--NME-C",
    "NME-C--NME-H1",
    "NME-C--NME-H2",
    "NME-C--NME-H3",
)


def _angle_indices() -> np.ndarray:
    neighbours: list[list[int]] = [[] for _ in range(AA_ALA2_NUM_ATOMS)]
    for left, right in AA_ALA2_BONDS:
        neighbours[int(left)].append(int(right))
        neighbours[int(right)].append(int(left))
    triples: list[tuple[int, int, int]] = []
    for centre, bonded in enumerate(neighbours):
        triples.extend((left, centre, right) for left, right in combinations(sorted(bonded), 2))
    return np.asarray(triples, dtype=np.int64)


AA_ALA2_ANGLES = _angle_indices()


def _collision_pairs() -> np.ndarray:
    """Return pairs separated by more than two bonds in the covalent graph."""

    excluded = {tuple(sorted(map(int, bond))) for bond in AA_ALA2_BONDS}
    excluded.update(
        tuple(sorted((int(left), int(right))))
        for left, _centre, right in AA_ALA2_ANGLES
    )
    return np.asarray(
        [
            (left, right)
            for left in range(AA_ALA2_NUM_ATOMS)
            for right in range(left + 1, AA_ALA2_NUM_ATOMS)
            if (left, right) not in excluded
        ],
        dtype=np.int64,
    )


AA_ALA2_COLLISION_PAIRS = _collision_pairs()


def _load(source: str | Path | Mapping[str, Any]) -> dict[str, np.ndarray]:
    if isinstance(source, Mapping):
        return {str(key): np.asarray(value) for key, value in source.items()}
    with np.load(source, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _coordinates(data: Mapping[str, Any]) -> np.ndarray:
    for key in ("R", "positions", "coordinates", "samples"):
        if key in data:
            value = np.asarray(data[key], dtype=np.float64)
            break
    else:
        raise KeyError("Full-atom Ala2 input requires R, positions, coordinates, or samples")
    if value.ndim == 2 and value.shape[-1] == 3 * AA_ALA2_NUM_ATOMS:
        value = value.reshape((-1, AA_ALA2_NUM_ATOMS, 3))
    if value.ndim != 3 or value.shape[1:] != (AA_ALA2_NUM_ATOMS, 3):
        raise ValueError(
            "Full-atom Ala2 coordinates must have shape (B,22,3) or (B,66), "
            f"got {value.shape}"
        )
    if value.shape[0] == 0:
        raise ValueError("Full-atom Ala2 archive is empty")
    return value


def _to_nm(
    coordinates: np.ndarray,
    units: Literal["auto", "nm", "angstrom"],
) -> tuple[np.ndarray, str]:
    if units == "nm":
        return coordinates, "nm"
    if units == "angstrom":
        return coordinates * 0.1, "angstrom"
    if units != "auto":
        raise ValueError("units must be 'auto', 'nm', or 'angstrom'")
    bonds = np.linalg.norm(
        coordinates[:, AA_ALA2_BONDS[:, 0]] - coordinates[:, AA_ALA2_BONDS[:, 1]],
        axis=-1,
    )
    finite = bonds[np.isfinite(bonds)]
    if finite.size == 0:
        raise ValueError("Cannot infer units from non-finite coordinates")
    inferred = "angstrom" if float(np.median(finite)) > 0.5 else "nm"
    return (coordinates * 0.1, inferred) if inferred == "angstrom" else (coordinates, inferred)


def all_atom_ala2_dihedrals(coordinates: Any) -> np.ndarray:
    """Return ``(phi, psi)`` using the official BMS atom order."""

    value = np.asarray(coordinates, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (AA_ALA2_NUM_ATOMS, 3):
        raise ValueError(f"Expected (B,22,3), got {value.shape}")
    phi = compute_dihedral(value[:, AA_ALA2_PHI_INDICES, :])
    psi = compute_dihedral(value[:, AA_ALA2_PSI_INDICES, :])
    return np.stack((phi, psi), axis=-1)


def all_atom_ala2_impropers(coordinates: Any) -> np.ndarray:
    """Return upstream-BMS CB and HA improper torsions."""

    value = np.asarray(coordinates, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (AA_ALA2_NUM_ATOMS, 3):
        raise ValueError(f"Expected (B,22,3), got {value.shape}")
    cb = compute_dihedral(value[:, AA_ALA2_CB_IMPROPER_INDICES, :])
    ha = compute_dihedral(value[:, AA_ALA2_HA_IMPROPER_INDICES, :])
    return np.stack((cb, ha), axis=-1)


def all_atom_ala2_bond_lengths(coordinates_nm: Any) -> np.ndarray:
    value = np.asarray(coordinates_nm, dtype=np.float64)
    return np.linalg.norm(
        value[:, AA_ALA2_BONDS[:, 0]] - value[:, AA_ALA2_BONDS[:, 1]],
        axis=-1,
    )


def all_atom_ala2_angles(coordinates: Any) -> np.ndarray:
    value = np.asarray(coordinates, dtype=np.float64)
    left = value[:, AA_ALA2_ANGLES[:, 0]] - value[:, AA_ALA2_ANGLES[:, 1]]
    right = value[:, AA_ALA2_ANGLES[:, 2]] - value[:, AA_ALA2_ANGLES[:, 1]]
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    cosine = np.sum(left * right, axis=-1) / np.maximum(
        denominator, np.finfo(np.float64).tiny
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _normalised_weights(weights: np.ndarray | None, size: int) -> np.ndarray | None:
    if weights is None:
        return None
    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value.shape != (size,):
        raise ValueError("Weight count does not match coordinate count")
    value = np.where(np.isfinite(value) & (value >= 0.0), value, 0.0)
    total = float(np.sum(value))
    if not total > 0.0:
        raise ValueError("Weights contain no positive finite mass")
    return value / total


def _weighted_quantile(
    values: Any,
    quantiles: tuple[float, ...],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(value)
    if weights is None:
        return np.quantile(value[finite], quantiles) if np.any(finite) else np.full(len(quantiles), np.nan)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weight.shape != value.shape:
        raise ValueError("Expanded weights do not match values")
    finite &= np.isfinite(weight) & (weight >= 0.0)
    if not np.any(finite) or float(np.sum(weight[finite])) <= 0.0:
        return np.full(len(quantiles), np.nan)
    value = value[finite]
    weight = weight[finite]
    order = np.argsort(value)
    value = value[order]
    weight = weight[order]
    centres = (np.cumsum(weight) - 0.5 * weight) / np.sum(weight)
    return np.interp(quantiles, centres, value, left=value[0], right=value[-1])


def _summary(values: Any, weights: np.ndarray | None = None) -> dict[str, float]:
    q01, q50, q99 = _weighted_quantile(values, (0.01, 0.5, 0.99), weights)
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "finite_fraction": float(np.mean(np.isfinite(value))),
        "q01": float(q01),
        "q50": float(q50),
        "q99": float(q99),
    }


def _expanded_weights(weights: np.ndarray | None, width: int) -> np.ndarray | None:
    if weights is None:
        return None
    return np.repeat(weights, width)


def _improper_report(
    impropers: np.ndarray,
    weights: np.ndarray | None,
) -> dict[str, Any]:
    cb = np.rad2deg(impropers[:, 0])
    ha = np.rad2deg(impropers[:, 1])

    def report(values: np.ndarray, low: float, high: float, expected_sign: int) -> dict[str, Any]:
        finite = np.isfinite(values)
        if weights is None:
            denominator = float(np.sum(finite))
            sign_mass = float(np.sum(finite & (expected_sign * values > 0.0)))
            flat_mass = float(np.sum(finite & (values >= low) & (values <= high)))
        else:
            denominator = float(np.sum(weights[finite]))
            sign_mass = float(np.sum(weights[finite & (expected_sign * values > 0.0)]))
            flat_mass = float(np.sum(weights[finite & (values >= low) & (values <= high)]))
        return {
            "indices": None,
            "expected_sign": "positive" if expected_sign > 0 else "negative",
            "flat_bottom_degrees": [low, high],
            "expected_sign_fraction": sign_mass / denominator if denominator > 0.0 else float("nan"),
            "flat_bottom_fraction": flat_mass / denominator if denominator > 0.0 else float("nan"),
            "angle_degrees": _summary(values, weights),
        }

    cb_report = report(cb, 10.0, 60.0, 1)
    cb_report["indices"] = list(AA_ALA2_CB_IMPROPER_INDICES)
    ha_report = report(ha, -60.0, -10.0, -1)
    ha_report["indices"] = list(AA_ALA2_HA_IMPROPER_INDICES)
    return {"CB": cb_report, "HA": ha_report}


def _geometry_report(
    coordinates_nm: np.ndarray,
    weights: np.ndarray | None,
    collision_threshold_nm: float,
) -> dict[str, Any]:
    batch = coordinates_nm.shape[0]
    bonds = all_atom_ala2_bond_lengths(coordinates_nm)
    angles = np.rad2deg(all_atom_ala2_angles(coordinates_nm))
    impropers = all_atom_ala2_impropers(coordinates_nm)
    nonlocal_distances = np.linalg.norm(
        coordinates_nm[:, AA_ALA2_COLLISION_PAIRS[:, 0]]
        - coordinates_nm[:, AA_ALA2_COLLISION_PAIRS[:, 1]],
        axis=-1,
    )
    collision = nonlocal_distances < float(collision_threshold_nm)
    frame_collision = np.any(collision, axis=1)
    if weights is None:
        frame_collision_fraction = float(np.mean(frame_collision))
        pair_collision_fraction = float(np.mean(collision))
    else:
        frame_collision_fraction = float(np.sum(weights * frame_collision))
        pair_collision_fraction = float(
            np.sum(weights[:, None] * collision) / collision.shape[1]
        )
    return {
        "num_frames": int(batch),
        "finite_coordinate_fraction": float(
            np.mean(np.all(np.isfinite(coordinates_nm), axis=(1, 2)))
        ),
        "bond_lengths_nm": {
            "aggregate": _summary(bonds, _expanded_weights(weights, bonds.shape[1])),
            "per_bond": {
                name: _summary(bonds[:, index], weights)
                for index, name in enumerate(AA_ALA2_BOND_NAMES)
            },
        },
        "angles_degrees": {
            "num_angle_types": int(angles.shape[1]),
            "aggregate": _summary(angles, _expanded_weights(weights, angles.shape[1])),
        },
        "improper": _improper_report(impropers, weights),
        "collision": {
            "definition": "nonlocal atom pairs separated by more than two covalent bonds",
            "threshold_nm": float(collision_threshold_nm),
            "num_pairs": int(nonlocal_distances.shape[1]),
            "frame_fraction": frame_collision_fraction,
            "pair_fraction": pair_collision_fraction,
            "minimum_distance_nm": _summary(np.min(nonlocal_distances, axis=1), weights),
        },
    }


def _energy(data: Mapping[str, Any]) -> tuple[np.ndarray | None, str | None]:
    for key in ("U_target", "U", "energy_kj_mol", "potential_energy_kj_mol", "energies"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(-1), key
    return None, None


def _weighted_energy_distance(
    proposal: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None,
    *,
    n_quantiles: int = 1000,
) -> dict[str, float]:
    proposal = np.asarray(proposal, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    proposal_finite = np.isfinite(proposal)
    reference = reference[np.isfinite(reference)]
    if reference.size == 0 or not np.any(proposal_finite):
        return {"W1": float("nan"), "W2": float("nan")}
    proposal = proposal[proposal_finite]
    proposal_weights = None if weights is None else weights[proposal_finite]
    quantiles = tuple(np.linspace(0.0, 1.0, n_quantiles))
    proposal_q = _weighted_quantile(proposal, quantiles, proposal_weights)
    reference_q = np.quantile(reference, quantiles)
    difference = proposal_q - reference_q
    return {
        "W1": float(np.mean(np.abs(difference))),
        "W2": float(np.sqrt(np.mean(np.square(difference)))),
    }


def _raw_log_weights(data: Mapping[str, Any]) -> np.ndarray | None:
    for key in ("logw_raw", "logw"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    return None


def _hist(
    axis: Any,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    weights: np.ndarray | None = None,
    bins: int = 80,
    value_range: tuple[float, float] | None = None,
) -> None:
    finite = np.isfinite(values)
    selected_weights = None if weights is None else weights[finite]
    axis.hist(
        values[finite],
        bins=bins,
        range=value_range,
        density=True,
        weights=selected_weights,
        histtype="step",
        color=color,
        label=label,
    )


def _plot_rama(axis: Any, angles: np.ndarray, weights: np.ndarray | None, title: str, kT: float) -> None:
    hist, xedge, yedge = np.histogram2d(
        angles[:, 0],
        angles[:, 1],
        bins=72,
        range=((-np.pi, np.pi), (-np.pi, np.pi)),
        weights=weights,
    )
    probability = hist / max(float(np.sum(hist)), np.finfo(float).tiny)
    fes = -(float(kT) / 4.184) * np.log(np.maximum(probability, np.finfo(float).tiny))
    fes -= np.nanmin(fes)
    axis.pcolormesh(xedge, yedge, fes.T, shading="auto", vmin=0.0, vmax=5.25)
    axis.set(
        title=title,
        xlabel=r"$\phi$",
        ylabel=r"$\psi$",
        xlim=(-np.pi, np.pi),
        ylim=(-np.pi, np.pi),
    )


def evaluate_all_atom_ala2(
    *,
    target: str | Path | Mapping[str, Any],
    sample: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    kT: float = 2.494338785445972,
    target_units: Literal["auto", "nm", "angstrom"] = "auto",
    sample_units: Literal["auto", "nm", "angstrom"] = "auto",
    collision_threshold_nm: float = 0.08,
) -> dict[str, Any]:
    """Write full-atom Ala2 figures and a machine-readable diagnostic report."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not kT > 0.0:
        raise ValueError("kT must be positive")
    target_data = _load(target)
    sample_data = _load(sample)
    target_coordinates_nm, target_units_resolved = _to_nm(
        _coordinates(target_data), target_units
    )
    sample_coordinates_nm, sample_units_resolved = _to_nm(
        _coordinates(sample_data), sample_units
    )
    weights, _resolved_logw, weight_source = resolve_weights(sample_data, kT=kT)
    weights = _normalised_weights(weights, sample_coordinates_nm.shape[0])
    raw_logw = _raw_log_weights(sample_data)
    if raw_logw is not None and raw_logw.shape != (sample_coordinates_nm.shape[0],):
        raise ValueError("logw count does not match coordinates")

    target_angles = all_atom_ala2_dihedrals(target_coordinates_nm)
    sample_angles = all_atom_ala2_dihedrals(sample_coordinates_nm)
    ranges = ((-np.pi, np.pi), (-np.pi, np.pi))
    unweighted_rama = distribution_metrics(target_angles, sample_angles, ranges=ranges)
    weighted_rama = (
        None
        if weights is None
        else distribution_metrics(
            target_angles,
            sample_angles,
            ranges=ranges,
            sample_weights=weights,
        )
    )

    target_energy, target_energy_key = _energy(target_data)
    sample_energy, sample_energy_key = _energy(sample_data)
    if target_energy is not None and target_energy.shape != (target_coordinates_nm.shape[0],):
        raise ValueError("Target energy count does not match coordinates")
    if sample_energy is not None and sample_energy.shape != (sample_coordinates_nm.shape[0],):
        raise ValueError("Sample energy count does not match coordinates")

    energy_report: dict[str, Any] | None = None
    if target_energy is not None and sample_energy is not None:
        energy_report = {
            "units": "kJ/mol",
            "target_key": target_energy_key,
            "sample_key": sample_energy_key,
            "reference": _summary(target_energy),
            "proposal": _summary(sample_energy),
            "reweighted": None if weights is None else _summary(sample_energy, weights),
            "distance_proposal_to_reference": _weighted_energy_distance(
                sample_energy, target_energy, None
            ),
            "distance_reweighted_to_reference": (
                None
                if weights is None
                else _weighted_energy_distance(sample_energy, target_energy, weights)
            ),
        }

    report: dict[str, Any] = {
        "system": "Ac-Ala-NHMe (upstream BMS all-atom Ala2)",
        "coordinate_contract": {
            "num_atoms": AA_ALA2_NUM_ATOMS,
            "ambient_dimension": 66,
            "phi_indices": list(AA_ALA2_PHI_INDICES),
            "psi_indices": list(AA_ALA2_PSI_INDICES),
            "cb_improper_indices": list(AA_ALA2_CB_IMPROPER_INDICES),
            "ha_improper_indices": list(AA_ALA2_HA_IMPROPER_INDICES),
            "target_input_units": target_units_resolved,
            "sample_input_units": sample_units_resolved,
            "geometry_output_units": "nm",
        },
        "weight_source": weight_source,
        "weight_diagnostics": (
            None
            if weights is None
            else importance_weight_diagnostics(weights, logw_raw=raw_logw)
        ),
        "ramachandran": {
            "unweighted": unweighted_rama,
            "reweighted": weighted_rama,
        },
        "geometry": {
            "reference": _geometry_report(
                target_coordinates_nm, None, collision_threshold_nm
            ),
            "proposal": _geometry_report(
                sample_coordinates_nm, None, collision_threshold_nm
            ),
            "reweighted": (
                None
                if weights is None
                else _geometry_report(
                    sample_coordinates_nm, weights, collision_threshold_nm
                )
            ),
        },
        "energy": energy_report,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Primary scientific panel: reference, proposal, reweighted Rama and energy.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    _plot_rama(axes[0], target_angles, None, "MD reference", kT)
    _plot_rama(axes[1], sample_angles, None, "PF/SDE proposal", kT)
    if weights is None:
        axes[2].text(
            0.5,
            0.5,
            "Proposal only\n(no likelihood/weights)",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set(title="Reweighted unavailable", xlabel=r"$\phi$", ylabel=r"$\psi$")
    else:
        _plot_rama(axes[2], sample_angles, weights, "Formal no-clip reweighted", kT)
    if target_energy is None or sample_energy is None:
        axes[3].text(
            0.5,
            0.5,
            "Energy unavailable",
            ha="center",
            va="center",
            transform=axes[3].transAxes,
        )
    else:
        finite_target = target_energy[np.isfinite(target_energy)]
        low, high = np.quantile(finite_target, (0.005, 0.995))
        padding = max(0.05 * float(high - low), np.finfo(float).eps)
        energy_range = (float(low - padding), float(high + padding))
        _hist(
            axes[3],
            target_energy,
            label="MD reference",
            color="C2",
            value_range=energy_range,
        )
        _hist(
            axes[3],
            sample_energy,
            label="Proposal",
            color="C1",
            value_range=energy_range,
        )
        if weights is not None:
            _hist(
                axes[3],
                sample_energy,
                weights=weights,
                label="Reweighted",
                color="C0",
                value_range=energy_range,
            )
        axes[3].set(
            xlabel="Potential energy (kJ/mol)",
            ylabel="Density",
            title="Target 0.5–99.5% energy window",
        )
        axes[3].legend(fontsize=7)
        report["energy"]["plot_window_kj_mol"] = list(energy_range)
        report["energy"]["proposal_outside_plot_window_fraction"] = float(
            np.mean((sample_energy < energy_range[0]) | (sample_energy > energy_range[1]))
        )
    primary_path = output / "ala2_all_atom_rama_energy.png"
    fig.tight_layout()
    fig.savefig(primary_path, dpi=200)
    plt.close(fig)

    # Geometry panel.  The same x limits are used for all distributions in a
    # subplot, so a visually attractive reweighted curve cannot hide collisions.
    target_bonds = all_atom_ala2_bond_lengths(target_coordinates_nm).reshape(-1)
    sample_bonds = all_atom_ala2_bond_lengths(sample_coordinates_nm).reshape(-1)
    target_angle_values = np.rad2deg(all_atom_ala2_angles(target_coordinates_nm)).reshape(-1)
    sample_angle_values = np.rad2deg(all_atom_ala2_angles(sample_coordinates_nm)).reshape(-1)
    target_impropers = np.rad2deg(all_atom_ala2_impropers(target_coordinates_nm))
    sample_impropers = np.rad2deg(all_atom_ala2_impropers(sample_coordinates_nm))
    expanded_bond_weights = _expanded_weights(weights, AA_ALA2_BONDS.shape[0])
    expanded_angle_weights = _expanded_weights(weights, AA_ALA2_ANGLES.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = (
        (target_bonds, sample_bonds, expanded_bond_weights, "Bond length (nm)", None),
        (target_angle_values, sample_angle_values, expanded_angle_weights, "Angle (degrees)", (0.0, 180.0)),
        (target_impropers[:, 0], sample_impropers[:, 0], weights, "CB improper (degrees)", (-180.0, 180.0)),
        (target_impropers[:, 1], sample_impropers[:, 1], weights, "HA improper (degrees)", (-180.0, 180.0)),
        (target_angles[:, 0], sample_angles[:, 0], weights, r"$\phi$", (-np.pi, np.pi)),
        (target_angles[:, 1], sample_angles[:, 1], weights, r"$\psi$", (-np.pi, np.pi)),
    )
    for axis, (target_value, sample_value, panel_weights, xlabel, value_range) in zip(
        axes.reshape(-1), panels, strict=True
    ):
        _hist(axis, target_value, label="MD reference", color="C2", value_range=value_range)
        _hist(axis, sample_value, label="Proposal", color="C1", value_range=value_range)
        if panel_weights is not None:
            _hist(
                axis,
                sample_value,
                weights=panel_weights,
                label="Reweighted",
                color="C0",
                value_range=value_range,
            )
        axis.set(xlabel=xlabel, ylabel="Density")
        axis.legend(fontsize=7)
    geometry_path = output / "ala2_all_atom_geometry.png"
    fig.tight_layout()
    fig.savefig(geometry_path, dpi=200)
    plt.close(fig)

    metrics_path = output / "ala2_all_atom_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    return {
        "metrics": report,
        "metrics_path": str(metrics_path),
        "images": {
            "rama_energy": str(primary_path),
            "geometry": str(geometry_path),
        },
    }

