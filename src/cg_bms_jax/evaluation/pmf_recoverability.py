"""Read-only diagnostics for Ala2 PMF cold-start recoverability.

The BMS terminal target is a score, so a finite PMF value alone does not show
whether a cold endpoint can be pulled back toward the molecular manifold.  The
helpers in this module inspect the two graphs seen by the current experiment
and the instantaneous change of every fixed core-beta bond under an injected
target-score direction.  They never mutate parameters or alter a rollout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .proposal_geometry import (
    CORE_BETA_BOND_NAMES,
    CORE_BETA_BONDS,
    MACE_GRAPH_CUTOFF_NM,
    PAINN_GRAPH_CUTOFF_NM,
    prepare_core_beta_coordinates,
)


def _as_batch(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (6, 3):
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] != (6, 3):
        raise ValueError(f"{label} must have shape (6,3) or (B,6,3), got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{label} is empty")
    return array


def _quantiles(value: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"q01": None, "q50": None, "q99": None, "mean": None}
    q01, q50, q99 = np.quantile(finite, (0.01, 0.50, 0.99))
    return {
        "q01": float(q01),
        "q50": float(q50),
        "q99": float(q99),
        "mean": float(np.mean(finite)),
    }


def _minimum_image(displacement: np.ndarray, box_nm: Any | None) -> np.ndarray:
    if box_nm is None:
        return displacement
    box = np.asarray(box_nm, dtype=np.float64)
    lengths = np.diag(box) if box.shape == (3, 3) else box
    if lengths.shape != (3,) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("box_nm must contain three positive finite box lengths")
    return displacement - lengths * np.round(displacement / lengths)


def neighbor_graph_metrics(
    coordinates_nm: Any,
    *,
    cutoff_nm: float,
    box_nm: Any | None = None,
) -> dict[str, Any]:
    """Report connectivity and isolated beads for one all-pairs cutoff graph.

    Supplying ``box_nm`` applies minimum-image distances, as used by the MACE
    PMF.  Leaving it unset uses direct Cartesian distances, as used by PaiNN.
    """

    if not np.isfinite(cutoff_nm) or cutoff_nm <= 0.0:
        raise ValueError("cutoff_nm must be positive and finite")
    coordinates = _as_batch(coordinates_nm, label="coordinates_nm")
    finite_frames = np.all(np.isfinite(coordinates), axis=(1, 2))
    coordinates = coordinates[finite_frames]
    if coordinates.shape[0] == 0:
        raise ValueError("coordinates_nm contains no finite frame")

    displacement = coordinates[:, :, None, :] - coordinates[:, None, :, :]
    displacement = _minimum_image(displacement, box_nm)
    distance = np.linalg.norm(displacement, axis=-1)
    adjacency = distance < float(cutoff_nm)
    diagonal = np.arange(6)
    adjacency[:, diagonal, diagonal] = False
    degrees = np.sum(adjacency, axis=-1)
    isolated = degrees == 0

    reach = adjacency.copy()
    reach[:, diagonal, diagonal] = True
    for node in range(6):
        reach |= reach[:, :, node, None] & reach[:, None, node, :]
    connected = np.all(reach, axis=(1, 2))
    isolated_count = np.sum(isolated, axis=-1)

    return {
        "cutoff_nm": float(cutoff_nm),
        "periodic_minimum_image": box_nm is not None,
        "num_input_frames": int(finite_frames.shape[0]),
        "num_finite_frames": int(coordinates.shape[0]),
        "finite_frame_fraction": float(np.mean(finite_frames)),
        "connected_frame_fraction": float(np.mean(connected)),
        "frames_with_isolated_bead_fraction": float(np.mean(isolated_count > 0)),
        "isolated_bead_fraction": float(np.mean(isolated)),
        "mean_isolated_beads_per_frame": float(np.mean(isolated_count)),
        "isolated_fraction_by_bead": {
            str(index): float(np.mean(isolated[:, index])) for index in range(6)
        },
        "degree": _quantiles(degrees),
    }


def ala2_neighbor_graph_metrics(
    coordinates_nm: Any,
    *,
    mace_box_nm: Any | None,
) -> dict[str, Any]:
    """Return the PMF-MACE and controller-PaiNN graph diagnostics together."""

    return {
        "mace_0p5_nm": neighbor_graph_metrics(
            coordinates_nm,
            cutoff_nm=MACE_GRAPH_CUTOFF_NM,
            box_nm=mace_box_nm,
        ),
        "painn_0p8_nm": neighbor_graph_metrics(
            coordinates_nm,
            cutoff_nm=PAINN_GRAPH_CUTOFF_NM,
            box_nm=None,
        ),
    }


def bond_score_direction_metrics(
    coordinates_nm: Any,
    target_score: Any,
    *,
    score_coordinate_scale_nm: float = 1.0,
    reference_bond_lengths_nm: Any | None = None,
    reference_tolerance_nm: float = 1.0e-4,
    support_lower_nm: Any | None = None,
    support_upper_nm: Any | None = None,
) -> dict[str, Any]:
    r"""Measure ``dr_ij/dtau`` for all five bonds under a target score.

    ``target_score`` is interpreted in the coordinates used by the model.  If
    it is a score with respect to standardized coordinates ``x`` and physical
    coordinates are ``R = scale*x``, pass that scale as
    ``score_coordinate_scale_nm``.  The diagnostic direction is then
    ``dR/dtau = scale*score``.  Positive derivatives lengthen a bond and
    negative derivatives shorten it.
    """

    coordinates = _as_batch(coordinates_nm, label="coordinates_nm")
    score = _as_batch(target_score, label="target_score")
    if score.shape != coordinates.shape:
        raise ValueError("target_score and coordinates_nm must have identical shapes")
    if not np.isfinite(score_coordinate_scale_nm) or score_coordinate_scale_nm <= 0.0:
        raise ValueError("score_coordinate_scale_nm must be positive and finite")
    if not np.isfinite(reference_tolerance_nm) or reference_tolerance_nm < 0.0:
        raise ValueError("reference_tolerance_nm must be finite and non-negative")

    finite = np.all(np.isfinite(coordinates), axis=(1, 2)) & np.all(
        np.isfinite(score), axis=(1, 2)
    )
    coordinates = coordinates[finite]
    velocity_nm = score[finite] * float(score_coordinate_scale_nm)
    if coordinates.shape[0] == 0:
        raise ValueError("coordinates/score contain no jointly finite frame")

    left = CORE_BETA_BONDS[:, 0]
    right = CORE_BETA_BONDS[:, 1]
    displacement = coordinates[:, left, :] - coordinates[:, right, :]
    distance = np.linalg.norm(displacement, axis=-1)
    safe_distance = np.maximum(distance, np.finfo(np.float64).tiny)
    relative_velocity = velocity_nm[:, left, :] - velocity_nm[:, right, :]
    derivative = np.sum(displacement * relative_velocity, axis=-1) / safe_distance

    reference = None
    if reference_bond_lengths_nm is not None:
        reference = np.asarray(reference_bond_lengths_nm, dtype=np.float64).reshape(-1)
        if reference.shape != (5,) or np.any(~np.isfinite(reference)) or np.any(reference <= 0.0):
            raise ValueError("reference_bond_lengths_nm must contain five positive finite values")
    support_bounds = None
    if support_lower_nm is not None or support_upper_nm is not None:
        if support_lower_nm is None or support_upper_nm is None:
            raise ValueError("support lower and upper bounds must be supplied together")
        lower = np.asarray(support_lower_nm, dtype=np.float64).reshape(-1)
        upper = np.asarray(support_upper_nm, dtype=np.float64).reshape(-1)
        if (
            lower.shape != (5,)
            or upper.shape != (5,)
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(lower <= 0.0)
            or np.any(lower >= upper)
        ):
            raise ValueError("support bounds must contain five finite 0 < lower < upper pairs")
        support_bounds = (lower, upper)

    per_bond: dict[str, Any] = {}
    for index, name in enumerate(CORE_BETA_BOND_NAMES):
        entry: dict[str, Any] = {
            "distance_nm": _quantiles(distance[:, index]),
            "dr_dtau_nm": _quantiles(derivative[:, index]),
            "shortening_fraction": float(np.mean(derivative[:, index] < 0.0)),
            "lengthening_fraction": float(np.mean(derivative[:, index] > 0.0)),
        }
        if reference is not None:
            error = distance[:, index] - reference[index]
            active = np.abs(error) > reference_tolerance_nm
            long = error > reference_tolerance_nm
            short = error < -reference_tolerance_nm
            restorative = error * derivative[:, index] < 0.0
            entry.update(
                {
                    "reference_nm": float(reference[index]),
                    "active_fraction": float(np.mean(active)),
                    "restorative_fraction_active": (
                        float(np.mean(restorative[active])) if np.any(active) else None
                    ),
                    "long_bond_fraction": float(np.mean(long)),
                    "long_bond_shortening_fraction": (
                        float(np.mean(derivative[long, index] < 0.0)) if np.any(long) else None
                    ),
                    "short_bond_fraction": float(np.mean(short)),
                    "short_bond_lengthening_fraction": (
                        float(np.mean(derivative[short, index] > 0.0)) if np.any(short) else None
                    ),
                }
            )
        if support_bounds is not None:
            lower, upper = support_bounds
            below = distance[:, index] < lower[index]
            above = distance[:, index] > upper[index]
            flat = ~(below | above)
            entry.update(
                {
                    "support_lower_nm": float(lower[index]),
                    "support_upper_nm": float(upper[index]),
                    "below_support_fraction": float(np.mean(below)),
                    "below_support_lengthening_fraction": (
                        float(np.mean(derivative[below, index] > 0.0))
                        if np.any(below)
                        else None
                    ),
                    "above_support_fraction": float(np.mean(above)),
                    "above_support_shortening_fraction": (
                        float(np.mean(derivative[above, index] < 0.0))
                        if np.any(above)
                        else None
                    ),
                    "flat_support_fraction": float(np.mean(flat)),
                }
            )
        per_bond[name] = entry

    return {
        "num_input_frames": int(finite.shape[0]),
        "num_jointly_finite_frames": int(coordinates.shape[0]),
        "jointly_finite_fraction": float(np.mean(finite)),
        "score_coordinate_scale_nm": float(score_coordinate_scale_nm),
        "sign_convention": "negative dr_dtau shortens; positive dr_dtau lengthens",
        "bonds": per_bond,
    }


def reference_bond_medians(reference_nm: Any) -> np.ndarray:
    """Return five topology-ordered reference bond medians in nanometres."""

    reference = prepare_core_beta_coordinates(reference_nm, variant="cg", units="nm")
    displacement = (
        reference[:, CORE_BETA_BONDS[:, 0], :]
        - reference[:, CORE_BETA_BONDS[:, 1], :]
    )
    return np.median(np.linalg.norm(displacement, axis=-1), axis=0)


def radial_scale_coordinates(reference_nm: Any, scale: float) -> np.ndarray:
    """Scale molecular shape about each frame's geometric centre."""

    reference = prepare_core_beta_coordinates(reference_nm, variant="cg", units="nm")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    centre = np.mean(reference, axis=1, keepdims=True)
    return centre + float(scale) * (reference - centre)


def radial_score_direction_metrics(
    reference_nm: Any,
    scale: float,
    target_score: Any,
    *,
    score_coordinate_scale_nm: float = 1.0,
) -> dict[str, Any]:
    r"""Project the score onto the one-dimensional reference scaling path.

    For ``R(alpha)=COM+alpha*(R_ref-COM)``, the reported value is the
    least-squares instantaneous ``dalpha/dtau`` induced by the score.  A value
    with sign opposite to ``alpha-1`` points back toward the reference scale.
    """

    reference = prepare_core_beta_coordinates(reference_nm, variant="cg", units="nm")
    scaled = radial_scale_coordinates(reference, scale)
    score = _as_batch(target_score, label="target_score")
    if score.shape != scaled.shape:
        raise ValueError("target_score shape must match the scaled reference batch")
    finite = np.all(np.isfinite(score), axis=(1, 2))
    if not np.any(finite):
        raise ValueError("target_score contains no finite frame")
    direction = reference - np.mean(reference, axis=1, keepdims=True)
    velocity = score * float(score_coordinate_scale_nm)
    velocity -= np.mean(velocity, axis=1, keepdims=True)
    denominator = np.sum(direction * direction, axis=(1, 2))
    dalpha = np.sum(direction * velocity, axis=(1, 2)) / denominator
    dalpha = dalpha[finite]
    return {
        "scale": float(scale),
        "dalpha_dtau": _quantiles(dalpha),
        "toward_reference_fraction": (
            None if scale == 1.0 else float(np.mean((float(scale) - 1.0) * dalpha < 0.0))
        ),
    }


def recoverability_report(
    coordinates_nm: Any,
    target_score: Any,
    *,
    score_coordinate_scale_nm: float,
    mace_box_nm: Any | None,
    reference_bond_lengths_nm: Any | None = None,
    support_lower_nm: Any | None = None,
    support_upper_nm: Any | None = None,
) -> dict[str, Any]:
    """Compose graph and five-bond score-direction diagnostics."""

    score = _as_batch(target_score, label="target_score")
    score_norm = np.linalg.norm(score, axis=-1)
    return {
        "neighbor_graphs": ala2_neighbor_graph_metrics(
            coordinates_nm,
            mace_box_nm=mace_box_nm,
        ),
        "target_score_norm_per_bead": _quantiles(score_norm),
        "bond_score_directions": bond_score_direction_metrics(
            coordinates_nm,
            score,
            score_coordinate_scale_nm=score_coordinate_scale_nm,
            reference_bond_lengths_nm=reference_bond_lengths_nm,
            support_lower_nm=support_lower_nm,
            support_upper_nm=support_upper_nm,
        ),
    }


__all__ = [
    "ala2_neighbor_graph_metrics",
    "bond_score_direction_metrics",
    "neighbor_graph_metrics",
    "radial_scale_coordinates",
    "radial_score_direction_metrics",
    "recoverability_report",
    "reference_bond_medians",
]
