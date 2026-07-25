"""Global, fixed-topology support extension for the six-bead Ala2 PMF.

The published CG-BG MACE checkpoint is a local PMF fitted on mapped molecular
configurations.  It has no fixed bond graph and it is invariant to permutations
of beads with the same species.  Those properties are useful inside the data
manifold but make the bare checkpoint an unsafe global oracle for a sampler
started from independent Gaussian coordinates.

This module supplies an explicit *physical target extension*.  Its broad
flat-bottom terms are zero on the molecular support and act only when a
configuration has lost the canonical core-beta topology.  Unlike the separate
training wall, these terms are part of the formal target and therefore must
also be included in likelihood reweighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array


def _validate_coordinates(coordinates: Array) -> Array:
    value = jnp.asarray(coordinates)
    if value.ndim < 2 or value.shape[-2:] != (6, 3):
        raise ValueError(f"Expected (...,6,3) core-beta coordinates, got {value.shape}")
    return value


def _safe_norm(vector: Array, *, epsilon: float = 1.0e-12) -> Array:
    """Differentiable norm that stays finite at coincident/collinear points."""

    value = jnp.asarray(vector)
    eps = jnp.asarray(epsilon, dtype=value.dtype)
    return jnp.sqrt(jnp.sum(jnp.square(value), axis=-1) + eps * eps)


def _minimum_image(displacement: Array, box_lengths_nm: Array) -> Array:
    lengths = jnp.asarray(box_lengths_nm, dtype=jnp.asarray(displacement).dtype)
    return displacement - lengths * jnp.round(displacement / lengths)


def _pair_distances(
    coordinates: Array,
    indices: tuple[tuple[int, int], ...],
    box_lengths_nm: Array,
) -> Array:
    value = _validate_coordinates(coordinates)
    pairs = jnp.asarray(indices, dtype=jnp.int32)
    displacement = value[..., pairs[:, 0], :] - value[..., pairs[:, 1], :]
    displacement = _minimum_image(displacement, box_lengths_nm)
    return _safe_norm(displacement)


def _bond_angles(
    coordinates: Array,
    indices: tuple[tuple[int, int, int], ...],
    box_lengths_nm: Array,
) -> Array:
    """Return robust angles in radians for arbitrary leading batch axes."""

    value = _validate_coordinates(coordinates)
    triples = jnp.asarray(indices, dtype=jnp.int32)
    left = value[..., triples[:, 0], :] - value[..., triples[:, 1], :]
    right = value[..., triples[:, 2], :] - value[..., triples[:, 1], :]
    left = _minimum_image(left, box_lengths_nm)
    right = _minimum_image(right, box_lengths_nm)
    # atan2(||u x v||, u.v) is more stable than acos near 0 and pi.  The
    # regularised cross norm also prevents a NaN gradient at exactly collinear
    # Gaussian/source configurations.
    sine = _safe_norm(jnp.cross(left, right))
    cosine = jnp.sum(left * right, axis=-1)
    return jnp.arctan2(sine, cosine)


@dataclass(frozen=True)
class SmoothLowerEnergyBound:
    r"""Smoothly remove spurious PMF wells below a trusted energy floor.

    For raw energy :math:`u`, the effective energy is

    .. math::

       u_{\mathrm{eff}} = u_{\min} + \tau\,\mathrm{softplus}
       ((u-u_{\min})/\tau).

    It approaches the identity above the floor, is bounded below by
    ``minimum_kj_mol``, and scales the raw gradient by a sigmoid.  Both the
    transformed energy and transformed gradient are used by the formal target.
    """

    minimum_kj_mol: float
    softness_kj_mol: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.minimum_kj_mol)):
            raise ValueError("minimum_kj_mol must be finite")
        if not math.isfinite(float(self.softness_kj_mol)) or self.softness_kj_mol <= 0.0:
            raise ValueError("softness_kj_mol must be positive and finite")

    def energy(self, raw_energy_kj_mol: Array) -> Array:
        raw = jnp.asarray(raw_energy_kj_mol)
        minimum = jnp.asarray(self.minimum_kj_mol, dtype=raw.dtype)
        softness = jnp.asarray(self.softness_kj_mol, dtype=raw.dtype)
        return minimum + softness * jax.nn.softplus((raw - minimum) / softness)

    def gradient_scale(self, raw_energy_kj_mol: Array) -> Array:
        raw = jnp.asarray(raw_energy_kj_mol)
        minimum = jnp.asarray(self.minimum_kj_mol, dtype=raw.dtype)
        softness = jnp.asarray(self.softness_kj_mol, dtype=raw.dtype)
        return jax.nn.sigmoid((raw - minimum) / softness)

    def metadata(self) -> dict[str, float | str]:
        return {
            "kind": "smooth_lower_energy_bound",
            "minimum_kj_mol": float(self.minimum_kj_mol),
            "softness_kj_mol": float(self.softness_kj_mol),
        }


@dataclass(frozen=True)
class SmoothEnergyWindow:
    """Two-sided trusted PMF energy window with a matched chain-rule scale.

    The lower transform removes spurious very-low OOD wells and the upper
    transform saturates arbitrarily high OOD energies/forces.  On the pinned
    PMF training support both transforms are exponentially close to identity.
    """

    minimum_kj_mol: float
    maximum_kj_mol: float
    lower_softness_kj_mol: float
    upper_softness_kj_mol: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_kj_mol,
            self.maximum_kj_mol,
            self.lower_softness_kj_mol,
            self.upper_softness_kj_mol,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Energy-window parameters must be finite")
        if self.minimum_kj_mol >= self.maximum_kj_mol:
            raise ValueError("minimum_kj_mol must be smaller than maximum_kj_mol")
        if self.lower_softness_kj_mol <= 0.0 or self.upper_softness_kj_mol <= 0.0:
            raise ValueError("Energy-window softness values must be positive")

    def _lowered(self, raw_energy_kj_mol: Array) -> Array:
        raw = jnp.asarray(raw_energy_kj_mol)
        minimum = jnp.asarray(self.minimum_kj_mol, dtype=raw.dtype)
        softness = jnp.asarray(self.lower_softness_kj_mol, dtype=raw.dtype)
        return minimum + softness * jax.nn.softplus((raw - minimum) / softness)

    def energy(self, raw_energy_kj_mol: Array) -> Array:
        lowered = self._lowered(raw_energy_kj_mol)
        maximum = jnp.asarray(self.maximum_kj_mol, dtype=lowered.dtype)
        softness = jnp.asarray(self.upper_softness_kj_mol, dtype=lowered.dtype)
        return maximum - softness * jax.nn.softplus((maximum - lowered) / softness)

    def gradient_scale(self, raw_energy_kj_mol: Array) -> Array:
        raw = jnp.asarray(raw_energy_kj_mol)
        minimum = jnp.asarray(self.minimum_kj_mol, dtype=raw.dtype)
        lower_softness = jnp.asarray(self.lower_softness_kj_mol, dtype=raw.dtype)
        lowered = self._lowered(raw)
        maximum = jnp.asarray(self.maximum_kj_mol, dtype=raw.dtype)
        upper_softness = jnp.asarray(self.upper_softness_kj_mol, dtype=raw.dtype)
        lower_scale = jax.nn.sigmoid((raw - minimum) / lower_softness)
        upper_scale = jax.nn.sigmoid((maximum - lowered) / upper_softness)
        return lower_scale * upper_scale

    def metadata(self) -> dict[str, float | str]:
        return {
            "kind": "smooth_energy_window",
            "minimum_kj_mol": float(self.minimum_kj_mol),
            "maximum_kj_mol": float(self.maximum_kj_mol),
            "lower_softness_kj_mol": float(self.lower_softness_kj_mol),
            "upper_softness_kj_mol": float(self.upper_softness_kj_mol),
        }


@dataclass(frozen=True)
class Ala2CanonicalSupport:
    """Broad canonical topology guards for core-beta Ala2 coordinates.

    All distances are in nm, all angles are in radians, and returned energies
    are in kJ/mol.  The default/production parameters live in Hydra rather
    than in this class so every checkpoint records the complete target.
    """

    box_lengths_nm: tuple[float, float, float]
    bond_indices: tuple[tuple[int, int], ...]
    bond_lower_nm: tuple[float, ...]
    bond_upper_nm: tuple[float, ...]
    bond_force_constant_kj_mol_nm2: float
    angle_indices: tuple[tuple[int, int, int], ...]
    angle_lower_rad: tuple[float, ...]
    angle_upper_rad: tuple[float, ...]
    angle_force_constant_kj_mol_rad2: float
    repulsion_indices: tuple[tuple[int, int], ...]
    repulsion_min_nm: float
    repulsion_force_constant_kj_mol_nm2: float

    def __post_init__(self) -> None:
        box = tuple(float(value) for value in self.box_lengths_nm)
        bonds = tuple(tuple(int(index) for index in pair) for pair in self.bond_indices)
        angles = tuple(tuple(int(index) for index in triple) for triple in self.angle_indices)
        repulsion = tuple(
            tuple(int(index) for index in pair) for pair in self.repulsion_indices
        )
        lower_bond = tuple(float(value) for value in self.bond_lower_nm)
        upper_bond = tuple(float(value) for value in self.bond_upper_nm)
        lower_angle = tuple(float(value) for value in self.angle_lower_rad)
        upper_angle = tuple(float(value) for value in self.angle_upper_rad)

        if len(box) != 3 or not all(
            math.isfinite(value) and value > 0.0 for value in box
        ):
            raise ValueError("box_lengths_nm must contain three positive finite values")
        if not bonds or len(bonds) != len(lower_bond) or len(bonds) != len(upper_bond):
            raise ValueError("Every canonical bond needs one lower and one upper bound")
        if not angles or len(angles) != len(lower_angle) or len(angles) != len(upper_angle):
            raise ValueError("Every canonical angle needs one lower and one upper bound")
        if not repulsion:
            raise ValueError("At least one nonbonded repulsion pair is required")
        for pair in bonds + repulsion:
            if len(pair) != 2 or min(pair) < 0 or max(pair) >= 6 or pair[0] == pair[1]:
                raise ValueError(f"Invalid core-beta pair {pair!r}")
        for triple in angles:
            if (
                len(triple) != 3
                or min(triple) < 0
                or max(triple) >= 6
                or len(set(triple)) != 3
            ):
                raise ValueError(f"Invalid core-beta angle {triple!r}")
        if len({tuple(sorted(pair)) for pair in bonds}) != len(bonds):
            raise ValueError("Canonical bond pairs must be unique")
        if len({tuple(sorted(pair)) for pair in repulsion}) != len(repulsion):
            raise ValueError("Repulsion pairs must be unique")
        if {tuple(sorted(pair)) for pair in bonds}.intersection(
            tuple(sorted(pair)) for pair in repulsion
        ):
            raise ValueError("Bonded pairs cannot also be repulsion pairs")
        if not all(
            math.isfinite(low) and math.isfinite(high) and 0.0 < low < high
            for low, high in zip(lower_bond, upper_bond, strict=True)
        ):
            raise ValueError("Bond bounds must be finite and satisfy 0 < lower < upper")
        if not all(
            math.isfinite(low)
            and math.isfinite(high)
            and 0.0 < low < high < math.pi
            for low, high in zip(lower_angle, upper_angle, strict=True)
        ):
            raise ValueError("Angle bounds must lie strictly inside (0, pi)")
        constants = (
            self.bond_force_constant_kj_mol_nm2,
            self.angle_force_constant_kj_mol_rad2,
            self.repulsion_min_nm,
            self.repulsion_force_constant_kj_mol_nm2,
        )
        if not all(math.isfinite(float(value)) and value > 0.0 for value in constants):
            raise ValueError("Support force constants and repulsion_min_nm must be positive")

        object.__setattr__(self, "box_lengths_nm", box)
        object.__setattr__(self, "bond_indices", bonds)
        object.__setattr__(self, "bond_lower_nm", lower_bond)
        object.__setattr__(self, "bond_upper_nm", upper_bond)
        object.__setattr__(self, "angle_indices", angles)
        object.__setattr__(self, "angle_lower_rad", lower_angle)
        object.__setattr__(self, "angle_upper_rad", upper_angle)
        object.__setattr__(self, "repulsion_indices", repulsion)

    def energy_components(self, coordinates_nm: Array) -> dict[str, Array]:
        coordinates = _validate_coordinates(coordinates_nm)
        box = jnp.asarray(self.box_lengths_nm, dtype=coordinates.dtype)
        bond_distance = _pair_distances(coordinates, self.bond_indices, box)
        bond_lower = jnp.asarray(self.bond_lower_nm, dtype=coordinates.dtype)
        bond_upper = jnp.asarray(self.bond_upper_nm, dtype=coordinates.dtype)
        bond_excess_low = jax.nn.relu(bond_lower - bond_distance)
        bond_excess_high = jax.nn.relu(bond_distance - bond_upper)
        bond = 0.5 * jnp.asarray(
            self.bond_force_constant_kj_mol_nm2, dtype=coordinates.dtype
        ) * jnp.sum(bond_excess_low**2 + bond_excess_high**2, axis=-1)

        angle = _bond_angles(coordinates, self.angle_indices, box)
        angle_lower = jnp.asarray(self.angle_lower_rad, dtype=coordinates.dtype)
        angle_upper = jnp.asarray(self.angle_upper_rad, dtype=coordinates.dtype)
        angle_excess_low = jax.nn.relu(angle_lower - angle)
        angle_excess_high = jax.nn.relu(angle - angle_upper)
        angle_energy = 0.5 * jnp.asarray(
            self.angle_force_constant_kj_mol_rad2, dtype=coordinates.dtype
        ) * jnp.sum(angle_excess_low**2 + angle_excess_high**2, axis=-1)

        nonbonded_distance = _pair_distances(coordinates, self.repulsion_indices, box)
        repulsion_excess = jax.nn.relu(
            jnp.asarray(self.repulsion_min_nm, dtype=coordinates.dtype)
            - nonbonded_distance
        )
        repulsion = 0.5 * jnp.asarray(
            self.repulsion_force_constant_kj_mol_nm2, dtype=coordinates.dtype
        ) * jnp.sum(repulsion_excess**2, axis=-1)
        return {
            "U_bond": bond,
            "U_angle": angle_energy,
            "U_repulsion": repulsion,
        }

    def energy(self, coordinates_nm: Array) -> Array:
        components = self.energy_components(coordinates_nm)
        return components["U_bond"] + components["U_angle"] + components["U_repulsion"]

    def energy_and_grad(self, coordinates_nm: Array) -> tuple[Array, Array]:
        energy, gradient, _components = self.energy_and_grad_components(coordinates_nm)
        return energy, gradient

    def energy_and_grad_components(
        self,
        coordinates_nm: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        coordinates = _validate_coordinates(coordinates_nm)

        def summed_energy(value: Array) -> tuple[Array, dict[str, Array]]:
            components = self.energy_components(value)
            total = (
                components["U_bond"]
                + components["U_angle"]
                + components["U_repulsion"]
            )
            return jnp.sum(total), components

        (_summed, components), gradient = jax.value_and_grad(
            summed_energy,
            has_aux=True,
        )(coordinates)
        energy = (
            components["U_bond"]
            + components["U_angle"]
            + components["U_repulsion"]
        )
        return energy, gradient, components

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "ala2_core_beta_canonical_support_v1",
            "box_lengths_nm": list(self.box_lengths_nm),
            "bond_indices": [list(pair) for pair in self.bond_indices],
            "bond_lower_nm": list(self.bond_lower_nm),
            "bond_upper_nm": list(self.bond_upper_nm),
            "bond_force_constant_kj_mol_nm2": float(
                self.bond_force_constant_kj_mol_nm2
            ),
            "angle_indices": [list(triple) for triple in self.angle_indices],
            "angle_lower_rad": list(self.angle_lower_rad),
            "angle_upper_rad": list(self.angle_upper_rad),
            "angle_force_constant_kj_mol_rad2": float(
                self.angle_force_constant_kj_mol_rad2
            ),
            "repulsion_indices": [list(pair) for pair in self.repulsion_indices],
            "repulsion_min_nm": float(self.repulsion_min_nm),
            "repulsion_force_constant_kj_mol_nm2": float(
                self.repulsion_force_constant_kj_mol_nm2
            ),
        }


__all__ = [
    "Ala2CanonicalSupport",
    "SmoothEnergyWindow",
    "SmoothLowerEnergyBound",
]
