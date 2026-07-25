"""BMS-compatible chirality restraints for coarse-grained and all-atom Ala2.

The implementation intentionally follows the upstream BMS ``TorsionCV`` and
``ChiralityRestraint`` formulas term for term.  Energies exposed here are in
kJ/mol so they can be added directly to the CG-BG PMF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array

# CODATA-compatible conversion used by common molecular simulation packages.
EV_TO_KJ_MOL = 96.48533212331002
BMS_CB_FORCE_CONSTANT_KJ_MOL = 25.0 * EV_TO_KJ_MOL
BMS_CB_LOCATION_RAD = 0.6154797086703873
BMS_HA_LOCATION_RAD = -0.6154797086703873
BMS_IMPROPER_TOLERANCE_RAD = 0.4363323129985824
BMS_CB_INDICES = (2, 1, 4, 3)  # CA, N, C, CB in core-beta bead order.
BMS_AA_CB_INDICES = (8, 6, 14, 10)  # CA, N, C, CB in official BMS order.
BMS_AA_HA_INDICES = (8, 6, 14, 9)  # CA, N, C, HA in official BMS order.


def torsion_angle(coordinates: Array, indices: tuple[int, int, int, int]) -> Array:
    """Return the upstream-BMS improper torsion for arbitrary leading batches.

    This is a direct JAX translation of ``bms.potential.collective_variable``:
    ``b0=p0-p1``, ``b1=p2-p1``, and ``b2=p3-p2``, followed by projection onto
    the plane normal to the normalized ``b1`` axis and an ``atan2``.
    """

    x = jnp.asarray(coordinates)
    if x.ndim < 2 or x.shape[-1] != 3:
        raise ValueError(f"Expected (...,N,3) coordinates, got {x.shape}")
    if len(indices) != 4 or min(indices) < 0 or max(indices) >= x.shape[-2]:
        raise ValueError(f"Invalid torsion indices {indices!r} for {x.shape[-2]} particles")
    selected = x[..., jnp.asarray(indices), :]
    p0, p1, p2, p3 = (selected[..., index, :] for index in range(4))
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    # Keep the exact upstream regularizer and operation ordering.
    b1 = b1 / jnp.sqrt(jnp.sum(jnp.square(b1), axis=-1, keepdims=True) + 1.0e-10)
    v = b0 - b1 * jnp.sum(b0 * b1, axis=-1, keepdims=True)
    w = b2 - b1 * jnp.sum(b2 * b1, axis=-1, keepdims=True)
    return jnp.arctan2(
        jnp.sum(jnp.cross(b1, v) * w, axis=-1),
        jnp.sum(v * w, axis=-1),
    )


def periodic_displacement(value: Array, location: float | Array) -> Array:
    """Return the signed shortest angular displacement in ``[-pi, pi]``."""

    delta = jnp.asarray(value) - jnp.asarray(location)
    return jnp.arctan2(jnp.sin(delta), jnp.cos(delta))


@dataclass(frozen=True)
class BMSImproperRestraint:
    """Flat-bottom improper used by BMS, in kJ/mol.

    The default is the six-bead CA--N--C--CB improper.  Passing the official
    all-atom indices also represents the CB and HA restraints from the
    released BridgeMatchingSampler Ala2 configuration.
    """

    indices: tuple[int, int, int, int] = BMS_CB_INDICES
    location: float = BMS_CB_LOCATION_RAD
    force_constant_kj_mol: float = BMS_CB_FORCE_CONSTANT_KJ_MOL
    tolerance: float = BMS_IMPROPER_TOLERANCE_RAD

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.indices)
        if len(indices) != 4 or min(indices) < 0:
            raise ValueError("indices must contain four non-negative particle indices")
        if len(set(indices)) != 4:
            raise ValueError("torsion indices must be distinct")
        finite = (self.location, self.force_constant_kj_mol, self.tolerance)
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("restraint parameters must be finite")
        if self.force_constant_kj_mol <= 0.0:
            raise ValueError("force_constant_kj_mol must be positive")
        if not 0.0 <= self.tolerance < math.pi:
            raise ValueError("tolerance must lie in [0, pi)")
        object.__setattr__(self, "indices", indices)

    def torsion(self, coordinates: Array) -> Array:
        return torsion_angle(coordinates, self.indices)

    def energy(self, coordinates: Array) -> Array:
        """Return the dimensional flat-bottom restraint energy in kJ/mol."""

        displacement = periodic_displacement(self.torsion(coordinates), self.location)
        excess = jnp.maximum(jnp.abs(displacement) - self.tolerance, 0.0)
        return 0.5 * self.force_constant_kj_mol * jnp.square(excess)

    def energy_and_grad(self, coordinates: Array) -> tuple[Array, Array]:
        """Return energy and its Cartesian gradient in the input coordinates."""

        x = jnp.asarray(coordinates)
        energy = self.energy(x)
        gradient = jax.grad(lambda value: jnp.sum(self.energy(value)))(x)
        return energy, gradient


# Backwards-compatible public name retained for the existing six-bead configs.
BMSCBImproperRestraint = BMSImproperRestraint


__all__ = [
    "BMS_CB_FORCE_CONSTANT_KJ_MOL",
    "BMS_CB_INDICES",
    "BMS_CB_LOCATION_RAD",
    "BMS_HA_LOCATION_RAD",
    "BMS_AA_CB_INDICES",
    "BMS_AA_HA_INDICES",
    "BMS_IMPROPER_TOLERANCE_RAD",
    "BMSImproperRestraint",
    "BMSCBImproperRestraint",
    "EV_TO_KJ_MOL",
    "periodic_displacement",
    "torsion_angle",
]
