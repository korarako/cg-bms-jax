"""All-atom Ala2 OpenMM target on a full-rank 66D ambient space.

The molecular potential is evaluated on the centred 63D shape.  The missing
three translational degrees of freedom are supplied by an analytic Gaussian
auxiliary density.  With ``com_std=1/sqrt(N)`` the normalized centre
``a=sqrt(N)*mean(X)`` is standard normal when ``X ~ Normal(0, I_66)``.

Coordinates accepted by :class:`AmbientAllAtomAla2Potential` are standardized
Angstrom coordinates in official BridgeMatchingSampler atom order.  OpenMM
uses nanometres internally; all chain-rule factors are applied explicitly.
"""

from __future__ import annotations

import hashlib
import math
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from cg_bms_jax.coordinates import (
    com_aux_reduced_energy,
    com_aux_reduced_gradient,
    split_shape_com,
)
from cg_bms_jax.potential.base import PotentialResult, ScoreDecomposition
from cg_bms_jax.potential.restraint import BMSImproperRestraint

Array = jax.Array


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a target-defining asset."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_openmm_xtc_compat() -> None:
    """Avoid importing an optional XTC extension with an incompatible NumPy ABI."""

    if "openmm.app.xtcfile" not in sys.modules:
        module = types.ModuleType("openmm.app.xtcfile")

        class XTCFile:  # pragma: no cover - this project never writes XTC.
            pass

        module.XTCFile = XTCFile
        sys.modules["openmm.app.xtcfile"] = module


class AllAtomEnergyBackend(Protocol):
    """Host backend returning energy and its gradient in Angstrom units."""

    num_particles: int
    num_constraints: int

    def energy_and_grad_angstrom(
        self, coordinates_angstrom: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def metadata(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class OpenMMAla2Spec:
    """Serializable construction contract for the all-atom target."""

    pdb_path: str
    device: str = "cpu"
    precision: str = "mixed"
    device_index: str = "0"
    forcefield: str = "amber99sbildn.xml"
    # Exact force-field list pinned by the public boltzkit Ala2 target.
    implicit_forcefield: str = "amber99_obc.xml"
    expected_particles: int = 22
    expected_constraints: int = 0

    def __post_init__(self) -> None:
        if self.expected_particles != 22:
            raise ValueError("The Ala2 all-atom target requires exactly 22 atoms")
        if self.expected_constraints < 0:
            raise ValueError("expected_constraints must be non-negative")
        if self.device.lower() not in {"cpu", "cuda", "reference"}:
            raise ValueError("OpenMM device must be cpu, cuda, or reference")


class OpenMMAla2Backend:
    """Reusable OpenMM context for official-order Ala2 energy and gradients."""

    def __init__(self, spec: OpenMMAla2Spec) -> None:
        try:
            _ensure_openmm_xtc_compat()
            import openmm
            import openmm.app
            import openmm.unit as unit
        except ImportError as error:  # pragma: no cover - exercised on GPU host.
            raise RuntimeError(
                "All-atom Ala2 requires OpenMM. Install cg-bms-jax[all_atom]."
            ) from error

        pdb_path = Path(spec.pdb_path).expanduser().resolve()
        if not pdb_path.is_file():
            raise FileNotFoundError(pdb_path)
        pdb = openmm.app.PDBFile(str(pdb_path))
        forcefield_names = [spec.forcefield]
        if spec.implicit_forcefield:
            forcefield_names.append(spec.implicit_forcefield)
        forcefield = openmm.app.ForceField(*forcefield_names)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=openmm.app.NoCutoff,
            constraints=None,
            removeCMMotion=False,
        )
        num_particles = int(system.getNumParticles())
        num_constraints = int(system.getNumConstraints())
        if num_particles != spec.expected_particles:
            raise ValueError(
                f"OpenMM particle count {num_particles} != {spec.expected_particles}"
            )
        if num_constraints != spec.expected_constraints:
            raise ValueError(
                "A formal ambient density cannot silently change dimension: "
                f"OpenMM constraints={num_constraints}, expected={spec.expected_constraints}"
            )

        device = spec.device.lower()
        if device == "cuda":
            platform = openmm.Platform.getPlatformByName("CUDA")
            properties = {
                "DeviceIndex": str(spec.device_index),
                "Precision": str(spec.precision),
                "UseCpuPme": "false",
            }
        elif device == "reference":
            platform = openmm.Platform.getPlatformByName("Reference")
            properties = {}
        else:
            platform = openmm.Platform.getPlatformByName("CPU")
            properties = {}
        integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
        simulation = openmm.app.Simulation(
            pdb.topology, system, integrator, platform, properties
        )

        self.spec = spec
        self.openmm = openmm
        self.unit = unit
        self.simulation = simulation
        self.num_particles = num_particles
        self.num_constraints = num_constraints
        self.pdb_sha256 = file_sha256(pdb_path)
        self.platform_name = str(platform.getName())

    def energy_and_grad_angstrom(
        self, coordinates_angstrom: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return kJ/mol and kJ/mol/Angstrom in official PDB atom order."""

        values = np.asarray(coordinates_angstrom, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3 or values.shape[1:] != (22, 3):
            raise ValueError(
                f"Expected all-atom coordinates (B,22,3), got {values.shape}"
            )
        energies = np.empty((values.shape[0],), dtype=np.float64)
        gradients = np.empty_like(values)
        for index, coordinates in enumerate(values):
            self.simulation.context.setPositions(coordinates * self.unit.angstrom)
            state = self.simulation.context.getState(
                getEnergy=True, getForces=True
            )
            energies[index] = state.getPotentialEnergy().value_in_unit(
                self.unit.kilojoule_per_mole
            )
            forces_nm = np.asarray(
                state.getForces(asNumpy=True).value_in_unit(
                    self.unit.kilojoule_per_mole / self.unit.nanometer
                ),
                dtype=np.float64,
            )
            # grad_A U = -force_nm * d(nm)/d(A) = -force_nm / 10.
            gradients[index] = -0.1 * forces_nm
        # Preserve host precision.  The adapter below casts to the state dtype,
        # so float32 training remains unchanged while float64 parity and
        # likelihood audits do not lose precision at this boundary.
        return energies, gradients

    def metadata(self) -> dict[str, Any]:
        return {
            "implementation_abi": "openmm_ala2_unconstrained_host_v1",
            "pdb_path": str(Path(self.spec.pdb_path).expanduser().resolve()),
            "pdb_sha256": self.pdb_sha256,
            "forcefield": self.spec.forcefield,
            "implicit_forcefield": self.spec.implicit_forcefield,
            "forcefield_profile": (
                f"{self.spec.forcefield}+"
                f"{self.spec.implicit_forcefield or 'vacuum'}"
            ),
            "platform": self.platform_name,
            "precision": self.spec.precision,
            "num_particles": self.num_particles,
            "num_constraints": self.num_constraints,
            "openmm_version": getattr(self.openmm, "__version__", "unknown"),
        }


class AmbientAllAtomAla2Potential:
    """OpenMM + optional BMS impropers + exact Gaussian COM augmentation."""

    def __init__(
        self,
        backend: AllAtomEnergyBackend,
        *,
        temperature_kelvin: float = 300.0,
        physical_std_angstrom: float = 1.0,
        com_std: float | None = None,
        restraints: Sequence[BMSImproperRestraint] = (),
    ) -> None:
        if int(backend.num_particles) != 22:
            raise ValueError("All-atom Ala2 backend must contain 22 particles")
        if int(backend.num_constraints) != 0:
            raise ValueError(
                "Full-rank 66D density requires an unconstrained OpenMM system"
            )
        if not math.isfinite(temperature_kelvin) or temperature_kelvin <= 0.0:
            raise ValueError("temperature_kelvin must be finite and positive")
        if (
            not math.isfinite(physical_std_angstrom)
            or physical_std_angstrom <= 0.0
        ):
            raise ValueError("physical_std_angstrom must be finite and positive")
        resolved_com_std = 22**-0.5 if com_std is None else float(com_std)
        if not math.isfinite(resolved_com_std) or resolved_com_std <= 0.0:
            raise ValueError("com_std must be finite and positive")
        if not math.isclose(
            resolved_com_std,
            22**-0.5,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "The exact ambient-66 target requires com_std=1/sqrt(22), "
                "so z=sqrt(22)*mean(x) is standard normal"
            )

        self.backend = backend
        self.temperature_kelvin = float(temperature_kelvin)
        # OpenMM's molar gas constant in kJ mol^-1 K^-1.
        self.kT = 0.00831446261815324 * self.temperature_kelvin
        self.physical_std_angstrom = float(physical_std_angstrom)
        self.com_std = resolved_com_std
        self.restraints = tuple(restraints)

    @staticmethod
    def _batched(x: Array) -> tuple[Array, bool]:
        value = jnp.asarray(x)
        if value.ndim == 2:
            value = value[None, ...]
            squeezed = True
        elif value.ndim == 3:
            squeezed = False
        else:
            raise ValueError(f"Expected (22,3) or (B,22,3), got {value.shape}")
        if value.shape[1:] != (22, 3):
            raise ValueError(f"Expected (B,22,3), got {value.shape}")
        return value, squeezed

    def _physical_components(
        self, standardized_ambient: Array
    ) -> tuple[Array, Array, dict[str, Array]]:
        x, squeezed = self._batched(standardized_ambient)
        physical = x * jnp.asarray(self.physical_std_angstrom, dtype=x.dtype)
        shape, _center = split_shape_com(physical)
        energy_openmm_np, gradient_openmm_np = self.backend.energy_and_grad_angstrom(
            np.asarray(jax.device_get(shape))
        )
        # Remove the translational force component while the OpenMM result is
        # still float64.  Casting an off-manifold collision force to float32
        # before this reduction can overflow or leave a large common residual.
        gradient_openmm_np = np.asarray(gradient_openmm_np, dtype=np.float64)
        gradient_openmm_np = gradient_openmm_np - np.mean(
            gradient_openmm_np, axis=-2, keepdims=True, dtype=np.float64
        )
        energy_openmm = jnp.asarray(energy_openmm_np, dtype=x.dtype)
        gradient_physical = jnp.asarray(gradient_openmm_np, dtype=x.dtype)
        # Translation invariance is enforced even if backend rounding leaves a
        # tiny residual total force.
        gradient_physical = gradient_physical - jnp.mean(
            gradient_physical, axis=-2, keepdims=True
        )

        energy_restraints = jnp.zeros_like(energy_openmm)
        for restraint in self.restraints:
            restraint_energy, restraint_gradient = restraint.energy_and_grad(shape)
            energy_restraints = energy_restraints + restraint_energy
            gradient_physical = gradient_physical + restraint_gradient
        gradient_physical = gradient_physical - jnp.mean(
            gradient_physical, axis=-2, keepdims=True
        )
        total = energy_openmm + energy_restraints
        components = {
            "U_openmm": energy_openmm,
            "U_improper": energy_restraints,
            "U_target": total,
        }
        if squeezed:
            return (
                total[0],
                gradient_physical[0] * self.physical_std_angstrom,
                {name: value[0] for name, value in components.items()},
            )
        return (
            total,
            gradient_physical * self.physical_std_angstrom,
            components,
        )

    def energy(self, standardized_ambient: Array) -> Array:
        energy, _gradient, _components = self._physical_components(
            standardized_ambient
        )
        return energy

    def energy_and_grad(
        self, standardized_ambient: Array
    ) -> tuple[Array, Array]:
        energy, gradient, _components = self._physical_components(
            standardized_ambient
        )
        return energy, gradient

    def evaluate(
        self,
        standardized_ambient: Array,
        *,
        include_training_wall: bool = True,
    ) -> PotentialResult:
        # ``include_training_wall`` is accepted for RuntimeSystem symmetry.
        # This target deliberately has no hidden training-only energy term.
        del include_training_wall
        x = jnp.asarray(standardized_ambient)
        energy, gradient_x, components = self._physical_components(x)
        reduced_shape_energy = energy / self.kT
        reduced_shape_gradient = gradient_x / self.kT
        reduced_shape_gradient = reduced_shape_gradient - jnp.mean(
            reduced_shape_gradient, axis=-2, keepdims=True
        )
        reduced_energy = reduced_shape_energy + com_aux_reduced_energy(
            x, com_std=self.com_std
        )
        reduced_com_gradient = com_aux_reduced_gradient(
            x, com_std=self.com_std
        )
        reduced_gradient = reduced_shape_gradient + reduced_com_gradient
        shape_score = -reduced_shape_gradient
        com_score = -reduced_com_gradient
        valid = jnp.isfinite(reduced_energy) & jnp.all(
            jnp.isfinite(reduced_gradient), axis=(-2, -1)
        )
        components = {
            **components,
            "U_com_reduced": com_aux_reduced_energy(x, com_std=self.com_std),
        }
        return PotentialResult(
            energy=energy,
            gradient=reduced_gradient * self.kT,
            reduced_energy=reduced_energy,
            reduced_gradient=reduced_gradient,
            score=-reduced_gradient,
            valid_mask=valid,
            components=components,
            score_decomposition=ScoreDecomposition(
                clippable=shape_score,
                preserved=com_score,
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **dict(self.backend.metadata()),
            "temperature_kelvin": self.temperature_kelvin,
            "kT_kj_mol": self.kT,
            "physical_std_angstrom": self.physical_std_angstrom,
            "com_std": self.com_std,
            "density_mode": "ambient_66d_aux_com_exact",
            "restraints": [
                {
                    "indices": list(restraint.indices),
                    "location_rad": restraint.location,
                    "tolerance_rad": restraint.tolerance,
                    "force_constant_kj_mol_rad2": (
                        restraint.force_constant_kj_mol
                    ),
                }
                for restraint in self.restraints
            ],
        }


__all__ = [
    "AllAtomEnergyBackend",
    "AmbientAllAtomAla2Potential",
    "OpenMMAla2Backend",
    "OpenMMAla2Spec",
    "file_sha256",
]
