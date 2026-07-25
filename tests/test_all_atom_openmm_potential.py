from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.data import (
    reorder_ala2_bms_to_tam,
    reorder_ala2_tam_to_bms,
)
from cg_bms_jax.potential import (
    AmbientAllAtomAla2Potential,
    OpenMMAla2Backend,
    OpenMMAla2Spec,
    file_sha256,
)
from cg_bms_jax.runtime import build_runtime_system


class _QuadraticShapeBackend:
    num_particles = 22
    num_constraints = 0

    def energy_and_grad_angstrom(
        self, coordinates_angstrom: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(coordinates_angstrom, dtype=np.float32)
        return 0.5 * np.sum(x * x, axis=(1, 2)), x

    def metadata(self) -> dict[str, object]:
        return {"implementation_abi": "quadratic_test"}


def test_all_atom_ambient_target_is_full_rank_and_translation_separable() -> None:
    potential = AmbientAllAtomAla2Potential(
        _QuadraticShapeBackend(),
        temperature_kelvin=300.0,
        physical_std_angstrom=1.0,
    )
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(size=(3, 22, 3)).astype(np.float32))
    result = potential.evaluate(x)

    centered = x - jnp.mean(x, axis=1, keepdims=True)
    center = jnp.mean(x, axis=1)
    expected_shape_energy = 0.5 * jnp.sum(centered**2, axis=(1, 2))
    expected_com_reduced = 0.5 * 22.0 * jnp.sum(center**2, axis=-1)
    np.testing.assert_allclose(result.energy, expected_shape_energy, rtol=2e-6)
    np.testing.assert_allclose(
        result.reduced_energy,
        expected_shape_energy / potential.kT + expected_com_reduced,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        jnp.mean(result.reduced_gradient, axis=1),
        center,
        rtol=2e-6,
        atol=2e-6,
    )
    assert result.score_decomposition is not None
    parts = result.score_decomposition
    np.testing.assert_allclose(
        parts.clippable + parts.preserved,
        result.score,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        jnp.mean(parts.clippable, axis=1),
        0.0,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        parts.preserved,
        jnp.broadcast_to(-center[:, None, :], x.shape),
        rtol=2e-6,
        atol=2e-6,
    )
    assert np.asarray(result.valid_mask).all()


def test_all_atom_shape_energy_is_translation_invariant_but_augmented_target_is_not() -> None:
    potential = AmbientAllAtomAla2Potential(_QuadraticShapeBackend())
    x = jnp.linspace(-1.0, 1.0, 66, dtype=jnp.float32).reshape(1, 22, 3)
    shift = jnp.asarray([[[0.4, -0.2, 0.1]]], dtype=x.dtype)
    original = potential.evaluate(x)
    translated = potential.evaluate(x + shift)

    np.testing.assert_allclose(original.energy, translated.energy, rtol=2e-6)
    expected_delta = 0.5 * 22.0 * (
        jnp.sum((jnp.mean(x, axis=1) + shift[:, 0]) ** 2, axis=-1)
        - jnp.sum(jnp.mean(x, axis=1) ** 2, axis=-1)
    )
    np.testing.assert_allclose(
        translated.reduced_energy - original.reduced_energy,
        expected_delta,
        rtol=2e-6,
        atol=2e-6,
    )


def test_all_atom_target_rejects_constrained_backend() -> None:
    backend = _QuadraticShapeBackend()
    backend.num_constraints = 1
    try:
        AmbientAllAtomAla2Potential(backend)
    except ValueError as error:
        assert "unconstrained" in str(error)
    else:  # pragma: no cover
        raise AssertionError("constrained backend should be rejected")


def test_default_com_std_matches_standard_normal_66d_source() -> None:
    potential = AmbientAllAtomAla2Potential(_QuadraticShapeBackend())
    assert math.isclose(potential.com_std, 1.0 / math.sqrt(22.0))


def test_nonstandard_com_is_rejected_by_exact_ambient_target() -> None:
    with pytest.raises(ValueError, match=r"com_std=1/sqrt\(22\)"):
        AmbientAllAtomAla2Potential(
            _QuadraticShapeBackend(),
            com_std=0.25,
        )


def test_physical_standardization_chain_rule_is_applied_once() -> None:
    physical_std = 2.5
    potential = AmbientAllAtomAla2Potential(
        _QuadraticShapeBackend(),
        physical_std_angstrom=physical_std,
    )
    x = jnp.linspace(-0.8, 1.1, 66, dtype=jnp.float32).reshape(1, 22, 3)
    centered = x - jnp.mean(x, axis=1, keepdims=True)
    center = jnp.mean(x, axis=1, keepdims=True)
    result = potential.evaluate(x)

    expected_energy = 0.5 * physical_std**2 * jnp.sum(
        centered**2, axis=(1, 2)
    )
    expected_reduced_gradient = (
        physical_std**2 * centered / potential.kT
        + jnp.broadcast_to(center, x.shape)
    )
    np.testing.assert_allclose(result.energy, expected_energy, rtol=2e-6)
    np.testing.assert_allclose(
        result.reduced_gradient,
        expected_reduced_gradient,
        rtol=2e-6,
        atol=2e-6,
    )


def _all_atom_runtime_config(pdb_path: Path) -> dict[str, Any]:
    return {
        "name": "ala2_aa_runtime_unit",
        "state_shape": [22, 3],
        "coordinate_mode": "ala2_aa_ambient66",
        "temperature_kelvin": 300.0,
        "kT": 2.494338785445972,
        "coordinates": {
            "physical_std_angstrom": 1.0,
            "com_sigma": 1.0 / math.sqrt(22.0),
        },
        "source": {"mean": 0.0, "sigma": 1.0},
        "sde": {"sigma_min": 0.001, "sigma_max": 6.0, "rho": 3.0},
        "model": {
            "num_features": 8,
            "num_radial_basis": 4,
            "num_layers": 1,
            "num_elements": 22,
            "r_max_angstrom": 8.0,
            "time_init_mode": "node",
            "parity_breaking": True,
            "unique_atom_indices": True,
            "conservative": False,
            "com_hidden_dims": [8],
            "com_time_embedding_dim": 8,
        },
        "target": {
            "mode": "openmm_bms_chirality",
            "openmm": {
                "pdb": str(pdb_path),
                "pdb_sha256": file_sha256(pdb_path),
                "device": "cpu",
                "forcefield": "amber99sbildn.xml",
                "implicit_forcefield": "amber99_obc.xml",
                "expected_particles": 22,
                "expected_constraints": 0,
            },
            "improper_restraints": {},
        },
    }


def test_all_atom_runtime_builds_without_importing_openmm() -> None:
    pdb_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "ala2"
        / "ala2_ref_bms.pdb"
    )
    system = build_runtime_system(
        _all_atom_runtime_config(pdb_path),
        key=jax.random.PRNGKey(41),
        load_potential=False,
    )
    assert system.event_shape == (22, 3)
    assert system.source.event_size == 66
    assert system.source.scale == 1.0
    assert system.transform is not None
    assert system.transform.shape_dim == 63
    assert system.transform.com_dim == 3
    assert system.controller.com_coordinates == "orthogonal"
    assert system.potential is None
    assert system.identity["density_mode"] == "ambient_66d_aux_com_exact"
    assert system.identity["target_terms"][-1] == "gaussian_com_auxiliary"

    state = system.source.sample(jax.random.PRNGKey(42), 2)
    output = system.apply(
        system.initial_variables,
        jnp.asarray([0.2, 0.8], dtype=state.dtype),
        state,
    )
    assert output.shape == state.shape
    assert bool(jnp.all(jnp.isfinite(output)))


def test_all_atom_runtime_rejects_nonstandard_source_or_com() -> None:
    pdb_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "ala2"
        / "ala2_ref_bms.pdb"
    )
    config = _all_atom_runtime_config(pdb_path)
    config["source"]["sigma"] = 0.9
    with pytest.raises(ValueError, match=r"N\(0,I_66\)"):
        build_runtime_system(config, load_potential=False)

    config = _all_atom_runtime_config(pdb_path)
    config["coordinates"]["com_sigma"] = 0.25
    with pytest.raises(ValueError, match=r"com_sigma=1/sqrt\(22\)"):
        build_runtime_system(config, load_potential=False)


def test_openmm_spec_defaults_match_official_boltzkit_target() -> None:
    spec = OpenMMAla2Spec(pdb_path="unused.pdb")
    assert spec.forcefield == "amber99sbildn.xml"
    assert spec.implicit_forcefield == "amber99_obc.xml"
    assert spec.expected_constraints == 0


def test_openmm_backend_is_unconstrained_and_translation_invariant() -> None:
    pytest.importorskip("openmm")
    openmm_app = pytest.importorskip("openmm.app")
    openmm_unit = pytest.importorskip("openmm.unit")
    pdb_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "ala2"
        / "ala2_ref_bms.pdb"
    )
    backend = OpenMMAla2Backend(
        OpenMMAla2Spec(pdb_path=str(pdb_path), device="cpu")
    )
    assert backend.num_particles == 22
    assert backend.num_constraints == 0
    pdb = openmm_app.PDBFile(str(pdb_path))
    coordinates = np.asarray(
        pdb.positions.value_in_unit(openmm_unit.angstrom), dtype=np.float32
    )[None]
    energy, gradient = backend.energy_and_grad_angstrom(coordinates)
    shifted_energy, shifted_gradient = backend.energy_and_grad_angstrom(
        coordinates + np.asarray([[[0.7, -0.3, 0.2]]], dtype=np.float32)
    )
    assert np.isfinite(energy).all()
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(energy, shifted_energy, rtol=2e-5, atol=2e-3)
    np.testing.assert_allclose(
        gradient, shifted_gradient, rtol=2e-4, atol=2e-3
    )


def test_amboltz_and_bms_atom_orders_have_openmm_energy_force_parity() -> None:
    """Optional real-asset audit for the TAM->BMS gather and inverse gradient."""

    pytest.importorskip("openmm")
    archive_value = os.environ.get("CG_BMS_AMBOLTZ_REFERENCE")
    tam_pdb_value = os.environ.get("CG_BMS_AMBOLTZ_PDB")
    if not archive_value or not tam_pdb_value:
        pytest.skip(
            "set CG_BMS_AMBOLTZ_REFERENCE and CG_BMS_AMBOLTZ_PDB "
            "to run the real ordering parity audit"
        )
    archive_path = Path(archive_value)
    tam_pdb_path = Path(tam_pdb_value)
    bms_pdb_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "ala2"
        / "ala2_ref_bms.pdb"
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        tam_angstrom = np.asarray(archive["R"][0:1], dtype=np.float64) * 10.0
    bms_angstrom = reorder_ala2_tam_to_bms(tam_angstrom)

    def backend(pdb_path: Path) -> OpenMMAla2Backend:
        return OpenMMAla2Backend(
            OpenMMAla2Spec(
                pdb_path=str(pdb_path),
                device="cpu",
                forcefield="amber99sbildn.xml",
                implicit_forcefield="implicit/obc1.xml",
            )
        )

    energy_tam, gradient_tam = backend(tam_pdb_path).energy_and_grad_angstrom(
        tam_angstrom
    )
    energy_bms, gradient_bms = backend(bms_pdb_path).energy_and_grad_angstrom(
        bms_angstrom
    )
    gradient_bms_in_tam_order = reorder_ala2_bms_to_tam(gradient_bms)
    energy_error = float(np.max(np.abs(energy_bms - energy_tam)))
    gradient_difference = gradient_bms_in_tam_order - gradient_tam
    gradient_max_error = float(np.max(np.abs(gradient_difference)))
    gradient_l2_error = float(np.linalg.norm(gradient_difference))
    print(
        "OpenMM TAM/BMS ordering parity: "
        f"energy_abs={energy_error:.6g}, "
        f"gradient_max={gradient_max_error:.6g}, "
        f"gradient_l2={gradient_l2_error:.6g}"
    )
    # Reordering changes OpenMM's floating-point force accumulation order, so
    # CPU results are not bitwise identical.  These absolute tolerances are
    # several orders below the force magnitudes while still catching a wrong
    # topology, atom permutation, or implicit-solvent definition.
    assert energy_error < 1.0e-4, f"energy ordering error={energy_error}"
    assert gradient_max_error < 1.0e-4, (
        f"gradient ordering max error={gradient_max_error}, "
        f"l2 error={gradient_l2_error}"
    )
