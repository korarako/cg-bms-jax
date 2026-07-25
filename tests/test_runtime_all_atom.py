from __future__ import annotations

import math
from pathlib import Path

import jax
import numpy as np

from cg_bms_jax import runtime
from cg_bms_jax.experiment.training_support import asset_provenance
from cg_bms_jax.runtime import (
    build_runtime_system,
    resolve_project_or_package_path,
)


def _experiment() -> dict[str, object]:
    return {
        "name": "ala2_aa66_test",
        "state_shape": [22, 3],
        "coordinate_mode": "ala2_aa_ambient66",
        "temperature_kelvin": 300.0,
        "kT": 2.494338785445972,
        "source": {"mean": 0.0, "sigma": 1.0},
        "sde": {
            "sigma_min": 0.001,
            "sigma_max": 6.0,
            "rho": 3.0,
            "steps": 2,
        },
        "model": {
            "num_features": 8,
            "num_radial_basis": 4,
            "num_layers": 1,
            "num_elements": 22,
            "r_max_angstrom": 8.0,
            "r_offset_angstrom": 0.0,
            "time_init_mode": "node",
            "parity_breaking": True,
            "unique_atom_indices": True,
            "conservative": False,
            "com_hidden_dims": [8],
            "com_time_embedding_dim": 8,
        },
        "coordinates": {
            "physical_std_angstrom": 1.0,
            "com_sigma": 1.0 / math.sqrt(22.0),
        },
        "assets": {
            "reference": "../amboltz/data/ala2/implicit_obc1_openmm.npz",
        },
        "bridge_data": {
            "path": "../amboltz/data/ala2/implicit_obc1_openmm.npz",
            "sha256": (
                "75c9d837e58e01dfca9dadd6faf6fa2011d3567b2fe7e4ee88cc891c803c62e4"
            ),
        },
        "target": {
            "mode": "openmm_bms_chirality",
            "implementation_abi": (
                "ala2_aa_openmm_bms_chirality_augmented_com_v1"
            ),
            "openmm": {
                "pdb": "data/ala2/ala2_ref_bms.pdb",
                "pdb_sha256": (
                    "e42b95379c4e755f59530ec511d335832d5da2fad58415a11257029d133a65c8"
                ),
                "device": "cpu",
                "forcefield": "amber99sbildn.xml",
                "implicit_forcefield": "implicit/obc1.xml",
                "expected_particles": 22,
                "expected_constraints": 0,
            },
            "improper_restraints": {
                "cb": {"enabled": True},
                "ha": {"enabled": True},
            },
        },
        "training": {
            "outer_iterations": 1,
            "gradient_steps": 1,
            "batch_size": 1,
            "rollout_samples": 1,
            "buffer_capacity": 2,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "damping": 0.0,
        },
    }


def test_build_all_atom_runtime_without_importing_openmm() -> None:
    system = build_runtime_system(
        {"experiment": _experiment()},
        key=jax.random.PRNGKey(4),
        load_potential=False,
    )
    assert system.event_shape == (22, 3)
    assert system.potential is None
    assert system.transform is not None
    assert system.transform.ambient_dim == 66
    assert system.transform.shape_dim == 63
    assert system.transform.com_dim == 3
    assert system.identity["density_mode"] == "ambient_66d_aux_com_exact"
    assert system.identity["atom_order"] == "official_bms_22"

    sample = system.source.sample(jax.random.PRNGKey(5), 3)
    prediction = system.apply(system.initial_variables, 0.5, sample)
    assert prediction.shape == (3, 22, 3)
    assert np.isfinite(np.asarray(prediction)).all()


def test_all_atom_asset_provenance_records_full_rank_measure() -> None:
    system = build_runtime_system(
        {"experiment": _experiment()},
        key=jax.random.PRNGKey(6),
        load_potential=False,
    )
    provenance = asset_provenance(system)
    assert provenance.num_particles == 22
    assert provenance.spatial_dimension == 3
    assert provenance.coordinate_unit == "angstrom"
    assert provenance.density_mode == "ambient_66d_aux_com_exact"
    assert provenance.mapping_indices == tuple(range(22))


def test_wheel_bundled_pdb_fallback_is_independent_of_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "site-packages" / "cg_bms_jax"
    bundled = package / "data" / "ala2" / "ala2_ref_bms.pdb"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("MODEL\nENDMDL\n", encoding="ascii")

    monkeypatch.setattr(runtime, "__file__", str(package / "runtime.py"))
    monkeypatch.setenv("CG_BMS_PROJECT_ROOT", str(tmp_path / "empty-project"))

    resolved = resolve_project_or_package_path(
        "data/ala2/ala2_ref_bms.pdb"
    )
    assert resolved == bundled.resolve()
