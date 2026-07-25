from __future__ import annotations

import copy

import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.coordinates import cgbg_correct_log_density
from cg_bms_jax.reweight import (
    build_reweight_payload,
    compute_cgbg_compat_weights,
    compute_exact_ambient_weights,
    load_reweight_archive,
    save_reweight_archive,
    validate_reweight_payload,
)


def _domain_metadata() -> dict[str, object]:
    return {
        "experiment": "ala2_cb",
        "domain": {
            "box": [3.0, 3.0, 3.0],
            "anchor": 0,
            "margin": 0.2,
            "support_mode": "relative_fundamental",
        },
    }


def test_exact_ambient_weights_are_stable_softmax() -> None:
    reduced_energy = jnp.asarray([4.0, 2.0, 1.0, 8.0])
    logq = jnp.asarray([-2.0, -1.0, -4.0, 1000.0])
    valid = jnp.asarray([True, True, True, False])
    result = compute_exact_ambient_weights(reduced_energy, logq, valid_mask=valid)

    raw = -np.asarray(reduced_energy[:3]) - np.asarray(logq[:3])
    expected = np.exp(raw - raw.max())
    expected /= expected.sum()
    np.testing.assert_allclose(np.asarray(result.weights[:3]), expected, rtol=1e-6)
    assert float(result.weights[3]) == 0.0
    assert np.isneginf(np.asarray(result.logw)[3])
    assert result.density_mode == "ambient_exact"


def test_float32_large_log_weights_remain_normalized() -> None:
    raw = jnp.asarray(
        [100000.0, 99999.75, 99999.5, 99999.0],
        dtype=jnp.float32,
    )
    result = compute_exact_ambient_weights(jnp.zeros_like(raw), -raw)
    weights = np.asarray(result.weights)

    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.exp(np.asarray(result.logw)), weights, rtol=1.0e-6)


def test_cgbg_compat_uses_radial_com_and_scale_correction() -> None:
    x = jnp.zeros((3, 6, 3)).at[:, :, 0].set(jnp.asarray([0.1, 0.2, 0.4])[:, None])
    energy = jnp.asarray([1.5, 2.0, 3.5])
    logq_ambient = jnp.asarray([-5.0, -4.0, -3.0])
    result = compute_cgbg_compat_weights(
        energy,
        2.5,
        logq_ambient,
        x,
        0.2,
    )
    corrected = cgbg_correct_log_density(logq_ambient, x, 0.2)
    expected_raw = -energy / 2.5 - corrected
    np.testing.assert_allclose(result.proposal_log_density, corrected)
    np.testing.assert_allclose(result.logw_raw, expected_raw)
    np.testing.assert_allclose(jnp.sum(result.weights), 1.0, atol=1e-6)
    assert result.density_mode == "cgbg_compat"


def test_archive_rejects_sde_and_preserves_domain_metadata(tmp_path) -> None:
    coordinates = np.zeros((3, 6, 3), dtype=np.float64)
    energy = np.asarray([1.0, 2.0, 4.0])
    logq = np.asarray([-2.0, -3.0, -4.0])
    support = np.asarray([True, False, True])
    result = compute_exact_ambient_weights(energy, logq, valid_mask=support)
    payload = build_reweight_payload(
        physical_coordinates=coordinates,
        energy=energy,
        logq_ambient=logq,
        result=result,
        metadata=_domain_metadata(),
        standardized_ambient=coordinates,
        support_mask=support,
    )
    assert payload["weights"][1] == 0.0
    assert validate_reweight_payload(payload)["domain"]["anchor"] == 0

    path = save_reweight_archive(tmp_path / "samples_and_weights.npz", payload)
    loaded = load_reweight_archive(
        path,
        expected_density_mode="ambient_exact",
        expected_metadata={"experiment": "ala2_cb"},
    )
    np.testing.assert_allclose(loaded["weights"], payload["weights"])

    invalid = copy.deepcopy(payload)
    invalid["sampler_kind"] = np.asarray("sde")
    with pytest.raises(ValueError, match="PF-ODE"):
        validate_reweight_payload(invalid)


def test_archive_requires_out_of_domain_weights_to_be_zero() -> None:
    energy = np.asarray([0.0, 0.0])
    logq = np.asarray([0.0, 0.0])
    result = compute_exact_ambient_weights(energy, logq)
    with pytest.raises(ValueError, match="Out-of-domain"):
        build_reweight_payload(
            physical_coordinates=np.zeros((2, 6, 3)),
            energy=energy,
            logq_ambient=logq,
            result=result,
            metadata=_domain_metadata(),
            support_mask=np.asarray([True, False]),
        )


def test_archive_rejects_all_invalid_zero_weight_result() -> None:
    energy = np.asarray([np.nan, np.inf])
    logq = np.asarray([0.0, 0.0])
    result = compute_exact_ambient_weights(energy, logq)
    with pytest.raises(ValueError, match="no valid"):
        build_reweight_payload(
            physical_coordinates=np.zeros((2, 6, 3)),
            energy=energy,
            logq_ambient=logq,
            result=result,
            metadata=_domain_metadata(),
        )


def test_canonical_archive_enforces_formal_target_component_sum() -> None:
    raw = np.asarray([-300.0, -310.0, -320.0])
    effective = raw + np.asarray([0.0, 0.01, 0.2])
    bond = np.asarray([0.0, 1.0, 2.0])
    angle = np.asarray([0.0, 0.5, 1.0])
    repulsion = np.asarray([0.0, 0.25, 0.5])
    support = bond + angle + repulsion
    cb = np.zeros(3)
    target = effective + support + cb
    logq = np.asarray([-2.0, -3.0, -4.0])
    result = compute_exact_ambient_weights(target / 2.5, logq)
    metadata = {
        **_domain_metadata(),
        "target_mode": "pmf_canonical_support",
        "formal_target_signature": "formal-v1",
        "training_target_signature": "training-v1",
        "topology_signature": "topology-v1",
    }
    components = {
        "U_pmf": raw,
        "U_pmf_raw": raw,
        "U_pmf_effective": effective,
        "U_pmf_floor_delta": effective - raw,
        "U_pmf_transform_delta": effective - raw,
        "pmf_topology_gate": np.ones(3),
        "U_pmf_gated": effective,
        "U_bond": bond,
        "U_angle": angle,
        "U_repulsion": repulsion,
        "U_support": support,
        "U_cb": cb,
        "U_target": target,
    }
    payload = build_reweight_payload(
        physical_coordinates=np.zeros((3, 6, 3)),
        energy=target,
        logq_ambient=logq,
        result=result,
        metadata=metadata,
        standardized_ambient=np.zeros((3, 6, 3)),
        extra=components,
    )
    assert validate_reweight_payload(payload)["formal_target_signature"] == "formal-v1"

    corrupted = copy.deepcopy(payload)
    corrupted["U_bond"] = np.asarray(corrupted["U_bond"]) + 0.1
    with pytest.raises(ValueError, match="U_support component sum"):
        validate_reweight_payload(corrupted)


def test_additive_v3_archive_requires_scaled_pmf_without_gate_components() -> None:
    raw = np.asarray([-300.0, -310.0, -320.0])
    effective = raw + np.asarray([0.0, 0.01, 0.2])
    pmf_scale = 0.25
    scaled = pmf_scale * effective
    bond = np.asarray([0.0, 1.0, 2.0])
    angle = np.asarray([0.0, 0.5, 1.0])
    repulsion = np.asarray([0.0, 0.25, 0.5])
    support = bond + angle + repulsion
    cb = np.asarray([0.0, 0.1, 0.2])
    target = scaled + support + cb
    logq = np.asarray([-2.0, -3.0, -4.0])
    result = compute_exact_ambient_weights(target / 2.5, logq)
    metadata = {
        **_domain_metadata(),
        "target_mode": "pmf_canonical_support",
        "target_implementation_abi": "ala2_canonical_additive_v3",
        "pmf_scale": pmf_scale,
        "formal_target_signature": "formal-v3",
        "training_target_signature": "training-v3",
        "topology_signature": "topology-v1",
    }
    components = {
        "U_pmf": raw,
        "U_pmf_raw": raw,
        "U_pmf_effective": effective,
        "U_pmf_floor_delta": effective - raw,
        "U_pmf_transform_delta": effective - raw,
        "U_pmf_scaled": scaled,
        "U_bond": bond,
        "U_angle": angle,
        "U_repulsion": repulsion,
        "U_support": support,
        "U_cb": cb,
        "U_target": target,
    }
    payload = build_reweight_payload(
        physical_coordinates=np.zeros((3, 6, 3)),
        energy=target,
        logq_ambient=logq,
        result=result,
        metadata=metadata,
        standardized_ambient=np.zeros((3, 6, 3)),
        extra=components,
    )

    assert "pmf_topology_gate" not in payload
    assert "U_pmf_gated" not in payload
    assert validate_reweight_payload(payload)["pmf_scale"] == pmf_scale

    corrupted = copy.deepcopy(payload)
    corrupted["U_pmf_scaled"] = np.asarray(corrupted["U_pmf_scaled"]) + 0.1
    with pytest.raises(ValueError, match="U_pmf_scaled"):
        validate_reweight_payload(corrupted)


def test_additive_v3_archive_rejects_missing_or_invalid_pmf_scale() -> None:
    raw = np.asarray([-300.0, -310.0])
    support = np.asarray([0.0, 1.0])
    cb = np.asarray([0.0, 0.2])
    target = raw + support + cb
    logq = np.asarray([-2.0, -3.0])
    result = compute_exact_ambient_weights(target / 2.5, logq)
    metadata = {
        **_domain_metadata(),
        "target_mode": "pmf_canonical_support",
        "target_implementation_abi": "ala2_canonical_additive_v3",
        "formal_target_signature": "formal-v3",
        "training_target_signature": "training-v3",
        "topology_signature": "topology-v1",
    }
    components = {
        "U_pmf": raw,
        "U_pmf_raw": raw,
        "U_pmf_effective": raw,
        "U_pmf_floor_delta": np.zeros_like(raw),
        "U_pmf_transform_delta": np.zeros_like(raw),
        "U_pmf_scaled": raw,
        "U_bond": support,
        "U_angle": np.zeros_like(raw),
        "U_repulsion": np.zeros_like(raw),
        "U_support": support,
        "U_cb": cb,
        "U_target": target,
    }
    with pytest.raises(ValueError, match="missing pmf_scale"):
        build_reweight_payload(
            physical_coordinates=np.zeros((2, 6, 3)),
            energy=target,
            logq_ambient=logq,
            result=result,
            metadata=metadata,
            extra=components,
        )

    metadata["pmf_scale"] = float("nan")
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        build_reweight_payload(
            physical_coordinates=np.zeros((2, 6, 3)),
            energy=target,
            logq_ambient=logq,
            result=result,
            metadata=metadata,
            extra=components,
        )

    metadata["pmf_scale"] = -0.1
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        build_reweight_payload(
            physical_coordinates=np.zeros((2, 6, 3)),
            energy=target,
            logq_ambient=logq,
            result=result,
            metadata=metadata,
            extra=components,
        )


def test_all_atom_archive_enforces_exact_66d_target_decomposition() -> None:
    kT = 2.494338785445972
    openmm = np.asarray([12.0, 15.0, 9.0])
    improper = np.asarray([0.0, 2.0, 1.0])
    target = openmm + improper
    com_reduced = np.asarray([0.2, 1.1, 0.4])
    reduced = target / kT + com_reduced
    logq = np.asarray([-20.0, -22.0, -19.0])
    result = compute_exact_ambient_weights(reduced, logq)
    metadata = {
        **_domain_metadata(),
        "target_mode": "openmm_bms_chirality",
        "state_density_mode": "ambient_66d_aux_com_exact",
        "density_mode": "ambient_exact",
        "kT_kj_mol": kT,
        "formal_target_signature": "aa-formal",
        "training_target_signature": "aa-training",
        "topology_signature": "aa-topology",
        "atom_order_signature": "aa-order",
    }
    components = {
        "U_openmm": openmm,
        "U_improper": improper,
        "U_target": target,
        "U_com_reduced": com_reduced,
        "target_reduced_energy": reduced,
    }
    payload = build_reweight_payload(
        physical_coordinates=np.zeros((3, 22, 3)),
        energy=target,
        logq_ambient=logq,
        result=result,
        metadata=metadata,
        standardized_ambient=np.zeros((3, 22, 3)),
        extra=components,
    )
    assert (
        validate_reweight_payload(payload)["state_density_mode"]
        == "ambient_66d_aux_com_exact"
    )

    corrupted = copy.deepcopy(payload)
    corrupted["target_reduced_energy"] = (
        np.asarray(corrupted["target_reduced_energy"]) + 0.1
    )
    with pytest.raises(ValueError, match="target_reduced_energy"):
        validate_reweight_payload(corrupted)

    corrupted = copy.deepcopy(payload)
    corrupted["logq_ambient"] = np.asarray(corrupted["logq_ambient"]) + 0.1
    with pytest.raises(ValueError, match="logp == logq_ambient"):
        validate_reweight_payload(corrupted)

    corrupted = copy.deepcopy(payload)
    corrupted["logw_raw"] = np.asarray(corrupted["logw_raw"]) + 0.1
    with pytest.raises(ValueError, match="logw_raw"):
        validate_reweight_payload(corrupted)
