from __future__ import annotations

import hashlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.potential import CGBGMBPotential, load_trusted_pickle, muller_brown_energy, rbf_mlp_apply


def _parameters() -> dict[str, object]:
    return {
        "params": {
            "RBF_0": {"centers": jnp.asarray([[-1.0], [0.5], [2.0]])},
            "Dense_0": {
                "kernel": jnp.asarray([[0.2, -0.3], [0.5, 0.1], [-0.4, 0.7]]),
                "bias": jnp.asarray([0.1, -0.2]),
            },
            "Dense_1": {
                "kernel": jnp.asarray([[1.2], [-0.8]]),
                "bias": jnp.asarray([0.3]),
            },
        }
    }


def test_rbf_mlp_matches_manual_composition() -> None:
    params = _parameters()
    x = jnp.asarray([-0.25, 1.25])
    centres = params["params"]["RBF_0"]["centers"]
    hidden = jnp.exp(-((x[:, None] - centres[:, 0][None, :]) ** 2) / (2.0 * 2.0**2))
    hidden = jax.nn.softplus(
        hidden @ params["params"]["Dense_0"]["kernel"] + params["params"]["Dense_0"]["bias"]
    )
    expected = (
        hidden @ params["params"]["Dense_1"]["kernel"] + params["params"]["Dense_1"]["bias"]
    )[:, 0]
    np.testing.assert_allclose(rbf_mlp_apply(params, x, sigma=2.0), expected, rtol=1e-6)


def test_mb_potential_score_and_gradient() -> None:
    potential = CGBGMBPotential(params=_parameters(), kT=2.5, sigma=2.0)
    x = jnp.asarray([-0.4, 0.8, 1.6])
    result = potential.evaluate(x)
    automatic = jax.grad(lambda values: jnp.sum(potential.energy(values)))(x)
    np.testing.assert_allclose(result.gradient, automatic, rtol=1e-6)
    np.testing.assert_allclose(result.score, -automatic / 2.5, rtol=1e-6)
    assert np.asarray(result.valid_mask).all()


def test_muller_brown_bias_is_the_documented_gaussian() -> None:
    xy = jnp.asarray([[32.0, 12.0], [20.0, 24.0]])
    difference = muller_brown_energy(xy, biased=True) - muller_brown_energy(xy)
    expected = -4.0 * jnp.exp(-((xy[:, 0] - 32.0) ** 2) / (2.0 * 5.0**2))
    np.testing.assert_allclose(difference, expected, rtol=1e-6)


def test_pickle_restore_requires_trust_and_exact_hash(tmp_path) -> None:
    path = tmp_path / "parameters.pkl"
    value = {"params": {"answer": 42}}
    path.write_bytes(pickle.dumps(value))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(PermissionError):
        load_trusted_pickle(path, expected_sha256=digest)
    with pytest.raises(ValueError, match="mismatch"):
        load_trusted_pickle(path, expected_sha256="0" * 64, trusted=True)
    assert load_trusted_pickle(path, expected_sha256=digest, trusted=True) == {"answer": 42}

