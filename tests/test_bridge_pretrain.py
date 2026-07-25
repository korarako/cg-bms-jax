from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cg_bms_jax.data import GaussianSource
from cg_bms_jax.process import EDMSDE, conditional_score_1t
from cg_bms_jax.training import (
    BridgePretrainConfig,
    make_array_endpoint_provider,
    remaining_bridge_variance,
    train_bridge_pretrain,
    variance_weighted_bridge_loss,
)


def test_conditional_score_1t_matches_closed_form() -> None:
    sde = EDMSDE(sigma_min=0.01, sigma_max=0.8, rho=3.0)
    endpoint_1 = jnp.asarray(
        [[[0.7, -0.2], [0.3, 0.8]], [[-0.4, 0.6], [1.1, -0.5]]],
        dtype=jnp.float64,
    )
    state_t = endpoint_1 + jnp.asarray(
        [[[0.1, -0.3], [0.2, 0.4]], [[-0.2, 0.1], [0.5, -0.1]]],
        dtype=jnp.float64,
    )
    time = jnp.asarray([0.2, 0.83], dtype=jnp.float64)
    remaining = sde.remaining_variance(time[:, None, None])
    expected = (endpoint_1 - state_t) / remaining

    np.testing.assert_allclose(
        conditional_score_1t(sde, time, endpoint_1, state_t),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        remaining_bridge_variance(sde, time, endpoint_1),
        remaining,
        rtol=1e-12,
        atol=1e-12,
    )


def test_variance_weighted_bridge_loss_matches_explicit_mean() -> None:
    prediction = jnp.asarray([[1.0, -1.0], [2.0, 0.5]], dtype=jnp.float64)
    target = jnp.asarray([[0.0, 1.0], [1.5, -0.5]], dtype=jnp.float64)
    remaining = jnp.asarray([0.25, 0.75], dtype=jnp.float64)
    expected = np.mean(
        np.asarray(remaining)[:, None]
        * np.square(np.asarray(prediction) - np.asarray(target))
    )
    actual = variance_weighted_bridge_loss(prediction, target, remaining)
    assert float(actual) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)


def test_array_endpoint_provider_is_key_deterministic_and_samples_rows() -> None:
    endpoints = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)
    provider = make_array_endpoint_provider(endpoints)
    first = provider(jax.random.PRNGKey(3), 32)
    second = provider(jax.random.PRNGKey(3), 32)

    np.testing.assert_array_equal(first, second)
    endpoint_rows = {tuple(row) for row in np.asarray(endpoints)}
    assert all(tuple(row) in endpoint_rows for row in np.asarray(first))


def _tiny_control(params, _constants, time, state):
    time = jnp.asarray(time, dtype=state.dtype)
    time = time.reshape((state.shape[0],) + (1,) * (state.ndim - 1))
    return params["state_scale"] * state + params["time_scale"] * time + params["bias"]


def _run_tiny_pretrain(seed: int):
    target_key = jax.random.PRNGKey(44)
    target_endpoints = 0.35 * jax.random.normal(
        target_key,
        (128, 2),
        dtype=jnp.float32,
    ) + jnp.asarray([0.7, -0.4], dtype=jnp.float32)
    initial_params = {
        "state_scale": jnp.asarray(0.0, dtype=jnp.float32),
        "time_scale": jnp.asarray(0.0, dtype=jnp.float32),
        "bias": jnp.zeros((2,), dtype=jnp.float32),
    }
    result = train_bridge_pretrain(
        initial_params=initial_params,
        constants={},
        control_apply=_tiny_control,
        source=GaussianSource(event_shape=(2,), scale=1.0),
        sde=EDMSDE(sigma_min=0.05, sigma_max=0.5, rho=3.0),
        target_endpoints=target_endpoints,
        config=BridgePretrainConfig(
            updates=6,
            batch_size=24,
            learning_rate=2.0e-3,
            final_learning_rate=5.0e-4,
            learning_rate_warmup_updates=2,
            gradient_clip_norm=10.0,
            weight_decay=1.0e-3,
            max_time=0.99,
            progress_every=3,
            seed=seed,
        ),
    )
    return initial_params, result


def test_tiny_gaussian_pretrain_updates_parameters_is_finite_and_deterministic(capsys) -> None:
    initial_params, first = _run_tiny_pretrain(7)
    first_output = capsys.readouterr().out
    _, second = _run_tiny_pretrain(7)
    second_output = capsys.readouterr().out

    assert "[bridge-pretrain]" in first_output
    assert "step=6/6" in first_output
    assert first_output == second_output
    assert int(first.state.step) == 6
    assert len(first.metrics) == 6
    assert all(bool(item["finite"]) for item in first.metrics)

    initial_leaves = jax.tree_util.tree_leaves(initial_params)
    trained_leaves = jax.tree_util.tree_leaves(first.state.params)
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(initial_leaves, trained_leaves, strict=True)
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(first.state.params),
        jax.tree_util.tree_leaves(second.state.params),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    assert first.metrics == second.metrics


def test_nonfinite_endpoint_provider_trips_finite_guard() -> None:
    def bad_provider(_key, batch_size):
        return jnp.full((batch_size, 1), jnp.nan, dtype=jnp.float32)

    with pytest.raises(FloatingPointError, match="non-finite bridge-pretrain update"):
        train_bridge_pretrain(
            initial_params={"state_scale": jnp.asarray(0.0), "time_scale": jnp.asarray(0.0), "bias": jnp.zeros((1,))},
            constants={},
            control_apply=_tiny_control,
            source=GaussianSource(event_shape=(1,), scale=1.0),
            sde=EDMSDE(sigma_min=0.05, sigma_max=0.5, rho=3.0),
            endpoint_provider=bad_provider,
            config=BridgePretrainConfig(
                updates=1,
                batch_size=4,
                learning_rate_warmup_updates=0,
                progress_every=1,
            ),
        )


def test_periodic_checkpoint_callback_observes_fresh_steps_only() -> None:
    callbacks: list[tuple[int, int]] = []
    target_endpoints = jnp.zeros((16, 1), dtype=jnp.float32)

    result = train_bridge_pretrain(
        initial_params={
            "state_scale": jnp.asarray(0.0),
            "time_scale": jnp.asarray(0.0),
            "bias": jnp.zeros((1,)),
        },
        constants={},
        control_apply=_tiny_control,
        source=GaussianSource(event_shape=(1,), scale=1.0),
        sde=EDMSDE(sigma_min=0.05, sigma_max=0.5, rho=3.0),
        target_endpoints=target_endpoints,
        checkpoint_callback=lambda state, history: callbacks.append(
            (int(state.step), len(history))
        ),
        config=BridgePretrainConfig(
            updates=5,
            batch_size=4,
            learning_rate_warmup_updates=0,
            checkpoint_every=2,
            progress_every=5,
        ),
    )
    assert int(result.state.step) == 5
    assert callbacks == [(2, 2), (4, 4)]


def test_periodic_checkpoint_reserves_final_step_for_entrypoint_save() -> None:
    callbacks: list[tuple[int, int]] = []
    result = train_bridge_pretrain(
        initial_params={
            "state_scale": jnp.asarray(0.0),
            "time_scale": jnp.asarray(0.0),
            "bias": jnp.zeros((1,)),
        },
        constants={},
        control_apply=_tiny_control,
        source=GaussianSource(event_shape=(1,), scale=1.0),
        sde=EDMSDE(sigma_min=0.05, sigma_max=0.5, rho=3.0),
        target_endpoints=jnp.zeros((16, 1), dtype=jnp.float32),
        checkpoint_callback=lambda state, history: callbacks.append(
            (int(state.step), len(history))
        ),
        config=BridgePretrainConfig(
            updates=4,
            batch_size=4,
            learning_rate_warmup_updates=0,
            checkpoint_every=2,
            progress_every=4,
        ),
    )
    assert int(result.state.step) == 4
    assert callbacks == [(2, 2)]


def test_checkpoint_interval_requires_a_callback() -> None:
    with pytest.raises(ValueError, match="checkpoint_callback"):
        train_bridge_pretrain(
            initial_params={
                "state_scale": jnp.asarray(0.0),
                "time_scale": jnp.asarray(0.0),
                "bias": jnp.zeros((1,)),
            },
            constants={},
            control_apply=_tiny_control,
            source=GaussianSource(event_shape=(1,), scale=1.0),
            sde=EDMSDE(sigma_min=0.05, sigma_max=0.5, rho=3.0),
            target_endpoints=jnp.zeros((4, 1)),
            config=BridgePretrainConfig(
                updates=1,
                batch_size=4,
                learning_rate_warmup_updates=0,
                checkpoint_every=1,
            ),
        )
