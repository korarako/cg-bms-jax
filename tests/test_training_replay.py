from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from cg_bms_jax.data.buffer import ReplayBufferState
from cg_bms_jax.evaluation.training_replay import replay_endpoint_states


def test_replay_endpoint_states_respects_ring_order_validity_and_tail_limit():
    endpoint = jnp.arange(5, dtype=jnp.float32).reshape(5, 1)
    replay = ReplayBufferState(
        storage={
            "endpoint_1": endpoint,
            "valid": jnp.asarray([False, True, True, True, True]),
        },
        # A full ring with cursor=2 has chronological order 2,3,4,0,1.
        cursor=jnp.asarray(2, dtype=jnp.int32),
        size=jnp.asarray(5, dtype=jnp.int32),
    )

    selected = replay_endpoint_states(replay, max_samples=3)

    np.testing.assert_array_equal(selected[:, 0], np.asarray([3.0, 4.0, 1.0]))


def test_replay_endpoint_states_uses_only_initialized_prefix():
    replay = ReplayBufferState(
        storage={
            "endpoint_1": jnp.arange(5, dtype=jnp.float32).reshape(5, 1),
            "valid": jnp.ones(5, dtype=jnp.bool_),
        },
        cursor=jnp.asarray(3, dtype=jnp.int32),
        size=jnp.asarray(3, dtype=jnp.int32),
    )

    selected = replay_endpoint_states(replay, max_samples=None)

    np.testing.assert_array_equal(selected[:, 0], np.asarray([0.0, 1.0, 2.0]))
