"""A small functional replay buffer that remains on the JAX device."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class ReplayBufferState(NamedTuple):
    """Ring-buffer arrays plus device scalar cursor and current size."""

    storage: dict[str, Array]
    cursor: Array
    size: Array

    @property
    def capacity(self) -> int:
        first = next(iter(self.storage.values()))
        return int(first.shape[0])


def _validate_batch(batch: Mapping[str, Array]) -> int:
    if not batch:
        raise ValueError("batch must contain at least one field")
    sizes = {int(jnp.asarray(value).shape[0]) for value in batch.values()}
    if len(sizes) != 1:
        raise ValueError("all replay-buffer fields must have the same batch dimension")
    batch_size = sizes.pop()
    if batch_size <= 0:
        raise ValueError("batch must not be empty")
    return batch_size


def init_replay_buffer(capacity: int, example_batch: Mapping[str, Array]) -> ReplayBufferState:
    """Allocate device arrays using ``example_batch`` for shapes and dtypes."""

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    _validate_batch(example_batch)
    storage = {
        name: jnp.zeros((capacity, *jnp.asarray(value).shape[1:]), dtype=jnp.asarray(value).dtype)
        for name, value in example_batch.items()
    }
    return ReplayBufferState(
        storage=storage,
        cursor=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )


def extend_replay_buffer(
    state: ReplayBufferState,
    batch: Mapping[str, Array],
) -> ReplayBufferState:
    """Append a static-size batch, overwriting the oldest slots on wraparound."""

    if set(batch) != set(state.storage):
        raise ValueError("batch fields must exactly match replay-buffer fields")
    batch_size = _validate_batch(batch)
    capacity = state.capacity

    # JAX scatter semantics are ambiguous for duplicate indices.  Keeping only
    # the newest ``capacity`` rows avoids duplicates when a caller supplies a
    # larger static batch.
    if batch_size > capacity:
        batch = {name: jnp.asarray(value)[-capacity:] for name, value in batch.items()}
        batch_size = capacity

    indices = (state.cursor + jnp.arange(batch_size, dtype=jnp.int32)) % capacity
    # Cast incoming leaves to the allocated storage dtype.  This makes the
    # buffer contract explicit and avoids a deferred unsafe scatter cast when
    # JAX's global x64 setting differs between data construction and training.
    storage = {
        name: current.at[indices].set(jnp.asarray(batch[name], dtype=current.dtype))
        for name, current in state.storage.items()
    }
    return ReplayBufferState(
        storage=storage,
        cursor=(state.cursor + batch_size) % capacity,
        size=jnp.minimum(jnp.asarray(capacity, dtype=jnp.int32), state.size + batch_size),
    )


def sample_replay_buffer(
    key: Array,
    state: ReplayBufferState,
    batch_size: int,
) -> dict[str, Array]:
    """Sample with replacement from a non-empty valid buffer prefix.

    ``state.size > 0`` is a precondition so the function remains usable inside
    ``jax.jit`` without host-side scalar extraction.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if "valid" in state.storage:
        available = jnp.arange(state.capacity, dtype=jnp.int32) < state.size
        valid = available & jnp.asarray(state.storage["valid"], dtype=bool)
        logits = jnp.where(valid, 0.0, -jnp.inf)
        indices = jax.random.categorical(key, logits, shape=(batch_size,))
    else:
        indices = jax.random.randint(key, (batch_size,), minval=0, maxval=state.size)
    return {name: values[indices] for name, values in state.storage.items()}
