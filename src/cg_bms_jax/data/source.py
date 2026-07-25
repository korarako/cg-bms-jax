"""Analytic full-rank Gaussian source distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class GaussianSource:
    """An isotropic Gaussian over an explicitly full-rank event space.

    The source used by the 18-dimensional Ala2 construction has
    ``event_shape=(6, 3)``, ``mean=0`` and ``scale=1``.  No centering or
    projection is performed: the three centre-of-geometry degrees of freedom
    are part of the event and therefore part of its density.
    """

    event_shape: tuple[int, ...]
    scale: float = 1.0
    mean: Any = 0.0

    def __post_init__(self) -> None:
        if not self.event_shape or any(int(size) <= 0 for size in self.event_shape):
            raise ValueError("event_shape must contain positive dimensions")
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError("scale must be finite and positive")

        mean = jnp.asarray(self.mean)
        if mean.ndim != 0 and tuple(mean.shape) != tuple(self.event_shape):
            raise ValueError(f"mean must be scalar or have shape {self.event_shape}")

    @property
    def event_size(self) -> int:
        """Number of independent scalar coordinates in one event."""

        return math.prod(self.event_shape)

    def _mean(self, dtype: jnp.dtype) -> Array:
        return jnp.broadcast_to(jnp.asarray(self.mean, dtype=dtype), self.event_shape)

    def sample(
        self,
        key: Array,
        batch_size: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> Array:
        """Draw ``batch_size`` independent samples."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        noise = jax.random.normal(key, (batch_size, *self.event_shape), dtype=dtype)
        return self._mean(dtype) + jnp.asarray(self.scale, dtype=dtype) * noise

    def log_prob(self, samples: Array) -> Array:
        """Evaluate log density, summing exactly over the event dimensions."""

        samples = jnp.asarray(samples)
        event_ndim = len(self.event_shape)
        if samples.ndim < event_ndim or tuple(samples.shape[-event_ndim:]) != self.event_shape:
            raise ValueError(f"samples must end in event shape {self.event_shape}")

        scale = jnp.asarray(self.scale, dtype=samples.dtype)
        centered = (samples - self._mean(samples.dtype)) / scale
        axes = tuple(range(samples.ndim - event_ndim, samples.ndim))
        quadratic = jnp.sum(jnp.square(centered), axis=axes)
        log_normalizer = self.event_size * (jnp.log(scale) + 0.5 * math.log(2.0 * math.pi))
        return -0.5 * quadratic - log_normalizer

    def score(self, samples: Array) -> Array:
        """Return ``grad_samples log_prob(samples)`` analytically."""

        samples = jnp.asarray(samples)
        event_ndim = len(self.event_shape)
        if samples.ndim < event_ndim or tuple(samples.shape[-event_ndim:]) != self.event_shape:
            raise ValueError(f"samples must end in event shape {self.event_shape}")
        scale = jnp.asarray(self.scale, dtype=samples.dtype)
        return -(samples - self._mean(samples.dtype)) / jnp.square(scale)
