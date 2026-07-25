"""Full-rank EDM reference diffusion used by Bridge Matching Samplers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class EDMSDE:
    r"""Zero-drift EDM diffusion ``dX_t = g(t) dW_t``.

    Unlike the molecular BMS implementation's COM-free process, this version
    uses independent Brownian noise in every ambient coordinate.  CG Ala2 has
    18 degrees of freedom and all-atom Ala2 has all 66.
    """

    sigma_min: float = 1.0e-3
    sigma_max: float = 1.0
    rho: float = 7.0

    def __post_init__(self) -> None:
        values = (float(self.sigma_min), float(self.sigma_max), float(self.rho))
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("sigma_min, sigma_max and rho must be finite and positive")
        if self.sigma_min >= self.sigma_max:
            raise ValueError("sigma_min must be smaller than sigma_max")

    def diffusion(self, time: Array | float) -> Array:
        """Evaluate the scalar diffusion coefficient ``g(t)``."""

        time = jnp.asarray(time)
        maximum = jnp.asarray(self.sigma_max, dtype=time.dtype)
        minimum = jnp.asarray(self.sigma_min, dtype=time.dtype)
        rho = jnp.asarray(self.rho, dtype=time.dtype)
        interpolation = (1.0 - time) * maximum ** (1.0 / rho) + time * minimum ** (1.0 / rho)
        return interpolation**rho

    def integrated_variance(self, time: Array | float) -> Array:
        r"""Return ``kappa(t) = integral_0^t g(s)^2 ds`` analytically."""

        time = jnp.asarray(time)
        maximum = jnp.asarray(self.sigma_max, dtype=time.dtype)
        minimum = jnp.asarray(self.sigma_min, dtype=time.dtype)
        rho = jnp.asarray(self.rho, dtype=time.dtype)
        numerator = maximum ** (2.0 + 1.0 / rho) - self.diffusion(time) ** (2.0 + 1.0 / rho)
        denominator = (maximum ** (1.0 / rho) - minimum ** (1.0 / rho)) * (2.0 * rho + 1.0)
        return numerator / denominator

    def remaining_variance(self, time: Array | float) -> Array:
        r"""Return ``kappa(1) - kappa(t)`` without terminal cancellation.

        Forming this quantity as ``total_variance - integrated_variance(time)``
        loses all significant digits in float32 near ``t=1`` for the Ala2 EDM
        schedule.  The algebraically equivalent expression below subtracts
        quantities on the scale of the remaining variance instead.  This is
        also the quantity used in the terminal conditional score.
        """

        time = jnp.asarray(time)
        maximum = jnp.asarray(self.sigma_max, dtype=time.dtype)
        minimum = jnp.asarray(self.sigma_min, dtype=time.dtype)
        rho = jnp.asarray(self.rho, dtype=time.dtype)
        maximum_root = maximum ** (1.0 / rho)
        minimum_root = minimum ** (1.0 / rho)
        interpolation = (1.0 - time) * maximum_root + time * minimum_root
        power = 2.0 * rho + 1.0
        numerator = interpolation**power - minimum_root**power
        denominator = (maximum_root - minimum_root) * power
        return numerator / denominator

    @property
    def total_variance(self) -> Array:
        """Variance accumulated over the unit time interval."""

        return self.integrated_variance(1.0)

    def variance_fraction(self, time: Array | float) -> Array:
        """Return the Brownian-bridge time ``kappa(t) / kappa(1)``."""

        return self.integrated_variance(time) / self.total_variance
