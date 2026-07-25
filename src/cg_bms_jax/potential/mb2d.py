"""Finite-domain analytic Muller--Brown target in an affine 2D state space.

The probability-flow and controller code operate in a well-conditioned state
coordinate ``z``.  This module keeps the published CG-BG Muller--Brown energy
in its physical ``(x, y)`` coordinates and applies the explicit affine map

``xy = offset + scale * z``.

The formal target is the analytic energy on a finite rectangular support.
Training may additionally use a flat-bottom quadratic wall that starts inside
that support, while a bounded C2 extension of the physical coordinates keeps
the analytic energy and its score finite outside the box.  Neither device is
included in formal importance weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .base import PotentialResult
from .mb import muller_brown_energy

Array = jax.Array


def _pair(value: Any, *, name: str, positive: bool = False) -> tuple[float, float]:
    """Canonicalize one scalar or length-two sequence."""

    if isinstance(value, (int, float)):
        result = (float(value), float(value))
    else:
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be a scalar or length-two sequence") from error
        if len(result) != 2:
            raise ValueError(f"{name} must contain exactly two values")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    if positive and not all(item > 0.0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _box(
    value: Any,
    *,
    name: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Canonicalize ``[[x_min, x_max], [y_min, y_max]]``."""

    try:
        axes = tuple(tuple(float(bound) for bound in axis) for axis in value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must have shape (2, 2)") from error
    if len(axes) != 2 or any(len(axis) != 2 for axis in axes):
        raise ValueError(f"{name} must have shape (2, 2)")
    if not all(math.isfinite(bound) for axis in axes for bound in axis):
        raise ValueError(f"{name} must contain finite bounds")
    if any(lower >= upper for lower, upper in axes):
        raise ValueError(f"Each {name} lower bound must be smaller than its upper bound")
    return axes


@dataclass(frozen=True)
class CartesianBoxDomain:
    """Axis-aligned finite support expressed in controller state coordinates."""

    lower: tuple[float, float]
    upper: tuple[float, float]
    margin: tuple[float, float] = (0.0, 0.0)
    support_mode: str = "cartesian_box"

    def __post_init__(self) -> None:
        lower = _pair(self.lower, name="lower")
        upper = _pair(self.upper, name="upper")
        margin = _pair(self.margin, name="margin")
        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("Every lower bound must be smaller than its upper bound")
        if any(value < 0.0 for value in margin):
            raise ValueError("margin values must be non-negative")
        if any(
            2.0 * pad >= hi - lo
            for lo, hi, pad in zip(lower, upper, margin, strict=True)
        ):
            raise ValueError("margin must be smaller than each box half-width")
        if self.support_mode != "cartesian_box":
            raise ValueError("support_mode must be 'cartesian_box'")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "margin", margin)

    def support_mask(
        self,
        coordinates: Array,
        *,
        use_margin: bool = False,
    ) -> Array:
        value = jnp.asarray(coordinates)
        if value.ndim < 1 or value.shape[-1] != 2:
            raise ValueError(
                f"Cartesian coordinates must end in dimension 2, got {value.shape}"
            )
        lower = jnp.asarray(self.lower, dtype=value.dtype)
        upper = jnp.asarray(self.upper, dtype=value.dtype)
        if use_margin:
            margin = jnp.asarray(self.margin, dtype=value.dtype)
            lower = lower + margin
            upper = upper - margin
        finite = jnp.all(jnp.isfinite(value), axis=-1)
        return finite & jnp.all((value >= lower) & (value <= upper), axis=-1)

    def metadata(self) -> dict[str, object]:
        return {
            "box": [
                [self.lower[0], self.upper[0]],
                [self.lower[1], self.upper[1]],
            ],
            "lower": list(self.lower),
            "upper": list(self.upper),
            "anchor": 0,
            "margin": list(self.margin),
            "support_mode": self.support_mode,
        }


@dataclass(frozen=True)
class MB2DGridReference:
    """A midpoint grid whose probability masses sum exactly to one."""

    state: Array
    physical: Array
    reduced_energy: Array
    probability_mass: Array
    log_normalizer_state: Array
    cell_area_state: float


@dataclass(frozen=True)
class AnalyticMB2DPotential:
    """Analytic Muller--Brown energy with an exact finite formal support.

    Temperature appears exactly once: ``reduced_energy = beta * energy`` and
    ``kT = 1 / beta``.  The optional confinement term is a training aid, not a
    part of the formal finite-box target.
    """

    beta: float = 1.0
    offset: tuple[float, float] = (25.0, 25.0)
    scale: tuple[float, float] = (10.0, 10.0)
    physical_box: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 50.0),
        (0.0, 50.0),
    )
    confinement_margin: tuple[float, float] = (2.5, 2.5)
    confinement_strength: float = 1.0
    extension_width: tuple[float, float] = (2.5, 2.5)

    def __post_init__(self) -> None:
        beta = float(self.beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be finite and positive")
        offset = _pair(self.offset, name="offset")
        scale = _pair(self.scale, name="scale", positive=True)
        physical_box = _box(self.physical_box, name="physical_box")
        margin = _pair(self.confinement_margin, name="confinement_margin")
        extension_width = _pair(
            self.extension_width,
            name="extension_width",
            positive=True,
        )
        strength = float(self.confinement_strength)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("confinement_strength must be finite and non-negative")
        for (lower, upper), pad in zip(physical_box, margin, strict=True):
            if pad < 0.0 or 2.0 * pad >= upper - lower:
                raise ValueError(
                    "confinement_margin must be non-negative and smaller "
                    "than each physical-box half-width"
                )
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "physical_box", physical_box)
        object.__setattr__(self, "confinement_margin", margin)
        object.__setattr__(self, "confinement_strength", strength)
        object.__setattr__(self, "extension_width", extension_width)

    @property
    def kT(self) -> float:
        return 1.0 / self.beta

    @property
    def physical_lower(self) -> tuple[float, float]:
        return tuple(axis[0] for axis in self.physical_box)

    @property
    def physical_upper(self) -> tuple[float, float]:
        return tuple(axis[1] for axis in self.physical_box)

    @property
    def state_domain(self) -> CartesianBoxDomain:
        lower = tuple(
            (bound - shift) / factor
            for bound, shift, factor in zip(
                self.physical_lower,
                self.offset,
                self.scale,
                strict=True,
            )
        )
        upper = tuple(
            (bound - shift) / factor
            for bound, shift, factor in zip(
                self.physical_upper,
                self.offset,
                self.scale,
                strict=True,
            )
        )
        margin = tuple(
            pad / factor
            for pad, factor in zip(
                self.confinement_margin,
                self.scale,
                strict=True,
            )
        )
        return CartesianBoxDomain(lower=lower, upper=upper, margin=margin)

    def state_to_physical(self, state: Array) -> Array:
        value = jnp.asarray(state)
        if value.ndim < 1 or value.shape[-1] != 2:
            raise ValueError(f"MB2D state must end in dimension 2, got {value.shape}")
        offset = jnp.asarray(self.offset, dtype=value.dtype)
        scale = jnp.asarray(self.scale, dtype=value.dtype)
        return offset + scale * value

    def physical_to_state(self, physical: Array) -> Array:
        value = jnp.asarray(physical)
        if value.ndim < 1 or value.shape[-1] != 2:
            raise ValueError(
                f"MB2D physical coordinates must end in dimension 2, got {value.shape}"
            )
        offset = jnp.asarray(self.offset, dtype=value.dtype)
        scale = jnp.asarray(self.scale, dtype=value.dtype)
        return (value - offset) / scale

    def formal_support_mask(self, state: Array) -> Array:
        return self.state_domain.support_mask(state)

    def _safe_physical(self, physical: Array) -> Array:
        """C2 identity-on-box extension that saturates outside the box."""

        lower = jnp.asarray(self.physical_lower, dtype=physical.dtype)
        upper = jnp.asarray(self.physical_upper, dtype=physical.dtype)
        width = jnp.asarray(self.extension_width, dtype=physical.dtype)
        below = lower + width * jnp.tanh((physical - lower) / width)
        above = upper + width * jnp.tanh((physical - upper) / width)
        return jnp.where(
            physical < lower,
            below,
            jnp.where(physical > upper, above, physical),
        )

    def _energy_components(self, state: Array) -> tuple[Array, Array]:
        physical = self.state_to_physical(state)
        safe_physical = self._safe_physical(physical)
        mb_energy = muller_brown_energy(safe_physical)

        lower = jnp.asarray(self.physical_lower, dtype=physical.dtype)
        upper = jnp.asarray(self.physical_upper, dtype=physical.dtype)
        margin = jnp.asarray(self.confinement_margin, dtype=physical.dtype)
        lower_start = lower + margin
        upper_start = upper - margin
        lower_excess = jax.nn.relu(lower_start - physical)
        upper_excess = jax.nn.relu(physical - upper_start)
        wall = 0.5 * jnp.asarray(
            self.confinement_strength,
            dtype=physical.dtype,
        ) * jnp.sum(lower_excess**2 + upper_excess**2, axis=-1)
        return mb_energy, wall

    def energy(
        self,
        coordinates: Array,
        *,
        include_training_wall: bool = True,
    ) -> Array:
        mb_energy, wall = self._energy_components(jnp.asarray(coordinates))
        return mb_energy + wall if include_training_wall else mb_energy

    def energy_and_grad(
        self,
        coordinates: Array,
        *,
        include_training_wall: bool = True,
    ) -> tuple[Array, Array]:
        value = jnp.asarray(coordinates)
        if value.ndim < 1 or value.shape[-1] != 2:
            raise ValueError(f"MB2D state must end in dimension 2, got {value.shape}")
        single_value_grad = jax.value_and_grad(
            lambda state: self.energy(
                state,
                include_training_wall=include_training_wall,
            ).reshape(())
        )
        if value.ndim == 1:
            return single_value_grad(value)
        flat = value.reshape((-1, 2))
        energy, gradient = jax.vmap(single_value_grad)(flat)
        return energy.reshape(value.shape[:-1]), gradient.reshape(value.shape)

    def evaluate(
        self,
        coordinates: Array,
        *,
        include_training_wall: bool = True,
    ) -> PotentialResult:
        value = jnp.asarray(coordinates)
        energy, gradient = self.energy_and_grad(
            value,
            include_training_wall=include_training_wall,
        )
        mb_energy, wall = self._energy_components(value)
        selected_wall = wall if include_training_wall else jnp.zeros_like(wall)
        beta = jnp.asarray(self.beta, dtype=value.dtype)
        reduced_energy = beta * energy
        reduced_gradient = beta * gradient
        finite_gradient = jnp.all(jnp.isfinite(gradient), axis=-1)
        valid = (
            self.formal_support_mask(value)
            & jnp.isfinite(energy)
            & finite_gradient
        )
        return PotentialResult(
            energy=energy,
            gradient=gradient,
            reduced_energy=reduced_energy,
            reduced_gradient=reduced_gradient,
            score=-reduced_gradient,
            valid_mask=valid,
            components={
                "U_muller_brown": mb_energy,
                "U_confinement": selected_wall,
                "U_target": energy,
            },
        )

    def normalized_grid(
        self,
        resolution: int | tuple[int, int] = (256, 256),
        *,
        dtype: Any = jnp.float32,
    ) -> MB2DGridReference:
        """Return an exactly normalized midpoint discretization of the target.

        The continuous normalizer is approximated by the midpoint rule, while
        the returned discrete ``probability_mass`` is normalized by log-sum-exp
        and therefore sums to one up to floating-point roundoff.  Integration
        is in controller state coordinates, matching PF-ODE ``logq``.
        """

        if isinstance(resolution, int):
            shape = (resolution, resolution)
        else:
            shape = tuple(int(value) for value in resolution)
        if len(shape) != 2 or any(value < 2 for value in shape):
            raise ValueError("resolution must contain two integers >= 2")
        domain = self.state_domain
        lower = jnp.asarray(domain.lower, dtype=dtype)
        upper = jnp.asarray(domain.upper, dtype=dtype)
        counts = jnp.asarray(shape, dtype=dtype)
        widths = (upper - lower) / counts
        x_axis = lower[0] + (jnp.arange(shape[0], dtype=dtype) + 0.5) * widths[0]
        y_axis = lower[1] + (jnp.arange(shape[1], dtype=dtype) + 0.5) * widths[1]
        xx, yy = jnp.meshgrid(x_axis, y_axis, indexing="ij")
        state = jnp.stack((xx, yy), axis=-1)
        reduced_energy = jnp.asarray(self.beta, dtype=dtype) * self.energy(
            state,
            include_training_wall=False,
        )
        cell_area = widths[0] * widths[1]
        log_mass_unnormalized = -reduced_energy + jnp.log(cell_area)
        log_normalizer = logsumexp(log_mass_unnormalized)
        probability_mass = jnp.exp(log_mass_unnormalized - log_normalizer)
        return MB2DGridReference(
            state=state,
            physical=self.state_to_physical(state),
            reduced_energy=reduced_energy,
            probability_mass=probability_mass,
            log_normalizer_state=log_normalizer,
            cell_area_state=float(cell_area),
        )


__all__ = [
    "AnalyticMB2DPotential",
    "CartesianBoxDomain",
    "MB2DGridReference",
]
