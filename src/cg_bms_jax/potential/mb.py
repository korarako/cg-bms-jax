"""Muller--Brown targets and the CG-BG one-dimensional RBF PMF."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .base import PotentialResult
from .bundle import MBPMFBundle, load_trusted_pickle

Array = jax.Array


def muller_brown_energy(xy: Array, *, biased: bool = False) -> Array:
    """Analytic two-dimensional potential used by the CG-BG MB experiment."""

    xy = jnp.asarray(xy)
    if xy.shape[-1] != 2:
        raise ValueError(f"Expected (...,2) Muller--Brown coordinates, got {xy.shape}")
    x, y = xy[..., 0], xy[..., 1]
    energy = (
        -17.3 * jnp.exp(-0.0039 * (x - 48.0) ** 2 - 0.0391 * (y - 8.0) ** 2)
        - 8.7 * jnp.exp(-0.0039 * (x - 32.0) ** 2 - 0.0391 * (y - 16.0) ** 2)
        - 14.7
        * jnp.exp(
            -0.0254 * (x - 24.0) ** 2
            + 0.043 * (x - 24.0) * (y - 32.0)
            - 0.0254 * (y - 32.0) ** 2
        )
        + 1.3
        * jnp.exp(
            0.00273 * (x - 16.0) ** 2
            + 0.0023 * (x - 16.0) * (y - 24.0)
            + 0.00273 * (y - 24.0) ** 2
        )
    )
    if biased:
        energy = energy - 4.0 * jnp.exp(-((x - 32.0) ** 2) / (2.0 * 5.0**2))
    return energy


def _natural_key(name: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def _unwrap_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    if "params" in params and isinstance(params["params"], Mapping):
        return params["params"]
    return params


def rbf_mlp_apply(params: Mapping[str, Any], x: Array, *, sigma: float = 5.0) -> Array:
    """Apply the exact Flax parameter tree used by ``cg_bg.models.RBFMLP``.

    This manual apply function keeps runtime independent of CG-BG's trainer
    module (which imports a Torch DataLoader at module import time).
    """

    root = _unwrap_params(params)
    rbf_names = sorted((name for name in root if name.lower().startswith("rbf")), key=_natural_key)
    if len(rbf_names) != 1:
        raise KeyError(f"Expected one RBF parameter block, found {rbf_names}")
    centers = jnp.asarray(root[rbf_names[0]]["centers"])

    x_arr = jnp.asarray(x)
    scalar = x_arr.ndim == 0
    trailing_singleton = x_arr.ndim > 0 and x_arr.shape[-1] == 1
    if scalar:
        leading = ()
        flat = x_arr.reshape(1, 1)
    elif trailing_singleton:
        leading = x_arr.shape[:-1]
        flat = x_arr.reshape(-1, 1)
    else:
        leading = x_arr.shape
        flat = x_arr.reshape(-1, 1)
    diff = flat[:, None, :] - centers[None, :, :]
    hidden = jnp.exp(-jnp.sum(diff * diff, axis=-1) / (2.0 * sigma**2))

    dense_names = sorted((name for name in root if name.lower().startswith("dense")), key=_natural_key)
    if not dense_names:
        raise KeyError("No Dense parameter blocks found in CG-BG RBFMLP checkpoint")
    for index, name in enumerate(dense_names):
        block = root[name]
        hidden = hidden @ jnp.asarray(block["kernel"])
        if "bias" in block:
            hidden = hidden + jnp.asarray(block["bias"])
        if index != len(dense_names) - 1:
            hidden = jax.nn.softplus(hidden)
    output = jnp.squeeze(hidden, axis=-1).reshape(leading)
    return output if not scalar else output.reshape(())


@dataclass(frozen=True)
class CGBGMBPotential:
    params: Mapping[str, Any]
    kT: float = 1.0
    sigma: float = 5.0

    @classmethod
    def from_bundle(cls, bundle: MBPMFBundle, *, trusted: bool = False) -> CGBGMBPotential:
        params = load_trusted_pickle(
            bundle.checkpoint_path,
            expected_sha256=bundle.checkpoint_sha256,
            trusted=trusted,
        )
        return cls(params=params, kT=bundle.kT, sigma=float(bundle.model_config.get("sigma", 5.0)))

    def energy(self, coordinates: Array) -> Array:
        return rbf_mlp_apply(self.params, coordinates, sigma=self.sigma)

    def energy_and_grad(self, coordinates: Array) -> tuple[Array, Array]:
        x = jnp.asarray(coordinates)
        return jax.value_and_grad(lambda value: jnp.sum(self.energy(value)))(x)

    def evaluate(self, coordinates: Array) -> PotentialResult:
        energy, gradient = self.energy_and_grad(coordinates)
        # value_and_grad(sum) returns the per-coordinate derivative while the
        # separately evaluated energy retains one value per sample.
        energy = self.energy(coordinates)
        reduced_energy = energy / self.kT
        reduced_gradient = gradient / self.kT
        valid = jnp.isfinite(energy)
        gradient_valid = jnp.isfinite(gradient)
        if gradient.ndim > energy.ndim:
            gradient_valid = jnp.all(
                gradient_valid,
                axis=tuple(range(energy.ndim, gradient.ndim)),
            )
        valid = valid & gradient_valid
        return PotentialResult(
            energy=energy,
            gradient=gradient,
            reduced_energy=reduced_energy,
            reduced_gradient=reduced_gradient,
            score=-reduced_gradient,
            valid_mask=valid,
        )
