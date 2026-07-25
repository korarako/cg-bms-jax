"""Controller wrappers for scalar and ambient molecular BMS states."""

from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from cg_bms_jax.model.painn import TimeEmbedding, TorchLinear, _as_batch_time

Array = jax.Array


class MLPController(nn.Module):
    """Time-conditioned MLP controller for low-dimensional CG experiments.

    The output is reshaped to the exact input event shape, so both ``(B,)`` and
    ``(B, D)`` states are supported.  Like PaiNN, it returns the score-like BMS
    control ``u``; multiplying by ``g(t)^2`` is the SDE's responsibility.
    """

    hidden_features: tuple[int, ...] = (128, 128, 128)
    time_embedding_dim: int = 32
    dtype: Any | None = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, time: Array | float, state: Array) -> Array:
        state = jnp.asarray(state)
        if state.ndim < 1:
            raise ValueError("state must include a batch dimension.")
        if not jnp.issubdtype(state.dtype, jnp.floating):
            raise TypeError("state must have a floating dtype.")
        batch_size = int(state.shape[0])
        event_shape = tuple(state.shape[1:])
        event_size = math.prod(event_shape) if event_shape else 1
        flat_state = jnp.reshape(state, (batch_size, event_size))
        time_batch = _as_batch_time(time, batch_size, state.dtype)
        if int(self.time_embedding_dim) > 0:
            time_features = TimeEmbedding(
                int(self.time_embedding_dim),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="time_embedding",
            )(time_batch)
        else:
            time_features = time_batch[:, None]
        hidden = jnp.concatenate((flat_state, time_features), axis=-1)
        for index, width in enumerate(self.hidden_features):
            if int(width) <= 0:
                raise ValueError("MLP hidden widths must be positive.")
            hidden = TorchLinear(
                int(width),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"hidden_{index}",
            )(hidden)
            hidden = jax.nn.silu(hidden)
        output = TorchLinear(
            event_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="output",
        )(hidden)
        return jnp.reshape(output, state.shape)


class RadialCOMHead(nn.Module):
    """Rotation-equivariant control for an isotropic three-dimensional COM.

    The learned scalar coefficient only sees the invariant squared radius and
    time; multiplying it by its three-dimensional input makes the output
    transform as a proper vector.  :class:`ShapeCOMController` can supply
    either the raw geometric center ``c`` (legacy mode) or the orthonormal
    coordinate ``z=sqrt(N)c``.  In orthogonal mode the standard-normal terminal
    COM score is simply ``-z``.
    """

    hidden_features: tuple[int, ...] = (64, 64)
    time_embedding_dim: int = 32
    dtype: Any | None = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, time: Array | float, center: Array) -> Array:
        center = jnp.asarray(center)
        if center.ndim != 2 or center.shape[-1] != 3:
            raise ValueError(
                f"center must have shape (batch, 3), got {center.shape}."
            )
        if not jnp.issubdtype(center.dtype, jnp.floating):
            raise TypeError("center must have a floating dtype.")
        batch_size = int(center.shape[0])
        time_batch = _as_batch_time(time, batch_size, center.dtype)
        radius_squared = jnp.sum(center**2, axis=-1, keepdims=True)
        if int(self.time_embedding_dim) > 0:
            time_features = TimeEmbedding(
                int(self.time_embedding_dim),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="time_embedding",
            )(time_batch)
        else:
            time_features = time_batch[:, None]
        hidden = jnp.concatenate((radius_squared, time_features), axis=-1)
        for index, width in enumerate(self.hidden_features):
            if int(width) <= 0:
                raise ValueError("COM-head hidden widths must be positive.")
            hidden = TorchLinear(
                int(width),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"hidden_{index}",
            )(hidden)
            hidden = jax.nn.silu(hidden)
        coefficient = TorchLinear(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="output",
        )(hidden)
        return coefficient * center


class ShapeCOMController(nn.Module):
    """Orthogonally combine a mean-free shape control and ambient COM control.

    ``shape_model`` receives centered coordinates.  Its raw output is projected
    into the zero-sum particle subspace.  In legacy ``"center"`` mode,
    ``com_head`` receives the geometric center and returns one per-particle
    vector.  In ``"orthogonal"`` mode it receives ``z=sqrt(N) * center`` and
    returns a vector in that orthonormal three-dimensional coordinate; the
    wrapper maps it back by dividing by ``sqrt(N)`` before broadcasting.

    Crucially, the final result is *not* centered again.  For all-atom Ala2 with
    22 atoms this preserves the full-rank ``66 = 63 + 3`` state while the PaiNN
    shape branch remains rigorously centered.
    """

    shape_model: nn.Module
    com_head: nn.Module
    clip_value: float | None = None
    position_scale: float = 1.0
    com_coordinates: str = "center"
    num_particles: int | None = None

    @nn.compact
    def __call__(
        self,
        time: Array | float,
        positions: Array,
        *,
        return_components: bool = False,
    ) -> Array | tuple[Array, dict[str, Array]]:
        positions = jnp.asarray(positions)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(
                f"positions must have shape (batch, atoms, 3), got {positions.shape}."
            )
        n_particles = int(positions.shape[1])
        if self.num_particles is not None and n_particles != int(self.num_particles):
            raise ValueError(
                f"Expected {self.num_particles} particles, got {n_particles}."
            )
        if self.com_coordinates not in {"center", "orthogonal"}:
            raise ValueError(
                "com_coordinates must be either 'center' or 'orthogonal'"
            )
        center = jnp.mean(positions, axis=1)
        sqrt_n = jnp.sqrt(jnp.asarray(n_particles, dtype=positions.dtype))
        orthogonal_com = sqrt_n * center
        centered = positions - center[:, None, :]
        if not math.isfinite(float(self.position_scale)) or float(self.position_scale) <= 0.0:
            raise ValueError("position_scale must be finite and positive.")
        # The ambient state is commonly standardized by CG-BG.  Pair geometry
        # and the PaiNN cutoff, however, are defined in nm.  Only the geometry
        # supplied to the shape backbone is rescaled; the COM head and returned
        # stochastic control remain in standardized state coordinates.
        shape_positions = centered * jnp.asarray(
            self.position_scale, dtype=positions.dtype
        )
        raw_shape = self.shape_model(time, shape_positions)
        if raw_shape.shape != positions.shape:
            raise ValueError(
                "shape_model must return the same shape as positions; "
                f"got {raw_shape.shape} and {positions.shape}."
            )
        if self.clip_value is not None:
            raw_shape = jnp.clip(raw_shape, -self.clip_value, self.clip_value)
        shape_control = raw_shape - jnp.mean(raw_shape, axis=1, keepdims=True)

        com_input = center if self.com_coordinates == "center" else orthogonal_com
        raw_com_control = self.com_head(time, com_input)
        if raw_com_control.shape != center.shape:
            raise ValueError(
                "com_head must return shape (batch, 3); "
                f"got {raw_com_control.shape} and {center.shape}."
            )
        if self.com_coordinates == "center":
            com_control = raw_com_control
            orthogonal_com_control = sqrt_n * raw_com_control
        else:
            orthogonal_com_control = raw_com_control
            com_control = raw_com_control / sqrt_n
        control = shape_control + com_control[:, None, :]
        if return_components:
            return control, {
                "center": center,
                "orthogonal_com": orthogonal_com,
                "centered": centered,
                "shape_positions": shape_positions,
                "shape": shape_control,
                "com": com_control,
                "orthogonal_com_control": orthogonal_com_control,
            }
        return control
