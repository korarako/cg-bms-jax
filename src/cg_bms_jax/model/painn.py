"""Faithful Flax port of the BridgeMatchingSampler PaiNN controller.

The tensor layout and operation order intentionally follow the upstream Torch
implementation rather than another JAX PaiNN implementation.  In particular:

* scalar channels have shape ``(B, N, 1, F)``;
* vector channels have shape ``(B, N, 3, F)``;
* dense all-pairs messages use receiver ``i`` and sender ``j``;
* the parity-breaking message is ``v_j cross r_hat_ij``; and
* every interaction is followed by RMS norm, update, then a second RMS norm.

The module returns the raw per-bead vector field.  Clipping and translation
subspace handling belong to the controller wrapper, just as in upstream BMS.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array


def _symmetric_uniform(scale: float):
    """Return a PyTorch-style symmetric uniform initializer."""

    scale = float(scale)

    def init(key: Array, shape: tuple[int, ...], dtype: Any = jnp.float32) -> Array:
        return jax.random.uniform(
            key,
            shape,
            dtype=dtype,
            minval=-scale,
            maxval=scale,
        )

    return init


def _as_batch_time(time: Array | float, batch_size: int, dtype: Any) -> Array:
    """Normalize scalar or broadcastable time input to ``(B,)``."""

    time_array = jnp.asarray(time, dtype=dtype)
    if time_array.ndim == 0:
        return jnp.full((batch_size,), time_array, dtype=dtype)
    time_array = jnp.reshape(time_array, (-1,))
    if time_array.size == 1:
        return jnp.broadcast_to(time_array, (batch_size,))
    if time_array.shape != (batch_size,):
        raise ValueError(
            f"time must be scalar or contain one value per batch item; got "
            f"{time_array.shape} for batch size {batch_size}."
        )
    return time_array


class TorchLinear(nn.Module):
    """Dense layer with the exact default ``torch.nn.Linear`` distribution.

    Torch stores weights as ``(out, in)`` whereas this module stores kernels as
    ``(in, out)``.  A Torch golden-state importer must therefore transpose every
    linear weight; biases can be copied without modification.
    """

    features: int
    use_bias: bool = True
    dtype: Any | None = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        if inputs.ndim < 1:
            raise ValueError("TorchLinear input must have a feature axis.")
        fan_in = int(inputs.shape[-1])
        if fan_in <= 0 or int(self.features) <= 0:
            raise ValueError("TorchLinear requires positive input and output widths.")
        bound = 1.0 / math.sqrt(fan_in)
        kernel = self.param(
            "kernel",
            _symmetric_uniform(bound),
            (fan_in, int(self.features)),
            self.param_dtype,
        )
        compute_dtype = self.dtype or jnp.result_type(inputs.dtype, kernel.dtype)
        output = jnp.einsum(
            "...i,io->...o",
            jnp.asarray(inputs, dtype=compute_dtype),
            jnp.asarray(kernel, dtype=compute_dtype),
        )
        if self.use_bias:
            bias = self.param(
                "bias",
                _symmetric_uniform(bound),
                (int(self.features),),
                self.param_dtype,
            )
            output = output + jnp.asarray(bias, dtype=compute_dtype)
        return output


class AtomEmbedding(nn.Module):
    """BMS atom/slot embedding initialized uniformly on ``[-sqrt(3),sqrt(3)]``."""

    num_embeddings: int
    num_features: int
    dtype: Any | None = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, indices: Array) -> Array:
        if int(self.num_embeddings) <= 0 or int(self.num_features) <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        table = self.param(
            "embedding",
            _symmetric_uniform(math.sqrt(3.0)),
            (int(self.num_embeddings), int(self.num_features)),
            self.param_dtype,
        )
        compute_dtype = self.dtype or table.dtype
        return jnp.asarray(table, dtype=compute_dtype)[jnp.asarray(indices, dtype=jnp.int32)]


class TimeEmbedding(nn.Module):
    """Random Fourier time features used by upstream BMS.

    Frequencies are checkpointed in the non-trainable ``constants`` collection.
    Call ``init`` with a ``constants`` RNG (or rely on Flax's params fallback)
    and retain both the ``params`` and ``constants`` variable collections.
    """

    num_basis: int
    dtype: Any | None = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, values: Array) -> Array:
        if int(self.num_basis) <= 0 or int(self.num_basis) % 2:
            raise ValueError("TimeEmbedding num_basis must be a positive even integer.")

        def initialize_frequencies() -> Array:
            key = self.make_rng("constants")
            frequencies = jax.random.normal(
                key,
                (int(self.num_basis) // 2,),
                dtype=self.param_dtype,
            )
            return frequencies * jnp.asarray(2.0 * math.pi, dtype=self.param_dtype)

        frequencies = self.variable(
            "constants", "freqs", initialize_frequencies
        ).value
        values = jnp.asarray(values)
        compute_dtype = self.dtype or jnp.result_type(values.dtype, frequencies.dtype)
        arguments = (
            jnp.asarray(values, dtype=compute_dtype)[..., None]
            * jnp.asarray(frequencies, dtype=compute_dtype)
        )
        embedding = jnp.concatenate(
            (jnp.sin(arguments), jnp.cos(arguments)), axis=-1
        )
        return embedding * jnp.asarray(math.sqrt(2.0), dtype=compute_dtype)


class PolynomialEnvelope(nn.Module):
    """Polynomial cutoff envelope from the upstream radial-basis module."""

    exponent: int = 5

    @nn.compact
    def __call__(self, scaled_distance: Array) -> Array:
        if int(self.exponent) <= 0:
            raise ValueError("Polynomial envelope exponent must be positive.")
        p = float(self.exponent)
        a = -(p + 1.0) * (p + 2.0) / 2.0
        b = p * (p + 2.0)
        c = -p * (p + 1.0) / 2.0
        x = jnp.asarray(scaled_distance)
        value = 1.0 + a * x**p + b * x ** (p + 1.0) + c * x ** (p + 2.0)
        return jnp.where(x < 1.0, value, jnp.zeros_like(value))


class GaussianSmearing(nn.Module):
    """Fixed Gaussian RBF centers matching ``GaussianSmearing`` in BMS."""

    start: float = 0.0
    stop: float = 1.0
    num_gaussians: int = 50

    @nn.compact
    def __call__(self, distance: Array) -> Array:
        if int(self.num_gaussians) < 2:
            raise ValueError("GaussianSmearing requires at least two centers.")
        distance = jnp.asarray(distance)
        offsets = jnp.linspace(
            self.start,
            self.stop,
            int(self.num_gaussians),
            dtype=distance.dtype,
        )
        spacing = offsets[1] - offsets[0]
        coefficient = -0.5 / spacing**2
        delta = distance[..., None] - offsets
        return jnp.exp(coefficient * delta**2)


class RadialBasis(nn.Module):
    """Gaussian radial basis and fifth-order polynomial envelope."""

    num_radial: int
    cutoff: float
    envelope_exponent: int = 5

    @nn.compact
    def __call__(self, distance: Array) -> tuple[Array, Array]:
        if float(self.cutoff) <= 0.0:
            raise ValueError("Radial cutoff must be positive.")
        scaled = jnp.asarray(distance) / jnp.asarray(
            self.cutoff, dtype=jnp.asarray(distance).dtype
        )
        radial = GaussianSmearing(
            start=0.0,
            stop=1.0,
            num_gaussians=int(self.num_radial),
            name="rbf",
        )(scaled)
        envelope = PolynomialEnvelope(
            exponent=int(self.envelope_exponent), name="envelope"
        )(scaled)
        return radial, envelope


class _TwoLayerMLP(nn.Module):
    hidden_features: int
    output_features: int
    dtype: Any | None = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        hidden = TorchLinear(
            int(self.hidden_features),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="linear_0",
        )(inputs)
        hidden = jax.nn.silu(hidden)
        return TorchLinear(
            int(self.output_features),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="linear_1",
        )(hidden)


def _cross_feature_vectors(left: Array, right: Array) -> Array:
    """Cross vectors whose Cartesian axis is penultimate.

    The explicit formula avoids any ambiguity about the cross-product axis or
    operand ordering.  Feature dimensions of size one are broadcast normally.
    """

    c0 = left[..., 1, :] * right[..., 2, :] - left[..., 2, :] * right[..., 1, :]
    c1 = left[..., 2, :] * right[..., 0, :] - left[..., 0, :] * right[..., 2, :]
    c2 = left[..., 0, :] * right[..., 1, :] - left[..., 1, :] * right[..., 0, :]
    return jnp.stack((c0, c1, c2), axis=-2)


class ParityBreakingMessageBlock(nn.Module):
    num_features: int
    num_radial_basis: int
    dtype: Any | None = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        scalar: Array,
        vector: Array,
        radial_embeddings: Array,
        envelope: Array,
        unit_vectors: Array,
    ) -> tuple[Array, Array]:
        features = int(self.num_features)
        phi = _TwoLayerMLP(
            hidden_features=features,
            output_features=4 * features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mlp_phi",
        )(scalar)
        filters = TorchLinear(
            4 * features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="linear_W",
        )(radial_embeddings)
        filters = filters * envelope[..., None]
        messages = phi[:, None, ...] * filters
        x_s, x_vv, x_vs, x_vc = jnp.split(messages, 4, axis=-1)

        delta_scalar = jnp.sum(x_s, axis=2)
        sender_vector = vector[:, None, ...]
        directions = unit_vectors[..., None]
        # The ordering below is exactly torch.cross(v_j, r_hat_ij, dim=-2).
        chiral = _cross_feature_vectors(sender_vector, directions)
        vector_messages = (
            sender_vector * x_vv
            + directions * x_vs
            + chiral * x_vc
        )
        delta_vector = jnp.sum(vector_messages, axis=2)
        return scalar + delta_scalar, vector + delta_vector


class MessageBlock(nn.Module):
    """Reflection-equivariant BMS message block without the chiral channel."""

    num_features: int
    num_radial_basis: int
    dtype: Any | None = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        scalar: Array,
        vector: Array,
        radial_embeddings: Array,
        envelope: Array,
        unit_vectors: Array,
    ) -> tuple[Array, Array]:
        features = int(self.num_features)
        phi = _TwoLayerMLP(
            hidden_features=features,
            output_features=3 * features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mlp_phi",
        )(scalar)
        filters = TorchLinear(
            3 * features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="linear_W",
        )(radial_embeddings)
        filters = filters * envelope[..., None]
        messages = phi[:, None, ...] * filters
        x_s, x_vv, x_vs = jnp.split(messages, 3, axis=-1)
        delta_scalar = jnp.sum(x_s, axis=2)
        vector_messages = (
            vector[:, None, ...] * x_vv
            + unit_vectors[..., None] * x_vs
        )
        delta_vector = jnp.sum(vector_messages, axis=2)
        return scalar + delta_scalar, vector + delta_vector


class UpdateBlock(nn.Module):
    num_features: int
    dtype: Any | None = None
    param_dtype: Any = jnp.float32
    eps: float = 1.0e-6

    @nn.compact
    def __call__(self, scalar: Array, vector: Array) -> tuple[Array, Array]:
        features = int(self.num_features)
        mixed = TorchLinear(
            2 * features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="linear_UV",
        )(vector)
        u_vector, v_vector = jnp.split(mixed, 2, axis=-1)
        v_norm = jnp.sqrt(
            jnp.sum(v_vector**2, axis=-2, keepdims=True) + self.eps
        )
        coefficients = _TwoLayerMLP(
            hidden_features=features,
            output_features=3 * features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mlp_a",
        )(jnp.concatenate((scalar, v_norm), axis=-1))
        a_vv, a_sv, a_ss = jnp.split(coefficients, 3, axis=-1)
        delta_vector = a_vv * u_vector
        uv_dot = jnp.sum(u_vector * v_vector, axis=-2, keepdims=True)
        delta_scalar = a_ss + a_sv * uv_dot
        return scalar + delta_scalar, vector + delta_vector


class EquivariantRMSNorm(nn.Module):
    num_features: int
    affine: bool = True
    eps: float = 1.0e-6
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, scalar: Array, vector: Array) -> tuple[Array, Array]:
        scalar = scalar - jnp.mean(scalar, axis=-1, keepdims=True)
        scalar = scalar / jnp.sqrt(
            jnp.mean(scalar**2, axis=-1, keepdims=True) + self.eps
        )
        # Upstream averages over Cartesian and feature axes jointly.
        vector = vector / jnp.sqrt(
            jnp.mean(vector**2, axis=(-2, -1), keepdims=True) + self.eps
        )
        if self.affine:
            features = int(self.num_features)
            scalar_weight = self.param(
                "affine_s_weight", nn.initializers.ones, (features,), self.param_dtype
            )
            scalar_bias = self.param(
                "affine_s_bias", nn.initializers.zeros, (features,), self.param_dtype
            )
            vector_weight = self.param(
                "affine_v_weight", nn.initializers.ones, (features,), self.param_dtype
            )
            scalar = scalar * scalar_weight.astype(scalar.dtype) + scalar_bias.astype(
                scalar.dtype
            )
            vector = vector * vector_weight.astype(vector.dtype)
        return scalar, vector


class GatedEquivariantBlock(nn.Module):
    num_features: int
    num_output_features: int
    init_scale: float | None = None
    dtype: Any | None = None
    param_dtype: Any = jnp.float32
    eps: float = 1.0e-6

    @nn.compact
    def __call__(self, scalar: Array, vector: Array) -> tuple[Array, Array]:
        # ``init_scale`` is intentionally unused: it is stored but never applied
        # in the pinned upstream implementation.
        _ = self.init_scale
        features = int(self.num_features)
        output_features = int(self.num_output_features)
        vector_1 = TorchLinear(
            features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="linear_v1",
        )(vector)
        vector_1_norm = jnp.sqrt(
            jnp.sum(vector_1**2, axis=-2, keepdims=True) + self.eps
        )
        vector_2 = TorchLinear(
            output_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="linear_v2",
        )(vector)
        scalar_and_gate = _TwoLayerMLP(
            hidden_features=features,
            output_features=2 * output_features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mlp_s",
        )(jnp.concatenate((scalar, vector_1_norm), axis=-1))
        scalar_out, vector_scale = jnp.split(scalar_and_gate, 2, axis=-1)
        return jax.nn.silu(scalar_out), vector_2 * vector_scale


class VectorHead(nn.Module):
    num_features: int
    init_scale: float | None = None
    dtype: Any | None = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, scalar: Array, vector: Array) -> Array:
        features = int(self.num_features)
        if features < 2:
            raise ValueError("VectorHead requires at least two features.")
        hidden = features // 2
        scalar, vector = GatedEquivariantBlock(
            features,
            hidden,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="block_0",
        )(scalar, vector)
        _, vector = GatedEquivariantBlock(
            hidden,
            1,
            init_scale=self.init_scale,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="block_1",
        )(scalar, vector)
        return jnp.squeeze(vector, axis=-1)


class PaiNN(nn.Module):
    """Dense, time-conditioned, chiral BMS PaiNN vector controller."""

    num_features: int
    num_radial_basis: int
    num_layers: int
    num_elements: int
    r_max: float = 6.0
    r_offset: float = 0.0
    time_init_mode: Literal["node", "edge", "none"] = "node"
    parity_breaking: bool = True
    conservative: bool = False
    unique_atom_indices: bool = False
    dtype: Any | None = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        time: Array | float,
        positions: Array,
        atomic_numbers: Array | None = None,
    ) -> Array:
        if self.conservative:
            raise NotImplementedError(
                "The minimal JAX BMS package implements the vector controller only."
            )
        if self.time_init_mode not in ("node", "edge", "none"):
            raise ValueError(f"Invalid time embedding mode: {self.time_init_mode!r}.")
        if int(self.num_features) <= 0 or int(self.num_radial_basis) < 2:
            raise ValueError("PaiNN feature and radial dimensions are invalid.")
        if int(self.num_layers) <= 0:
            raise ValueError("PaiNN requires at least one interaction layer.")
        if float(self.r_max) <= 0.0 or float(self.r_max + self.r_offset) <= 0.0:
            raise ValueError("PaiNN cutoff must be positive.")

        positions = jnp.asarray(positions)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(
                f"positions must have shape (batch, atoms, 3), got {positions.shape}."
            )
        if not jnp.issubdtype(positions.dtype, jnp.floating):
            raise TypeError("positions must have a floating dtype.")
        batch_size, num_nodes, _ = positions.shape
        time_batch = _as_batch_time(time, int(batch_size), positions.dtype)

        displacement = positions[:, :, None, :] - positions[:, None, :, :]
        distance = jnp.sqrt(jnp.sum(displacement**2, axis=-1, keepdims=True) + 1.0e-6)
        unit_vectors = displacement / distance

        cutoff = float(self.r_max + self.r_offset)
        shifted_distance = jnp.minimum(
            jnp.squeeze(distance, axis=-1) + float(self.r_offset), cutoff
        )
        radial, envelope = RadialBasis(
            num_radial=int(self.num_radial_basis),
            cutoff=cutoff,
            envelope_exponent=5,
            name="radial_embedding",
        )(shifted_distance)
        if self.time_init_mode == "edge":
            radial = radial + TimeEmbedding(
                int(self.num_radial_basis),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="time_embedding",
            )(time_batch[:, None, None])
        radial = radial[:, :, :, None, :]
        envelope = envelope[:, :, :, None]

        mask = 1.0 - jnp.eye(int(num_nodes), dtype=positions.dtype)
        mask = mask[None, :, :, None]
        unit_vectors = unit_vectors * mask
        envelope = envelope * mask

        if self.unique_atom_indices:
            atom_indices = jnp.broadcast_to(
                jnp.arange(int(num_nodes), dtype=jnp.int32)[None, :],
                (int(batch_size), int(num_nodes)),
            )
            if int(num_nodes) > int(self.num_elements):
                raise ValueError(
                    "num_elements must cover every unique atom/slot index."
                )
        else:
            if atomic_numbers is None:
                atom_indices = jnp.zeros(
                    (int(batch_size), int(num_nodes)), dtype=jnp.int32
                )
            else:
                atom_indices = jnp.asarray(atomic_numbers, dtype=jnp.int32)
                if atom_indices.ndim == 1:
                    atom_indices = jnp.broadcast_to(
                        atom_indices[None, :], (int(batch_size), int(num_nodes))
                    )
                if atom_indices.shape != (int(batch_size), int(num_nodes)):
                    raise ValueError(
                        "atomic_numbers must have shape (atoms,) or (batch, atoms)."
                    )

        scalar = AtomEmbedding(
            num_embeddings=int(self.num_elements),
            num_features=int(self.num_features),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="atom_embedding",
        )(atom_indices)
        if self.time_init_mode == "node":
            scalar = scalar + TimeEmbedding(
                int(self.num_features),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="time_embedding",
            )(time_batch[:, None])
        scalar = scalar[:, :, None, :]
        vector = jnp.zeros(
            (int(batch_size), int(num_nodes), 3, int(self.num_features)),
            dtype=scalar.dtype,
        )

        for layer_index in range(int(self.num_layers)):
            if self.parity_breaking:
                message_cls = ParityBreakingMessageBlock
            else:
                message_cls = MessageBlock
            scalar, vector = message_cls(
                int(self.num_features),
                int(self.num_radial_basis),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"message_{layer_index}",
            )(scalar, vector, radial, envelope, unit_vectors)
            scalar, vector = EquivariantRMSNorm(
                int(self.num_features),
                affine=True,
                eps=1.0e-6,
                param_dtype=self.param_dtype,
                name=f"norm_1_{layer_index}",
            )(scalar, vector)
            scalar, vector = UpdateBlock(
                int(self.num_features),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                eps=1.0e-6,
                name=f"update_{layer_index}",
            )(scalar, vector)
            scalar, vector = EquivariantRMSNorm(
                int(self.num_features),
                affine=True,
                eps=1.0e-6,
                param_dtype=self.param_dtype,
                name=f"norm_2_{layer_index}",
            )(scalar, vector)

        return VectorHead(
            int(self.num_features),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="output_block",
        )(scalar, vector)
