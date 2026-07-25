"""CG-BG's full-rank ambient coordinate convention.

For a molecule with ``N`` coarse-grained beads, CG-BG first removes the
geometric centre, divides all coordinates by one global standard deviation,
and then adds one shared three-dimensional Gaussian displacement to every
bead.  The shared displacement has standard deviation ``1 / sqrt(N)``.  This
lifts the centred ``3N-3`` dimensional distribution to a full-rank ``3N``
dimensional ambient distribution.

The functions in this module deliberately use the *geometric* centre.  This is
what the upstream CG-BG implementation calls COM; no bead masses enter that
implementation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class AmbientDensityComponents(NamedTuple):
    """Exact log-density terms in the orthogonal shape/COM decomposition.

    ``shape`` is a density on the ``3N-3`` dimensional translation-free
    subspace, while ``com`` is a Cartesian density on the remaining three
    orthonormal centre coordinates.  ``total`` is their sum and is therefore a
    density in the original full-rank ``3N`` dimensional ambient coordinates.
    """

    shape: Array
    com: Array
    total: Array


def _as_batched(x: Array) -> tuple[Array, bool]:
    x = jnp.asarray(x)
    if x.ndim == 2:
        return x[None, ...], True
    if x.ndim != 3:
        raise ValueError(f"Expected coordinates with shape (N,3) or (B,N,3), got {x.shape}")
    if x.shape[-1] != 3:
        raise ValueError(f"The final coordinate dimension must be 3, got {x.shape}")
    return x, False


def helmert_basis(n_particles: int, *, dtype: jnp.dtype = jnp.float32) -> Array:
    """Return a deterministic orthonormal basis of the zero-sum subspace.

    The returned matrix has shape ``(N, N-1)`` and satisfies

    ``H.T @ H = I`` and ``H.T @ ones(N) = 0``.

    Together with the normalized constant vector ``ones(N) / sqrt(N)``, its
    columns form an orthogonal matrix.  Applying this transform independently
    to x/y/z coordinates therefore has absolute Jacobian determinant one.
    """

    if int(n_particles) < 2:
        raise ValueError("n_particles must be at least two")
    n_particles = int(n_particles)
    rows = jnp.arange(n_particles)[:, None]
    columns = jnp.arange(1, n_particles)[None, :]
    columns_float = columns.astype(dtype)
    denominator = jnp.sqrt(columns_float * (columns_float + 1.0))
    return jnp.where(
        rows < columns,
        1.0 / denominator,
        jnp.where(rows == columns, -columns_float / denominator, 0.0),
    ).astype(dtype)


def split_orthogonal_shape_com(x: Array) -> tuple[Array, Array]:
    """Split full-rank ambient coordinates into ``(3N-3) + 3`` coordinates.

    The shape coordinate is ``H.T @ x`` and has shape ``(..., N-1, 3)``.
    The COM coordinate is ``sqrt(N) * mean(x)`` and has shape ``(..., 3)``.
    This is an orthogonal, unit-Jacobian change of coordinates.
    """

    xb, squeezed = _as_batched(x)
    n_particles = int(xb.shape[-2])
    basis = helmert_basis(n_particles, dtype=xb.dtype)
    shape = jnp.einsum("ni,bnd->bid", basis, xb)
    normalized_com = jnp.sqrt(jnp.asarray(n_particles, dtype=xb.dtype)) * jnp.mean(
        xb, axis=-2
    )
    if squeezed:
        return shape[0], normalized_com[0]
    return shape, normalized_com


def combine_orthogonal_shape_com(shape: Array, normalized_com: Array) -> Array:
    """Invert :func:`split_orthogonal_shape_com` exactly."""

    shape = jnp.asarray(shape)
    normalized_com = jnp.asarray(normalized_com, dtype=shape.dtype)
    squeezed = shape.ndim == 2
    if squeezed:
        shape = shape[None, ...]
        normalized_com = normalized_com[None, ...]
    if shape.ndim != 3 or shape.shape[-1] != 3:
        raise ValueError(
            "shape must have shape (N-1,3) or (B,N-1,3), "
            f"got {shape.shape}"
        )
    if normalized_com.ndim != 2 or normalized_com.shape != (shape.shape[0], 3):
        raise ValueError(
            "normalized_com must have shape (3,) or (B,3) matching shape; "
            f"got {normalized_com.shape}"
        )
    n_particles = int(shape.shape[-2]) + 1
    basis = helmert_basis(n_particles, dtype=shape.dtype)
    centered = jnp.einsum("ni,bid->bnd", basis, shape)
    shared_com = normalized_com / jnp.sqrt(
        jnp.asarray(n_particles, dtype=shape.dtype)
    )
    ambient = centered + shared_com[:, None, :]
    return ambient[0] if squeezed else ambient


def _standard_normal_log_prob(value: Array, *, event_ndim: int) -> Array:
    value = jnp.asarray(value)
    if event_ndim <= 0 or value.ndim < event_ndim:
        raise ValueError("event_ndim must select at least one existing dimension")
    axes = tuple(range(value.ndim - event_ndim, value.ndim))
    event_size = math.prod(value.shape[-event_ndim:])
    return -0.5 * (
        jnp.sum(jnp.square(value), axis=axes)
        + jnp.asarray(event_size * math.log(2.0 * math.pi), dtype=value.dtype)
    )


def standard_com_log_prob(x: Array) -> Array:
    """Log density of the standard-normal orthogonal three-dimensional COM."""

    _, normalized_com = split_orthogonal_shape_com(x)
    return _standard_normal_log_prob(normalized_com, event_ndim=1)


def exact_ambient_density(
    shape_log_prob: Array,
    x: Array,
) -> AmbientDensityComponents:
    """Combine a ``3N-3`` shape density with the standard 3D COM density.

    This is the exact full-rank ambient density, without a radial Jacobian or a
    singular centered-coordinate correction.  Its ``total`` field can be used
    anywhere the existing probability-flow code expects one log probability
    per batch item.
    """

    xb, squeezed = _as_batched(x)
    shape_term = jnp.asarray(shape_log_prob, dtype=xb.dtype)
    if squeezed:
        if shape_term.ndim != 0:
            raise ValueError("Unbatched coordinates require a scalar shape_log_prob")
    elif shape_term.shape != (xb.shape[0],):
        raise ValueError(
            "shape_log_prob must have one value per batch item; "
            f"got {shape_term.shape} for batch {xb.shape[0]}"
        )
    com_term = standard_com_log_prob(x)
    return AmbientDensityComponents(
        shape=shape_term,
        com=com_term,
        total=shape_term + com_term,
    )


def standard_normal_ambient_density(x: Array) -> AmbientDensityComponents:
    """Decompose ``N(0, I_(3N))`` exactly into shape and COM terms."""

    shape, _ = split_orthogonal_shape_com(x)
    shape_term = _standard_normal_log_prob(shape, event_ndim=2)
    return exact_ambient_density(shape_term, x)


def split_shape_com(x: Array) -> tuple[Array, Array]:
    """Return centred shape coordinates and the shared geometric centre."""

    xb, squeezed = _as_batched(x)
    centre = jnp.mean(xb, axis=-2, keepdims=True)
    shape = xb - centre
    if squeezed:
        return shape[0], centre[0]
    return shape, centre


def compute_centered_std(physical_coordinates: Array, *, eps: float = 0.0) -> Array:
    """Reproduce CG-BG's scalar standard deviation after centre removal."""

    shape, _ = split_shape_com(physical_coordinates)
    std = jnp.std(shape)
    if eps > 0:
        std = jnp.maximum(std, jnp.asarray(eps, dtype=std.dtype))
    return std


def com_aux_reduced_energy(x: Array, com_std: float | Array | None = None) -> Array:
    """Reduced auxiliary COM energy for the ambient target.

    ``com_std=None`` selects CG-BG's value ``1/sqrt(N)``.  The returned array
    has one scalar per batch item.  Constants independent of ``x`` are omitted.
    """

    xb, squeezed = _as_batched(x)
    n_beads = xb.shape[-2]
    centre = jnp.mean(xb, axis=-2)
    sigma = jnp.asarray(
        1.0 / jnp.sqrt(jnp.asarray(n_beads, dtype=xb.dtype)) if com_std is None else com_std,
        dtype=xb.dtype,
    )
    if sigma.ndim != 0:
        raise ValueError("com_std must be a scalar")
    energy = 0.5 * jnp.sum(centre * centre, axis=-1) / (sigma * sigma)
    return energy[0] if squeezed else energy


def com_aux_reduced_gradient(x: Array, com_std: float | Array | None = None) -> Array:
    """Gradient of :func:`com_aux_reduced_energy` with respect to every bead."""

    xb, squeezed = _as_batched(x)
    n_beads = xb.shape[-2]
    centre = jnp.mean(xb, axis=-2, keepdims=True)
    sigma = jnp.asarray(
        1.0 / jnp.sqrt(jnp.asarray(n_beads, dtype=xb.dtype)) if com_std is None else com_std,
        dtype=xb.dtype,
    )
    grad_one = centre / (jnp.asarray(n_beads, xb.dtype) * sigma * sigma)
    grad = jnp.broadcast_to(grad_one, xb.shape)
    return grad[0] if squeezed else grad


def com_aux_score(x: Array, com_std: float | Array | None = None) -> Array:
    """Score (negative reduced-energy gradient) of the auxiliary COM target."""

    return -com_aux_reduced_gradient(x, com_std=com_std)


def lift_shape_gradient(
    shape_gradient: Array,
    x: Array,
    *,
    include_com_aux: bool = True,
    com_std: float | Array | None = None,
) -> Array:
    """Project a numerical PMF gradient to shape space and optionally add COM.

    A translation-invariant PMF has a zero-sum gradient.  Projection removes
    small numerical violations produced by an external potential implementation.
    """

    grad = jnp.asarray(shape_gradient)
    if grad.shape != jnp.asarray(x).shape:
        raise ValueError(f"Gradient shape {grad.shape} does not match coordinates {jnp.asarray(x).shape}")
    grad = grad - jnp.mean(grad, axis=-2, keepdims=True)
    if include_com_aux:
        grad = grad + com_aux_reduced_gradient(x, com_std=com_std)
    return grad


def cgbg_com_log_radial_density(x: Array, com_std: float | Array | None = None) -> Array:
    """Return CG-BG's exact ``com_energy_adjustment`` value.

    Despite the upstream name, this is the log density of the *radial* COM
    norm, including the ``r**2`` Jacobian.  It is retained as a compatibility
    convention and is distinct from the Cartesian three-dimensional Gaussian
    log density used by an exact ambient target.
    """

    xb, squeezed = _as_batched(x)
    n_beads = xb.shape[-2]
    sigma = jnp.asarray(
        1.0 / jnp.sqrt(jnp.asarray(n_beads, dtype=xb.dtype)) if com_std is None else com_std,
        dtype=xb.dtype,
    )
    radius_sq = jnp.sum(jnp.mean(xb, axis=-2) ** 2, axis=-1)
    log_term = (
        jnp.log(radius_sq)
        - jnp.log(jnp.sqrt(jnp.asarray(2.0, xb.dtype)) * sigma**3)
        - jax.scipy.special.gammaln(jnp.asarray(1.5, xb.dtype))
    )
    result = -radius_sq / (2.0 * sigma**2) + log_term
    return result[0] if squeezed else result


def cgbg_correct_log_density(
    logq_ambient: Array,
    standardized_x: Array,
    physical_std: float | Array,
) -> Array:
    """Apply the upstream CG-BG COM and scalar-standardisation corrections."""

    x = jnp.asarray(standardized_x)
    dim = x.shape[-2] * x.shape[-1]
    std = jnp.asarray(physical_std, dtype=x.dtype)
    if std.ndim != 0:
        raise ValueError("physical_std must be scalar")
    return (
        jnp.asarray(logq_ambient)
        - cgbg_com_log_radial_density(x)
        - jnp.asarray(dim, x.dtype) * jnp.log(std)
    )


def canonical_relative_displacements(
    physical_coordinates: Array,
    box: Array,
    *,
    anchor: int = 0,
) -> Array:
    """Wrap bead displacements from one anchor into ``[-L/2, L/2)``.

    This helper does not change the ambient COM.  It is provided for an
    optional, explicit fundamental-domain policy; CG-BG compatibility mode
    itself leaves the support unrestricted.
    """

    xb, squeezed = _as_batched(physical_coordinates)
    box_arr = jnp.asarray(box, dtype=xb.dtype)
    lengths = jnp.diag(box_arr) if box_arr.shape == (3, 3) else box_arr
    if lengths.shape != (3,):
        raise ValueError(f"Expected box lengths (3,) or box matrix (3,3), got {box_arr.shape}")
    delta = xb - xb[:, anchor : anchor + 1]
    wrapped = jnp.mod(delta + 0.5 * lengths, lengths) - 0.5 * lengths
    return wrapped[0] if squeezed else wrapped


def domain_support_mask(
    physical_coordinates: Array,
    *,
    box: Array | None = None,
    mode: Literal["ambient_all", "relative_fundamental"] = "ambient_all",
    anchor: int = 0,
    margin: float = 0.0,
    atol: float = 1e-7,
) -> Array:
    """Return finite-coordinate and optional fundamental-domain support mask."""

    xb, squeezed = _as_batched(physical_coordinates)
    mask = jnp.all(jnp.isfinite(xb), axis=(-2, -1))
    if mode == "ambient_all":
        return mask[0] if squeezed else mask
    if mode != "relative_fundamental":
        raise ValueError(f"Unknown support mode: {mode}")
    if box is None:
        raise ValueError("box is required for relative_fundamental support")
    box_arr = jnp.asarray(box, dtype=xb.dtype)
    lengths = jnp.diag(box_arr) if box_arr.shape == (3, 3) else box_arr
    if lengths.shape != (3,):
        raise ValueError(f"Expected box lengths (3,) or box matrix (3,3), got {box_arr.shape}")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    boundary = 0.5 * lengths - jnp.asarray(margin, dtype=xb.dtype)
    delta = xb - xb[:, anchor : anchor + 1]
    inside = (delta >= (-boundary - atol)) & (delta < (boundary - atol))
    mask = mask & jnp.all(inside, axis=(-2, -1))
    return mask[0] if squeezed else mask


@dataclass(frozen=True)
class AmbientTransform:
    """Serializable transform parameters for one full-rank ambient experiment.

    The class is particle-count agnostic: ``n_beads=22`` represents the
    all-atom Ala2 ``66 = 63 + 3`` construction.  In standardized coordinates
    its exact source is ``N(0, I_66)`` and its default auxiliary COM satisfies
    ``sqrt(22) * mean(x) ~ N(0, I_3)``.
    """

    n_beads: int
    physical_std: float
    spatial_dim: int = 3
    com_std: float | None = None

    def __post_init__(self) -> None:
        if self.n_beads <= 0:
            raise ValueError("n_beads must be positive")
        if self.spatial_dim != 3:
            raise ValueError("Only three-dimensional molecular coordinates are supported")
        if not self.physical_std > 0:
            raise ValueError("physical_std must be positive")
        if self.com_std is not None and not self.com_std > 0:
            raise ValueError("com_std must be positive")

    @property
    def ambient_dim(self) -> int:
        return self.n_beads * self.spatial_dim

    @property
    def shape_dim(self) -> int:
        """Dimension of the intrinsic translation-free shape coordinates."""

        return (self.n_beads - 1) * self.spatial_dim

    @property
    def com_dim(self) -> int:
        """Dimension of the orthogonal auxiliary COM coordinate."""

        return self.spatial_dim

    @property
    def source_event_shape(self) -> tuple[int, int]:
        """Event shape for the full-rank ``N(0, I_(3N))`` source."""

        return (self.n_beads, self.spatial_dim)

    @property
    def resolved_com_std(self) -> float:
        return self.n_beads ** -0.5 if self.com_std is None else self.com_std

    def centre_and_standardize(self, physical_coordinates: Array) -> Array:
        shape, _ = split_shape_com(physical_coordinates)
        return shape / jnp.asarray(self.physical_std, dtype=shape.dtype)

    def add_com_noise(self, centred_standardized: Array, key: Array) -> Array:
        xb, squeezed = _as_batched(centred_standardized)
        shape, _ = split_shape_com(xb)
        noise = jax.random.normal(key, (xb.shape[0], 1, 3), dtype=xb.dtype)
        noise = noise * jnp.asarray(self.resolved_com_std, dtype=xb.dtype)
        result = shape + noise
        return result[0] if squeezed else result

    def to_physical(self, standardized_ambient: Array) -> Array:
        x = jnp.asarray(standardized_ambient)
        return x * jnp.asarray(self.physical_std, dtype=x.dtype)

    def split_density_coordinates(
        self, standardized_ambient: Array
    ) -> tuple[Array, Array]:
        """Return orthogonal ``(shape, normalized_com)`` density coordinates."""

        x = jnp.asarray(standardized_ambient)
        if x.shape[-2:] != self.source_event_shape:
            raise ValueError(
                f"Expected coordinates ending in {self.source_event_shape}, got {x.shape}"
            )
        return split_orthogonal_shape_com(x)

    def combine_density_coordinates(
        self, shape: Array, normalized_com: Array
    ) -> Array:
        """Reconstruct ambient coordinates from the exact density components."""

        ambient = combine_orthogonal_shape_com(shape, normalized_com)
        if ambient.shape[-2:] != self.source_event_shape:
            raise ValueError(
                f"Density components do not describe {self.n_beads} particles"
            )
        return ambient

    def source_density(
        self, standardized_ambient: Array
    ) -> AmbientDensityComponents:
        """Exact ``N(0, I_(3N))`` source density split into shape and COM."""

        self.split_density_coordinates(standardized_ambient)
        return standard_normal_ambient_density(standardized_ambient)

    def exact_density(
        self,
        shape_log_prob: Array,
        standardized_ambient: Array,
    ) -> AmbientDensityComponents:
        """Combine any intrinsic shape density with the standard COM target."""

        self.split_density_coordinates(standardized_ambient)
        if not math.isclose(
            self.resolved_com_std,
            self.n_beads**-0.5,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "exact_density requires com_std=1/sqrt(N), so the orthogonal "
                "COM coordinate has a standard-normal target"
            )
        return exact_ambient_density(shape_log_prob, standardized_ambient)

    def logq_cgbg_compat(self, logq_ambient: Array, standardized_ambient: Array) -> Array:
        return cgbg_correct_log_density(logq_ambient, standardized_ambient, self.physical_std)


@dataclass(frozen=True)
class RelativeDomainSpec:
    """Pinned relative-coordinate domain and its training-wall margin."""

    box: tuple[float, float, float]
    anchor: int = 0
    margin: float = 0.0
    support_mode: str = "relative_fundamental"

    def __post_init__(self) -> None:
        if len(self.box) != 3 or any(length <= 0 for length in self.box):
            raise ValueError("box must contain three positive lengths")
        if self.anchor < 0:
            raise ValueError("anchor must be non-negative")
        if self.margin < 0 or any(self.margin >= 0.5 * length for length in self.box):
            raise ValueError("margin must lie inside every half box length")

    def support_mask(self, physical_coordinates: Array, *, use_margin: bool = False) -> Array:
        """Hard final support; by default the soft-wall margin is not excluded."""

        return domain_support_mask(
            physical_coordinates,
            box=jnp.asarray(self.box),
            mode="relative_fundamental",
            anchor=self.anchor,
            margin=self.margin if use_margin else 0.0,
        )

    def metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RelativeSoftWall:
    """C1 flat-bottom quadratic wall starting ``margin`` before the boundary.

    The wall is a training stabilizer, not part of the formal PMF reweighting
    target. Coordinates and box lengths are physical and must use the same unit.
    """

    domain: RelativeDomainSpec
    strength: float

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ValueError("strength must be non-negative")

    def reduced_energy(self, physical_coordinates: Array) -> Array:
        xb, squeezed = _as_batched(physical_coordinates)
        lengths = jnp.asarray(self.domain.box, dtype=xb.dtype)
        threshold = 0.5 * lengths - jnp.asarray(self.domain.margin, dtype=xb.dtype)
        delta = xb - xb[:, self.domain.anchor : self.domain.anchor + 1]
        # relu(z)^2 is continuously differentiable at the flat-bottom edge.
        excess = jax.nn.relu(jnp.abs(delta) - threshold)
        energy = 0.5 * jnp.asarray(self.strength, xb.dtype) * jnp.sum(excess**2, axis=(-2, -1))
        return energy[0] if squeezed else energy

    def reduced_gradient(self, physical_coordinates: Array) -> Array:
        x = jnp.asarray(physical_coordinates)
        if x.ndim == 2:
            return jax.grad(self.reduced_energy)(x)
        if x.ndim != 3:
            raise ValueError(f"Expected (N,3) or (B,N,3), got {x.shape}")
        return jax.vmap(jax.grad(self.reduced_energy))(x)

    def metadata(self) -> dict[str, object]:
        return {**self.domain.metadata(), "strength": self.strength, "kind": "relative_flat_bottom_quadratic"}
