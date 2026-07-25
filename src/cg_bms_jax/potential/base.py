"""Small framework-independent potential protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import jax

Array = jax.Array


@dataclass(frozen=True)
class ScoreDecomposition:
    """Terminal-score terms with distinct clipping semantics.

    ``clippable`` is the molecular/shape score that may be stabilized before
    entering BMS replay.  ``preserved`` is an analytic full-rank score (for
    example the Gaussian COM auxiliary) that must be restored unchanged.
    """

    clippable: Array
    preserved: Array


@dataclass(frozen=True)
class PotentialResult:
    """Energy and score evaluated in the coordinates accepted by a potential.

    ``gradient`` is the gradient of the dimensional energy.  The reduced
    quantities use ``U/kT`` and ``score == -reduced_gradient``.
    """

    energy: Array
    gradient: Array
    reduced_energy: Array
    reduced_gradient: Array
    score: Array
    valid_mask: Array
    components: Mapping[str, Array] = field(default_factory=dict)
    score_decomposition: ScoreDecomposition | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "energy": self.energy,
            "gradient": self.gradient,
            "reduced_energy": self.reduced_energy,
            "reduced_gradient": self.reduced_gradient,
            "score": self.score,
            "valid_mask": self.valid_mask,
            "components": dict(self.components),
            "score_decomposition": self.score_decomposition,
        }


@runtime_checkable
class Potential(Protocol):
    kT: float

    def energy(self, coordinates: Array) -> Array: ...

    def energy_and_grad(self, coordinates: Array) -> tuple[Array, Array]: ...

    def evaluate(self, coordinates: Array) -> PotentialResult: ...
