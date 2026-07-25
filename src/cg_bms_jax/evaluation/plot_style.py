"""Shared publication palette and filled-density plotting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

EXACT_COLOR = "#000000"
REFERENCE_COLOR = "#8FD18B"
PROPOSAL_COLOR = "#F2A174"
REWEIGHTED_COLOR = "#4472C4"
REWEIGHTED_FILL_COLOR = "#B9CBEA"
IMPLICIT_REFERENCE_COLOR = "#9467BD"

REFERENCE_ALPHA = 0.52
PROPOSAL_ALPHA = 0.56
REWEIGHTED_ALPHA = 0.58


def plot_style_metadata() -> dict[str, Any]:
    """Return a JSON-serializable record of the release plotting contract."""

    return {
        "palette": {
            "exact": EXACT_COLOR,
            "reference": REFERENCE_COLOR,
            "proposal": PROPOSAL_COLOR,
            "reweighted_outline": REWEIGHTED_COLOR,
            "reweighted_fill": REWEIGHTED_FILL_COLOR,
            "implicit_reference": IMPLICIT_REFERENCE_COLOR,
        },
        "fill_alpha": {
            "reference": REFERENCE_ALPHA,
            "proposal": PROPOSAL_ALPHA,
            "reweighted": REWEIGHTED_ALPHA,
        },
        "density_rendering": "filled_with_outline",
        "fes_colormap": "viridis",
        "two_dimensional_panels": "standalone_no_overlay",
    }


def filled_curve(
    axis: Any,
    x: np.ndarray,
    density: np.ndarray,
    *,
    role: str,
    label: str,
    linewidth: float = 1.5,
) -> None:
    """Plot a filled one-dimensional density with a visible outline."""

    styles = {
        "reference": (REFERENCE_COLOR, REFERENCE_COLOR, REFERENCE_ALPHA),
        "proposal": (PROPOSAL_COLOR, PROPOSAL_COLOR, PROPOSAL_ALPHA),
        "reweighted": (
            REWEIGHTED_FILL_COLOR,
            REWEIGHTED_COLOR,
            REWEIGHTED_ALPHA,
        ),
    }
    try:
        fill_color, edge_color, alpha = styles[role]
    except KeyError as error:
        raise ValueError(f"Unknown filled-density role {role!r}") from error
    axis.fill_between(
        x,
        0.0,
        density,
        color=fill_color,
        alpha=alpha,
        linewidth=0.0,
        label=label,
    )
    axis.plot(
        x,
        density,
        color=edge_color,
        linewidth=linewidth,
        label="_nolegend_",
    )


def filled_stairs(
    axis: Any,
    density: np.ndarray,
    edges: np.ndarray,
    *,
    role: str,
    label: str,
    linewidth: float = 1.4,
) -> None:
    """Plot histogram density as a translucent fill plus crisp outline."""

    styles = {
        "reference": (REFERENCE_COLOR, REFERENCE_COLOR, REFERENCE_ALPHA),
        "proposal": (PROPOSAL_COLOR, PROPOSAL_COLOR, PROPOSAL_ALPHA),
        "reweighted": (
            REWEIGHTED_FILL_COLOR,
            REWEIGHTED_COLOR,
            REWEIGHTED_ALPHA,
        ),
    }
    try:
        fill_color, edge_color, alpha = styles[role]
    except KeyError as error:
        raise ValueError(f"Unknown filled-density role {role!r}") from error
    axis.stairs(
        density,
        edges,
        fill=True,
        facecolor=fill_color,
        edgecolor="none",
        alpha=alpha,
        label=label,
    )
    axis.stairs(
        density,
        edges,
        fill=False,
        color=edge_color,
        linewidth=linewidth,
        label="_nolegend_",
    )
