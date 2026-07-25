"""Neural controllers used by the JAX Bridge Matching Sampler.

The :mod:`cg_bms_jax.model.painn` implementation is a direct JAX/Flax port of
the PaiNN controller shipped with BridgeMatchingSampler.  Translation handling
is deliberately kept outside the backbone: :class:`ShapeCOMController` combines
the mean-free molecular control with an explicit ambient COM control.
"""

from cg_bms_jax.model.controller import (
    MLPController,
    RadialCOMHead,
    ShapeCOMController,
)
from cg_bms_jax.model.painn import (
    AtomEmbedding,
    EquivariantRMSNorm,
    GatedEquivariantBlock,
    GaussianSmearing,
    MessageBlock,
    PaiNN,
    ParityBreakingMessageBlock,
    PolynomialEnvelope,
    RadialBasis,
    TimeEmbedding,
    TorchLinear,
    UpdateBlock,
    VectorHead,
)

__all__ = [
    "AtomEmbedding",
    "EquivariantRMSNorm",
    "GatedEquivariantBlock",
    "GaussianSmearing",
    "MLPController",
    "MessageBlock",
    "PaiNN",
    "ParityBreakingMessageBlock",
    "PolynomialEnvelope",
    "RadialBasis",
    "RadialCOMHead",
    "ShapeCOMController",
    "TimeEmbedding",
    "TorchLinear",
    "UpdateBlock",
    "VectorHead",
]
