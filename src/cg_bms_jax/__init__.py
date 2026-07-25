"""Pure-JAX coarse-grained Bridge Matching Sampler.

The published CG-BG MACE checkpoint contains GPU scatter reductions whose
non-deterministic lowering can alter energies appreciably.  Configure the same
deterministic XLA policy as the pinned reference before any package submodule
imports JAX.
"""

import os
from importlib.metadata import PackageNotFoundError, version

_DETERMINISTIC_XLA_FLAG = "--xla_gpu_deterministic_ops=true"
_xla_flags = os.environ.get("XLA_FLAGS", "").split()
if _DETERMINISTIC_XLA_FLAG not in _xla_flags:
    _xla_flags.append(_DETERMINISTIC_XLA_FLAG)
    os.environ["XLA_FLAGS"] = " ".join(_xla_flags)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    __version__ = version("cg-bms-jax")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
