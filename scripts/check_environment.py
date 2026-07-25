#!/usr/bin/env python3
"""Fail-fast validation of the reference JAX runtime."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys

EXPECTED = {
    "jax": "0.4.38",
    "jaxlib": "0.4.38",
    "jax-cuda12-plugin": "0.4.38",
    "jax-cuda12-pjrt": "0.4.38",
    "flax": "0.10.4",
    "optax": "0.2.5",
    "diffrax": "0.7.2",
    "jax-md": "0.2.8",
    "e3nn-jax": "0.21.0",
    "dm-haiku": "0.0.16",
    "chemtrain": "0.2.0",
    "chemutils": "0.0.1",
}

IMPORTS = (
    "flax",
    "optax",
    "diffrax",
    "e3nn_jax",
    "haiku",
    # Keep the same order as build_cgbg_mace_energy_fn.  The pinned
    # ChemTrain compatibility layer must load before the legacy jax-md ABI.
    "chemtrain.data.preprocessing",
    "chemtrain.learn.force_matching",
    "chemutils.models.mace",
    "jax_md.partition",
    "jax_md.space",
    "jax_md_mod.custom_quantity",
)


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python 3.11 required, got {sys.version.split()[0]}")
    for package, wanted in EXPECTED.items():
        installed = importlib.metadata.version(package)
        if installed != wanted:
            raise RuntimeError(f"{package}=={wanted} required, got {installed}")
    for module in IMPORTS:
        importlib.import_module(module)

    import jax

    devices = jax.devices()
    allow_cpu = os.environ.get("CG_BMS_ALLOW_CPU", "0") == "1"
    if not allow_cpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU backend required, got {jax.default_backend()}: {devices}")
    print(f"python={sys.version.split()[0]}")
    print(f"jax={jax.__version__} backend={jax.default_backend()} devices={devices}")


if __name__ == "__main__":
    main()
