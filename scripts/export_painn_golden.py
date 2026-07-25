#!/usr/bin/env python3
"""Export a tiny upstream Torch PaiNN golden fixture.

Run this script in the original ``cg-bms`` Torch environment, not in the pure
JAX runtime environment.  The generated ``npz`` contains only numeric arrays
and JSON metadata; consumers load it with ``allow_pickle=False`` and never need
Torch at test time.

Example
-------
```
/ds/project/weilong/ke/miniconda3/envs/cg-bms/bin/python \
  scripts/export_painn_golden.py \
  --bms-root /ds/project/weilong/ke/cg-bms \
  --output tests/fixtures/painn_tiny_torch_golden.npz
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

UPSTREAM_REVISION = "d19a27b854fd77387c43109e37b597bc35252e7d"


def _encode_state_key(key: str) -> str:
    return "state__" + key.replace(".", "__")


def _positions() -> np.ndarray:
    return np.asarray(
        [
            [
                [-0.20, 0.01, 0.03],
                [-0.08, 0.12, -0.02],
                [0.02, 0.02, 0.09],
                [0.06, -0.11, 0.04],
                [0.15, 0.04, -0.08],
                [0.24, -0.05, 0.01],
            ],
            [
                [-0.19, -0.03, 0.06],
                [-0.10, 0.10, 0.01],
                [0.01, 0.04, 0.11],
                [0.07, -0.09, 0.02],
                [0.16, 0.01, -0.07],
                [0.23, -0.03, -0.01],
            ],
        ],
        dtype=np.float32,
    )


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bms-root",
        type=Path,
        help="Optional BridgeMatchingSampler/cg-bms checkout containing src/bms.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/painn_tiny_torch_golden.npz"),
    )
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    if args.bms_root is not None:
        source = args.bms_root.expanduser().resolve() / "src"
        if not source.is_dir():
            raise FileNotFoundError(f"BMS source directory not found: {source}")
        sys.path.insert(0, str(source))

    # Import only after an optional source checkout has been prepended.
    from bms.model.painn import PaiNN  # noqa: PLC0415

    torch.manual_seed(args.seed)
    model_config: dict[str, Any] = {
        "num_features": 8,
        "num_radial_basis": 4,
        "num_layers": 2,
        "num_elements": 6,
        "r_max": 0.8,
        "r_offset": 0.0,
        "time_init_mode": "node",
        "parity_breaking": True,
        "conservative": False,
        "unique_atom_indices": True,
    }
    model = PaiNN(**model_config).eval()
    positions = torch.from_numpy(_positions())
    time = torch.tensor([0.2, 0.7], dtype=torch.float32)

    captured: dict[str, np.ndarray] = {}
    handles = []

    def capture_pair(name: str):
        def hook(_module, _inputs, output):
            scalar, vector = output
            captured[f"activation__{name}__scalar"] = _as_numpy(scalar)
            captured[f"activation__{name}__vector"] = _as_numpy(vector)

        return hook

    for index in range(model_config["num_layers"]):
        handles.append(model.messages[index].register_forward_hook(capture_pair(f"message_{index}")))
        handles.append(model.norms_1[index].register_forward_hook(capture_pair(f"norm_1_{index}")))
        handles.append(model.updates[index].register_forward_hook(capture_pair(f"update_{index}")))
        handles.append(model.norms_2[index].register_forward_hook(capture_pair(f"norm_2_{index}")))
    for index, block in enumerate(model.output_block.out_vector):
        handles.append(block.register_forward_hook(capture_pair(f"output_block_{index}")))

    try:
        with torch.no_grad():
            output = model(time, positions)
    finally:
        for handle in handles:
            handle.remove()

    metadata = {
        "schema_version": 1,
        "upstream_revision": UPSTREAM_REVISION,
        "torch_version": torch.__version__,
        "seed": args.seed,
        "dtype": "float32",
        "model": model_config,
        "dense_layout": "torch_out_in",
        "cross_product": "sender_vector_cross_unit_ij",
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "positions": _as_numpy(positions),
        "time": _as_numpy(time),
        "output": _as_numpy(output),
        **captured,
    }
    for key, value in model.state_dict().items():
        arrays[_encode_state_key(key)] = _as_numpy(value)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "arrays": len(arrays),
                "state_tensors": len(model.state_dict()),
                "metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

