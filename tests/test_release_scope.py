from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_validator():
    path = Path(__file__).parents[1] / "scripts" / "validate_release_scope.py"
    spec = importlib.util.spec_from_file_location("validate_release_scope", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _write_bundle(root: Path, *, forbidden: bool = False) -> None:
    files = {
        "BENCHMARKS.md": b"frozen benchmark summary\n",
        "metrics/mb_cg1d/summary.json": b'{"ess_fraction": 0.5}\n',
        "metrics/mb2d_analytic/cold_energy.json": b'{"ess_fraction": 0.8}\n',
        "metrics/ala2_cg/warm_bridge.json": b'{"ess_fraction": 0.1}\n',
        "figures/mb_cg1d/density.png": b"png",
        "figures/mb2d_analytic/cold_energy.png": b"png",
        "figures/ala2_cg/rama.png": b"png",
        "parameters/mb_cg1d/config.yaml": b"experiment: mb_cg1d\n",
        "parameters/mb2d_analytic/config.yaml": b"experiment: mb2d_analytic\n",
        "parameters/ala2_cg/config.yaml": b"experiment: ala2_ambient18_300k\n",
    }
    if forbidden:
        files["metrics/ala2_allatom/openmm.json"] = (
            b'{"coordinate_mode": "ambient66"}\n'
        )

    records = []
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "semantic_role": "test fixture",
            }
        )

    manifest = {
        "schema_version": 1,
        "release_scope": {
            "included_benchmark_families": ["mb_cg1d", "mb2d_analytic", "ala2_cg"],
            "excluded_benchmark_families": ["ala2_all_atom"],
        },
        "files": records,
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_release_scope_accepts_only_declared_cg_benchmarks(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    result = validator.validate_release_bundle(tmp_path)
    assert result["verification"] == "PASS"
    assert result["included_benchmark_families"] == [
        "ala2_cg",
        "mb2d_analytic",
        "mb_cg1d",
    ]


def test_release_scope_rejects_all_atom_result_path(tmp_path: Path) -> None:
    _write_bundle(tmp_path, forbidden=True)
    with pytest.raises(
        validator.ScopeValidationError, match="forbidden all-atom result marker"
    ):
        validator.validate_release_bundle(tmp_path)


def test_release_scope_uses_token_boundaries_for_66d_marker() -> None:
    assert validator._contains_marker('{"coordinate_mode": "66D"}') == "66d"
    assert validator._contains_marker("ala2_66d_checkpoint") == "66d"
    assert validator._contains_marker('{"sha256": "abc66def0123"}') is None
    assert validator._contains_marker('{"sha256": "0123abc66d"}') is None


def test_release_scope_rejects_unlisted_file(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "figures" / "mb_cg1d" / "extra.png").write_bytes(b"not recorded")
    with pytest.raises(
        validator.ScopeValidationError, match="files missing from MANIFEST"
    ):
        validator.validate_release_bundle(tmp_path)
