from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_validator():
    path = Path(__file__).parents[1] / "scripts" / "validate_publication_tree.py"
    spec = importlib.util.spec_from_file_location("validate_publication_tree", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def test_allowlist_excludes_historical_and_all_atom_material(tmp_path: Path) -> None:
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text(
        "\n".join(
            [
                "tree src",
                "glob configs/experiment ala2_ambient18*.yaml",
                "tree artifacts/release_v0.1.0",
            ]
        ),
        encoding="utf-8",
    )
    rules = validator.load_allowlist(allowlist)

    assert validator.is_allowed("src/cg_bms_jax/runtime.py", rules)
    assert validator.is_allowed(
        "configs/experiment/ala2_ambient18_300k.yaml", rules
    )
    assert validator.is_allowed(
        "artifacts/release_v0.1.0/figures/ala2_cg/rama.png", rules
    )
    assert not validator.is_allowed(
        "configs/experiment/ala2_allatom_bms_energy_formal.yaml", rules
    )
    assert not validator.is_allowed("artifacts/old_probe/result.json", rules)


def test_path_gate_rejects_large_file_and_all_atom_result(tmp_path: Path) -> None:
    rules = (
        validator.AllowRule("tree", "artifacts/release_v0.1.0"),
        validator.AllowRule("tree", "configs"),
    )
    all_atom = "configs/experiment/ala2_allatom_bms.yaml"
    large = "artifacts/release_v0.1.0/figures/mb2d_analytic/too_large.png"
    (tmp_path / all_atom).parent.mkdir(parents=True)
    (tmp_path / all_atom).write_text("experiment: allatom\n", encoding="utf-8")
    (tmp_path / large).parent.mkdir(parents=True)
    with (tmp_path / large).open("wb") as handle:
        handle.truncate(validator.MAX_FILE_BYTES + 1)

    errors = validator._validate_paths(tmp_path, {all_atom, large}, rules)
    assert any("all-atom benchmark material" in error for error in errors)
    assert any("25 MiB" in error for error in errors)


def test_text_gate_rejects_pending_marker_and_secret(tmp_path: Path) -> None:
    files = {
        "README.md": "result: PENDING_VERIFIED_AGGREGATION\n",
        "docs/RELEASE_RESULTS_V0.1.0.md": (
            "token accidentally copied: github_pat_" + "A" * 24 + "\n"
        ),
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    errors = validator._validate_text(tmp_path, set(files))
    assert any("unresolved release placeholder" in error for error in errors)
    assert any("GitHub token" in error for error in errors)


def test_readme_link_gate_reports_missing_relative_target(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "[good](docs/good.md) and [bad](docs/missing.md)\n", encoding="utf-8"
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/good.md").write_text("ok\n", encoding="utf-8")

    errors = validator._validate_readme_links(tmp_path, {"README.md"})
    assert errors == [
        "README link target is excluded from publication: docs/good.md",
        "broken README link: docs/missing.md",
    ]


def test_final_required_gate_requires_atomic_release_manifest() -> None:
    errors = validator._validate_required({"README.md"}, "final")
    assert any("artifacts/release_v0.1.0/MANIFEST.json" in error for error in errors)


def test_source_required_gate_does_not_require_atomic_release_manifest() -> None:
    files = set(validator.SOURCE_REQUIRED_FILES)
    files.update(
        f"{prefix}placeholder" for prefix in validator.SOURCE_REQUIRED_TREES
    )
    errors = validator._validate_required(files, "source")
    assert not errors


@pytest.mark.parametrize("suffix", [".tar", ".tgz", ".zip"])
def test_archives_are_forbidden(tmp_path: Path, suffix: str) -> None:
    relative = f"src/archive{suffix}"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"archive")
    rules = (validator.AllowRule("tree", "src"),)

    errors = validator._validate_paths(tmp_path, {relative}, rules)
    assert any("archive is forbidden" in error for error in errors)
