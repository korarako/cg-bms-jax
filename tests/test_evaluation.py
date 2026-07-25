from __future__ import annotations

import numpy as np
import pytest

from cg_bms_jax.evaluation import (
    bootstrap_distribution_metrics,
    compute_dihedral,
    evaluate_ala2,
    evaluate_mb,
    histogram_js_divergence,
    plot_ala2_clip_sweep,
    resolve_weights,
)
from cg_bms_jax.evaluation.mb import _extract_mb_x


def test_signed_dihedral_known_right_angle() -> None:
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(compute_dihedral(points), -0.5 * np.pi)


def test_resolve_weights_supports_proposal_only_and_energy_logq() -> None:
    weights, logw, source = resolve_weights({"R": np.arange(4)}, kT=2.0)
    assert weights is None and logw is None and source == "proposal_only"

    weights, logw, source = resolve_weights(
        {"U": np.asarray([1.0, 2.0]), "logp": np.asarray([-2.0, -4.0])},
        kT=2.0,
    )
    np.testing.assert_allclose(logw, np.asarray([1.5, 3.0]))
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert source == "computed_from_U_logp"


def test_histogram_metrics_and_bootstrap_shapes() -> None:
    rng = np.random.default_rng(12)
    target = rng.normal(size=(128, 2))
    assert histogram_js_divergence(target, target, bins=20, ranges=((-4.0, 4.0),) * 2) < 1e-12
    bootstrap = bootstrap_distribution_metrics(
        target,
        target.copy(),
        bins=12,
        ranges=((-4.0, 4.0),) * 2,
        n_bootstraps=4,
        seed=7,
    )
    assert set(bootstrap) == {"JS_Divergence", "PMF_Error", "ESS_Percent"}
    assert all(values.shape == (4,) for values in bootstrap.values())


def test_clip_sweep_actually_saves_figure(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(4)
    target = {"R": rng.normal(size=(64, 6, 3))}
    sample = {
        "R": rng.normal(size=(64, 6, 3)),
        "logw": np.linspace(-2.0, 2.0, 64),
    }
    output = tmp_path / "clip_sweep.png"
    result = plot_ala2_clip_sweep(
        target=target,
        sample=sample,
        output_path=output,
        clip_percentiles=(90.0, 95.0),
    )
    assert output.is_file() and output.stat().st_size > 0
    assert output.with_suffix(".npz").is_file()
    assert result["metrics"].shape == (2, 3)


def test_mb_evaluation_proposal_only(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    rng = np.random.default_rng(9)
    target = {"R": np.clip(rng.normal(31.0, 5.0, 96), 0.0, 50.0)}
    sample = {"R": np.clip(rng.normal(33.0, 6.0, 96), 0.0, 50.0)}
    result = evaluate_mb(
        target=target,
        sample=sample,
        output_dir=tmp_path,
        n_bootstraps=2,
    )
    assert result["metrics"]["weight_source"] == "proposal_only"
    assert result["metrics"]["weighted_direct"] is None
    assert result["metrics"]["plot_style"]["palette"] == {
        "exact": "#000000",
        "reference": "#8FD18B",
        "proposal": "#F2A174",
        "reweighted_outline": "#4472C4",
        "reweighted_fill": "#B9CBEA",
        "implicit_reference": "#9467BD",
    }
    assert result["metrics"]["plot_style"]["density_rendering"] == (
        "filled_with_outline"
    )
    assert (tmp_path / "mb_plots.png").is_file()
    assert (tmp_path / "mb_density.png").is_file()
    assert (tmp_path / "mb_free_energy.png").is_file()
    assert (tmp_path / "mb_metrics.json").is_file()


def test_mb_coordinate_projection_selects_x_from_full_reference() -> None:
    reference = np.asarray([[21.0, 31.0], [23.0, 33.0], [25.0, 35.0]])
    np.testing.assert_array_equal(
        _extract_mb_x(reference, full_reference=True),
        reference[:, 0],
    )
    np.testing.assert_array_equal(
        _extract_mb_x(reference[:, :1], full_reference=True),
        reference[:, 0],
    )
    with pytest.raises(ValueError, match="MB proposal coordinates"):
        _extract_mb_x(reference, full_reference=False)


def test_ala2_evaluation_saves_standalone_release_panels(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    rng = np.random.default_rng(17)
    target_energy = rng.normal(-300.0, 4.0, 96)
    sample_energy = rng.normal(-298.0, 6.0, 96)
    sample_energy[-1] = 1.0e5
    weights = np.full(96, 1.0 / 96.0)
    result = evaluate_ala2(
        target={
            "R": rng.normal(size=(96, 6, 3)),
            "U": target_energy,
        },
        sample={
            "R": rng.normal(size=(96, 6, 3)),
            "U": sample_energy,
            "weights": weights,
            "logw": np.log(weights),
        },
        output_dir=tmp_path,
        n_bootstraps=2,
    )
    for filename in (
        "ala2_cb_energy_distribution.png",
        "ala2_cb_phi_density.png",
        "ala2_cb_psi_density.png",
        "ala2_cb_phi_free_energy.png",
        "ala2_cb_psi_free_energy.png",
        "ala2_cb_reference_ramachandran_fes.png",
        "ala2_cb_proposal_ramachandran_fes.png",
        "ala2_cb_reweighted_ramachandran_fes.png",
    ):
        assert (tmp_path / filename).is_file()
    energy_view = result["metrics"]["energy_plot_view"]
    assert result["metrics"]["plot_style"]["two_dimensional_panels"] == (
        "standalone_no_overlay"
    )
    assert energy_view["display_only"] is True
    assert energy_view["limits"][1] < sample_energy[-1]
    assert energy_view["proposal_outside_fraction"] > 0.0
