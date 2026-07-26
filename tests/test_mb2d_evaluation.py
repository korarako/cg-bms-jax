from __future__ import annotations

import numpy as np
import pytest

from cg_bms_jax.evaluation.mb2d import (
    EXACT_ENERGY_DISPLAY_SMOOTH_SIGMA_BINS,
    _smooth_density_for_display,
    analytic_mb2d_reference,
    compare_likelihood_archives,
    evaluate_mb2d,
    evaluate_pooled_mb2d,
    mb2d_distribution_metrics,
    muller_brown_energy_numpy,
    sample_size_convergence,
    temperature_reweighting_sweep,
)
from cg_bms_jax.evaluation.metrics import importance_weight_diagnostics
from cg_bms_jax.potential.mb import muller_brown_energy


def test_numpy_mb2d_energy_matches_jax_definition() -> None:
    points = np.asarray(
        [[0.0, 0.0], [23.0, 31.0], [31.2, 15.5], [42.0, 8.5], [50.0, 50.0]]
    )
    np.testing.assert_allclose(
        muller_brown_energy_numpy(points),
        np.asarray(muller_brown_energy(points)),
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_analytic_reference_and_basin_metrics_are_normalized() -> None:
    reference = analytic_mb2d_reference(beta=1.0, bins=80)
    np.testing.assert_allclose(reference.probability.sum(), 1.0)
    assert np.isfinite(reference.energy).all()
    rng = np.random.default_rng(4)
    indices = rng.choice(
        reference.probability.size,
        size=30_000,
        p=reference.probability.reshape(-1),
    )
    samples = reference.points.reshape(-1, 2)[indices]
    metrics = mb2d_distribution_metrics(samples, reference=reference)
    assert metrics["js_2d"] < 0.03
    assert metrics["basin_l1_error"] < 0.04
    assert abs(metrics["energy_mean_error"]) < 0.2


def test_importance_weight_diagnostics_reports_concentration() -> None:
    weights = np.asarray([0.7, 0.2, 0.08, 0.02])
    result = importance_weight_diagnostics(
        weights,
        logw_raw=np.log(weights),
        top_fractions=(0.25, 0.5),
    )
    np.testing.assert_allclose(result["ess"], 1.0 / np.sum(weights**2))
    np.testing.assert_allclose(result["top_25_percent_mass"], 0.7)
    np.testing.assert_allclose(result["top_50_percent_mass"], 0.9)


def test_exact_energy_display_smoothing_preserves_mass_and_reduces_spikes() -> None:
    density = np.zeros(41, dtype=np.float64)
    density[[8, 15, 24, 33]] = [1.0, 0.6, 0.9, 0.4]
    smoothed = _smooth_density_for_display(density)
    np.testing.assert_allclose(smoothed.sum(), density.sum(), rtol=1.0e-12)
    assert smoothed.max() < density.max()
    assert np.sum(np.abs(np.diff(smoothed))) < np.sum(np.abs(np.diff(density)))
    assert EXACT_ENERGY_DISPLAY_SMOOTH_SIGMA_BINS == 2.0


def _uniform_proposal(seed: int = 7, size: int = 800) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, 50.0, size=(size, 2))
    energy = muller_brown_energy_numpy(coordinates)
    # Uniform q on the finite box is constant, so the omitted additive logq
    # constant cancels after normalization.
    raw = -energy
    shifted = raw - np.max(raw)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    return {
        "R": coordinates,
        "U": energy,
        "logp": np.zeros(size),
        "logq_ambient": np.zeros(size),
        "logw_raw": raw,
        "weights": weights,
        "valid_mask": np.ones(size, dtype=bool),
        "support_mask": np.ones(size, dtype=bool),
        "initial_X": coordinates.copy(),
    }


def test_mb2d_evaluation_and_sweeps_write_outputs(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    sample = _uniform_proposal()
    evaluation = evaluate_mb2d(
        sample=sample,
        output_dir=tmp_path / "evaluation",
        bins=50,
    )
    assert evaluation["metrics"]["weighted"] is not None
    assert evaluation["metrics"]["plot_style"]["palette"]["proposal"] == ("#F2A174")
    assert (
        evaluation["metrics"]["plot_style"]["palette"]["reweighted_outline"]
        == "#4472C4"
    )
    assert evaluation["metrics"]["plot_style"]["fes_colormap"] == "viridis"
    smoothing = evaluation["metrics"]["energy_plot_view"]["exact_curve_smoothing"]
    assert smoothing["display_only"]
    assert smoothing["mass_preserved"]
    assert smoothing["formal_metrics_unchanged"]
    assert (tmp_path / "evaluation" / "mb2d_plots.png").is_file()
    for filename in (
        "mb2d_exact_fes.png",
        "mb2d_proposal_fes.png",
        "mb2d_reweighted_fes.png",
        "mb2d_x_marginal.png",
        "mb2d_y_marginal.png",
        "mb2d_energy_distribution.png",
    ):
        assert (tmp_path / "evaluation" / filename).is_file()
    assert (tmp_path / "evaluation" / "mb2d_metrics.json").is_file()

    convergence = sample_size_convergence(
        sample=sample,
        output_dir=tmp_path / "convergence",
        sizes=(100, 400, 800),
        repeats=2,
        bins=40,
    )
    assert len(convergence["rows"]) == 6
    assert (tmp_path / "convergence" / "sample_size_convergence.png").is_file()

    temperature = temperature_reweighting_sweep(
        sample=sample,
        output_dir=tmp_path / "temperature",
        betas=(0.5, 1.0, 2.0),
        bins=40,
    )
    assert [row["beta"] for row in temperature["rows"]] == [0.5, 1.0, 2.0]
    assert (tmp_path / "temperature" / "temperature_reweighting.png").is_file()


def test_mb2d_energy_view_is_target_anchored_and_reports_outlier_mass(
    tmp_path,
) -> None:
    pytest.importorskip("matplotlib")
    base = _uniform_proposal(size=800)
    baseline = evaluate_mb2d(
        sample=base,
        output_dir=tmp_path / "baseline",
        bins=40,
    )
    outlier_point = np.asarray([[75.0, 75.0]])
    outlier_energy = muller_brown_energy_numpy(outlier_point)
    contaminated = {
        "R": np.concatenate((base["R"], outlier_point), axis=0),
        "U": np.concatenate((base["U"], outlier_energy), axis=0),
        "weights": np.concatenate((base["weights"] * 0.999, np.asarray([0.001]))),
    }
    result = evaluate_mb2d(
        sample=contaminated,
        output_dir=tmp_path / "contaminated",
        bins=40,
    )
    assert result["metrics"]["num_samples"] == 801
    assert result["metrics"]["energy_plot_view"]["sample_usage"] == "all"
    np.testing.assert_allclose(
        result["metrics"]["energy_plot_view"]["limits"],
        baseline["metrics"]["energy_plot_view"]["limits"],
    )
    assert (
        result["metrics"]["energy_plot_view"]["proposal"]["above_view_mass"]
        >= 1.0 / 801.0
    )
    assert (
        result["metrics"]["unweighted"]["energy_mean"]
        != baseline["metrics"]["unweighted"]["energy_mean"]
    )


def test_pooled_mb2d_visualization_uses_every_seed_sample(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    first = _uniform_proposal(seed=1, size=400)
    second = _uniform_proposal(seed=2, size=400)
    result = evaluate_pooled_mb2d(
        {"seed_0": first, "seed_1": second},
        output_dir=tmp_path / "pooled",
        bins=40,
    )
    assert result["metrics"]["num_samples"] == 800
    assert result["metrics"]["pooling"]["all_samples_used"]
    assert result["metrics"]["pooling"]["sample_counts"] == [400, 400]
    assert (tmp_path / "pooled" / "pool_manifest.json").is_file()
    assert (tmp_path / "pooled" / "mb2d_plots.png").is_file()
    assert (tmp_path / "pooled" / "mb2d_energy_distribution.png").is_file()


def test_mb2d_clip1_diagnostic_uses_raw_log_weights(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    sample = _uniform_proposal(size=800)
    result = evaluate_mb2d(
        sample=sample,
        output_dir=tmp_path / "clip1",
        bins=40,
        clip_percentile=99.0,
        clip_mode="drop",
    )
    transform = result["metrics"]["weight_transform"]
    assert not transform["formal_no_clip"]
    assert transform["clip_percentile"] == 99.0
    assert transform["removed_upper_percent"] == 1.0
    assert transform["clip_mode"] == "drop"
    assert transform["zero_weight_count"] >= 8
    assert result["metrics"]["weight_source"].startswith("logw_raw_drop_p99")


def test_pooled_mb2d_clip1_is_applied_per_seed(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    result = evaluate_pooled_mb2d(
        {
            "seed_0": _uniform_proposal(seed=1, size=400),
            "seed_1": _uniform_proposal(seed=2, size=400),
        },
        output_dir=tmp_path / "pooled_clip1",
        bins=40,
        clip_percentile=99.0,
        clip_mode="drop",
    )
    pooling = result["metrics"]["pooling"]
    assert pooling["total_samples"] == 800
    assert pooling["clip_percentile"] == 99.0
    assert pooling["removed_upper_percent"] == 1.0
    assert pooling["clip_scope"] == "per_seed_before_equal_mass_pooling"
    assert not result["metrics"]["weight_transform"]["formal_no_clip"]


def test_likelihood_archive_comparison_checks_common_initial_state() -> None:
    base = _uniform_proposal(size=20)
    strict = {
        **base,
        "logq_ambient": np.linspace(-3.0, -1.0, 20),
    }
    loose = {
        **strict,
        "R": strict["R"] + 1.0e-3,
        "logq_ambient": strict["logq_ambient"] + 2.0e-3,
    }
    result = compare_likelihood_archives(
        {"strict": strict, "loose": loose},
        reference_label="strict",
    )
    assert result["runs"]["loose"]["initial_samples_bitwise_identical"]
    np.testing.assert_allclose(result["runs"]["loose"]["coordinate_rmse"], 1.0e-3)
    np.testing.assert_allclose(result["runs"]["loose"]["logq_rmse"], 2.0e-3)
