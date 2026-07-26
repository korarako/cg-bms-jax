# v0.1 benchmark and frozen-result contract

This document defines the scientific claim boundary and artifact contract for
the first public `cg-bms-jax` release. It intentionally separates:

1. formal, unclipped importance-sampling results;
2. optional clipped diagnostics;
3. display-only plot windows;
4. historical stress tests; and
5. experimental source modules that are not benchmarked.

The authoritative frozen payload is
[`artifacts/release_v0.1.0`](../artifacts/release_v0.1.0/README.md).
Numerical values live in `summary.json` and `per_seed.json`; provenance and
input identities live in `SOURCE_RESULTS_INDEX.json`, `provenance/`, and
`MANIFEST.json`. This document does not duplicate values that can drift from
those machine-readable records.

## 1. Release claim

The release demonstrates that a stochastic BMS proposal can be paired with a
matched backward controller and a probability-flow ODE to obtain sample-bound
proposal densities. Importance weights

\[
\log w_i^{\rm raw}=-\beta U(x_i)-\log q(x_i)
\]

then correct residual proposal bias when target/proposal support overlaps.
The strength and cost of that correction are reported together: proposal and
weighted distribution errors, ESS/N, maximum weight, top-weight mass, and
log-weight spread must appear in the same benchmark record.

The release does **not** claim that reweighting repairs missing support, that
every BMS proposal is efficient, or that clipping is part of the formal
estimator.

## 2. Included benchmark matrix

| system | arm | endpoint/training-data identity | v0.1 disposition |
|---|---|---|---|
| MB CG1D | cold Energy-BMS + PF reweighting | endpoint-independent CG-BG 1D PMF | frozen canonical result |
| analytic MB2D | cold Energy-BMS + PF reweighting | endpoint-independent analytic target | frozen canonical result |
| analytic MB2D | equilibrium bridge + PF reweighting | `equilibrium_exact_v1` | frozen canonical three-seed result |
| analytic MB2D | equilibrium bridge-to-Energy + PF reweighting | `equilibrium_exact_v1`, then fresh Energy-BMS | frozen canonical three-seed result |
| analytic MB2D | biased bridge + PF reweighting | deliberately biased full-support synthetic mixture | frozen historical stress test |
| analytic MB2D | biased bridge-to-Energy + PF reweighting | same biased mixture, then fresh Energy-BMS | frozen historical stress test |
| six-bead CG Ala2 | selected bridge/warm and bridge-to-Energy paths + PF reweighting | hash-pinned CG endpoint/PMF assets | frozen positive and failure-case evidence with explicit ESS caveats |

The historical MB2D endpoint mixture is not an MD trajectory and is not an
equilibrium dataset. It is preserved because it is a useful, controlled test
of reweighting under known proposal bias.

## 3. Explicit exclusion

Generic experimental source modules may remain for development continuity.
Full-atom experiment configs, datasets, run scripts, result documents,
figures, checkpoints, and benchmark assets are omitted from the publishable
tree. **All full-atom Ala2 runs are excluded from the v0.1 benchmark, frozen
artifact bundle, supported workflows, and scientific claims.**

## 4. Formal weighting and diagnostic views

### 4.1 Formal result

The formal estimator uses every finite, in-support PF sample and normalizes
the original `logw_raw` values without clipping. Formal tables and benchmark
claims must use this no-clip result.

Every formal row must report at least:

- unweighted and weighted distribution error;
- ESS/N;
- maximum normalized weight;
- top 0.1% and top 1% weight mass when available;
- variance or span of `logw_raw`;
- PF validity/support fraction;
- sample count and number of independent seeds.

### 4.2 Clip diagnostic

A clip/drop-top-1% view may be saved beside the formal result to expose
weight-tail sensitivity. It must be labelled `diagnostic_clip1` or equivalent.
It is never substituted for the no-clip estimator and cannot be used as the
headline result.

### 4.3 Display-only energy window

Extreme proposal energies can make a histogram unreadable. Energy figures may
therefore use a target-anchored display interval, such as the exact/reference
0.5%--99.5% energy quantiles plus padding.

This operation is visual only:

- all samples still enter formal metrics and weight normalization;
- visible histograms must not silently renormalize away out-of-window mass;
- proposal and weighted mass below/above the displayed interval must be
  written into the figure metadata or companion JSON;
- the caption must say `display window; all-sample metrics`.

## 5. Frozen result records

### 5.1 MB CG1D

Canonical local artifact root:

```text
artifacts/mb_cg1d_three_seed_20260724/
```

The verified three-seed table and checkpoint hashes are recorded in
`artifacts/MB_MATRIX_RESULTS_20260724.md`. Required release files are:

- per-seed formal no-clip metrics;
- per-seed clip1 diagnostics;
- three-seed summary JSON;
- separately saved density and free-energy panels;
- the experiment config and forward/backward/PF provenance.

### 5.2 Analytic MB2D cold Energy-BMS

This is the endpoint-independent canonical baseline. Its result remains valid
when the old synthetic bridge endpoint identity is corrected because this arm
never reads endpoint data.

Canonical frozen source:

```text
artifacts/mb2d_robust_300k_20260724/cold_energy.metrics.json
```

The publication bundle must save each panel separately:

- exact target density;
- unweighted proposal density;
- no-clip reweighted density;
- x marginal;
- y marginal;
- energy distribution with the display-only robust x-axis;
- weight/ESS diagnostic.

### 5.3 Analytic MB2D equilibrium bridge arms

Dataset:

```text
data/mb2d_equilibrium_exact_v1/endpoints.npz
data/mb2d_equilibrium_exact_v1/endpoints.manifest.json
```

Dataset SHA-256:

```text
f4c43d8feff619102e5cc1f8ad2dbd422d9a201453d356d4f59e48cb9032347d
```

The verified per-seed and aggregate values for both equilibrium arms are stored
under the `mb2d_analytic/*` groups in
[`summary.json`](../artifacts/release_v0.1.0/summary.json) and
[`per_seed.json`](../artifacts/release_v0.1.0/per_seed.json). The records
include proposal and weighted errors, ESS/N, maximum weight, tail masses,
valid/support fractions, and seed count.

Required release files for each arm:

- three per-seed formal metrics and checkpoint/PF identities;
- mean/std or bootstrap aggregation;
- separately saved exact/proposal/reweighted density panels;
- separately saved x marginal, y marginal, energy, and weight panels;
- pooled plots explicitly labelled as equal-seed visualizations rather than
  the formal statistical estimate.

### 5.4 Analytic MB2D biased-endpoint stress tests

Canonical local artifact root:

```text
artifacts/mb2d_biased_endpoint_stress_test_20260724/
```

These results retain the old numeric content and are labelled:

- `biased_bridge_stress_test`;
- `biased_bridge_to_energy_stress_test`.

They support the narrower conclusion that exact-density reweighting can
substantially correct a deliberately biased, full-support proposal. Their low
ESS/heavy-tail diagnostics must remain next to the improved weighted
distribution. They do not support an equilibrium endpoint-learning claim.

### 5.5 Six-bead CG Ala2

Canonical remote source root:

```text
/ds/project/weilong/ke/cg-bms-jax/outputs/
  ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3/
```

The release bundle must distinguish:

- proposal-quality comparisons (cold, warm/bridge, and bridge-to-Energy where
  the corresponding verified artifacts exist);
- formal PF/no-clip reweighting results;
- clipped diagnostics;
- failure cases with low ESS, concentrated top-weight mass, or a weighted
  metric worse than its unweighted counterpart.

CG Ala2 is positive evidence for the reweighting mechanism only where the
weighted structural/energy observable improves and the matching ESS/tail
statistics are reported. It is not a blanket claim that every Ala2 run
improves. Cold-start failure, poor proposal coverage, and low-ESS PF results
must be retained rather than filtered from the release history.

The checksum-verified CG Ala2 values are stored under the `ala2_cg/*` groups in
[`summary.json`](../artifacts/release_v0.1.0/summary.json) and
[`per_seed.json`](../artifacts/release_v0.1.0/per_seed.json). Each arm retains
its release role and proposal/weighted metrics beside ESS and weight-tail
diagnostics, including outcomes where reweighting does not improve every
observable.

Required separate image files:

- unweighted Ramachandran density/FES;
- no-clip weighted Ramachandran density/FES;
- phi marginal;
- psi marginal;
- robust-window energy distribution;
- ESS/weight-tail diagnostic;
- proposal-only warm-versus-Energy comparison where available.

## 6. Parameter and provenance bundle

Every frozen arm must include:

- fully composed Hydra config;
- training mode and initialization identity;
- model dimensions and parameter count;
- source distribution and SDE parameters;
- forward/backward update counts, batch sizes, optimizer, LR schedule,
  damping, clipping, and checkpoint cadence;
- PF solver, tolerances, initial step, maximum steps, divergence mode, sample
  count, and random seed;
- target/PMF/dataset/checkpoint SHA-256 values;
- formal density mode and support convention;
- evaluation binning, bootstrap count, and display-window rule;
- software revision and environment lock.

Metrics without this provenance are not release benchmarks.

## 7. Artifact layout

The frozen bundle uses:

```text
artifacts/release_v0.1.0/
|-- MANIFEST.json
|-- README.md
|-- summary.{csv,json}
|-- per_seed.{csv,json}
|-- SOURCE_RESULTS_INDEX.{csv,json}
|-- SHA256SUMS
|-- configs/
|-- metrics/
|   `-- <system>/<arm>/<seed-or-summary>.json
|-- provenance/
`-- figures/
    `-- <system>/<arm>/<seed-or-pooled>/<view>/<individual-panel>.{png,pdf}
```

Composite figures may be included as previews, but every constituent panel
must also be saved independently for later paper assembly.

`MANIFEST.json` should record the SHA-256 and semantic role of every file. It
must also record that the all-atom Ala2 benchmark is excluded.

## 8. Release verification

The release is publishable only when both commands pass:

```bash
python scripts/validate_release_scope.py artifacts/release_v0.1.0
python scripts/validate_publication_tree.py --phase final --mode staged
```

These checks verify the manifest hashes, complete standalone panels and
machine-readable plot data, formal/diagnostic separation, robust energy-view
metadata, frozen configs and provenance, and the explicit absence of
full-atom Ala2 benchmark materials.
