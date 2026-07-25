# v0.1 benchmark and frozen-result contract

This document defines the scientific claim boundary and artifact contract for
the first public `cg-bms-jax` release. It intentionally separates:

1. formal, unclipped importance-sampling results;
2. optional clipped diagnostics;
3. display-only plot windows;
4. historical stress tests; and
5. experimental code that is shipped but not benchmarked.

No field marked `PENDING_VERIFIED_AGGREGATION` or `PENDING_VERIFIED_FREEZE` is
a result. Those fields must be replaced only after the corresponding
checkpoints, PF archives, metrics, and seed aggregation have passed provenance
verification.

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
| analytic MB2D | equilibrium bridge + PF reweighting | `equilibrium_exact_v1` | canonical; final three-seed aggregation pending |
| analytic MB2D | equilibrium bridge-to-Energy + PF reweighting | `equilibrium_exact_v1`, then fresh Energy-BMS | canonical; final three-seed aggregation pending |
| analytic MB2D | biased bridge + PF reweighting | deliberately biased full-support synthetic mixture | frozen historical stress test |
| analytic MB2D | biased bridge-to-Energy + PF reweighting | same biased mixture, then fresh Energy-BMS | frozen historical stress test |
| six-bead CG Ala2 | selected bridge/warm and bridge-to-Energy paths + PF reweighting | hash-pinned CG endpoint/PMF assets | frozen positive and failure-case evidence with explicit ESS caveats |

The historical MB2D endpoint mixture is not an MD trajectory and is not an
equilibrium dataset. It is preserved because it is a useful, controlled test
of reweighting under known proposal bias.

## 3. Explicit exclusion

The 22-atom OpenMM Ala2 implementation, configs, and smoke scripts remain in
the repository. **All full-atom Ala2 runs are excluded from the v0.1 benchmark,
release artifact bundle, and scientific claims.** They are unfinished
development material and should not appear in the release results table.

This exclusion does not require deleting the implementation.

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

The following values remain intentionally unset until all three seeds pass
checkpoint/PF provenance validation and final aggregation:

| arm | unweighted error | weighted error | ESS/N | max weight | seed count |
|---|---|---|---|---|---|
| equilibrium bridge | `PENDING_VERIFIED_AGGREGATION` | `PENDING_VERIFIED_AGGREGATION` | `PENDING_VERIFIED_AGGREGATION` | `PENDING_VERIFIED_AGGREGATION` | `PENDING_VERIFIED_AGGREGATION` |
| equilibrium bridge-to-Energy | `PENDING_VERIFIED_AGGREGATION` | `PENDING_VERIFIED_AGGREGATION` | `PENDING_VERIFIED_AGGREGATION` | `PENDING_VERIFIED_AGGREGATION` | `PENDING_VERIFIED_AGGREGATION` |

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

The final numerical Ala2 benchmark table is populated only from locally
frozen, checksum-verified metrics:

| frozen arm | proposal metric | weighted metric | ESS/N | max/top mass | status |
|---|---|---|---|---|---|
| warm/bridge | `PENDING_VERIFIED_FREEZE` | `PENDING_VERIFIED_FREEZE` | `PENDING_VERIFIED_FREEZE` | `PENDING_VERIFIED_FREEZE` | positive or failure after audit |
| warm/bridge-to-Energy | `PENDING_VERIFIED_FREEZE` | `PENDING_VERIFIED_FREEZE` | `PENDING_VERIFIED_FREEZE` | `PENDING_VERIFIED_FREEZE` | positive or failure after audit |
| selected cold failure | `PENDING_VERIFIED_FREEZE` | `PENDING_VERIFIED_FREEZE` | `PENDING_VERIFIED_FREEZE` | `PENDING_VERIFIED_FREEZE` | failure case |

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

The final frozen bundle should use:

```text
artifacts/release_v0.1.0/
|-- MANIFEST.json
|-- BENCHMARKS.md
|-- parameters/
|   |-- mb_cg1d/
|   |-- mb2d_cold_energy/
|   |-- mb2d_equilibrium_bridge/
|   |-- mb2d_equilibrium_bridge_energy/
|   |-- mb2d_biased_bridge/
|   |-- mb2d_biased_bridge_energy/
|   `-- ala2_cg/
|-- metrics/
|   `-- <system>/<arm>/<seed-or-summary>.json
`-- figures/
    `-- <system>/<arm>/<individual-panel>.{png,pdf}
```

Composite figures may be included as previews, but every constituent panel
must also be saved independently for later paper assembly.

`MANIFEST.json` should record the SHA-256 and semantic role of every file. It
must also record that the all-atom Ala2 benchmark is excluded.

## 8. Release gate

- [ ] MB CG1D frozen files copied and checksummed.
- [ ] MB2D cold Energy-BMS frozen files copied and checksummed.
- [ ] Equilibrium MB2D bridge three-seed aggregation verified.
- [ ] Equilibrium MB2D bridge-to-Energy three-seed aggregation verified.
- [ ] Legacy biased synthetic stress tests relabelled and checksummed.
- [ ] CG Ala2 positive and failure cases audited and copied locally.
- [ ] Every composite plot has separately saved panels.
- [ ] Energy distributions use a documented display-only robust window.
- [ ] Formal no-clip and diagnostic clip1 outputs are visibly separated.
- [ ] Parameter/provenance bundle is complete.
- [ ] All-atom Ala2 is absent from benchmark tables and release artifacts.
- [ ] Artifact manifest hashes pass.
- [ ] Release tag is created only after every non-pending benchmark field is
      backed by a frozen file.
