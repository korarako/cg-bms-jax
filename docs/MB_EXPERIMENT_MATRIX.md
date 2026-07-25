# Pure-JAX MB experiment matrix

## 1. Repository boundary

This matrix belongs to **`cg-bms-jax`**, the pure-JAX implementation.
The older `cg-bms` repository combines Torch BMS code with a JAX PMF worker and
is not modified or imported here.

This document is the detailed MB protocol. The authoritative public-release
scope, including six-bead CG Ala2 and the explicit exclusion of all-atom Ala2,
is [`RELEASE_RESULTS_V0.1.0.md`](RELEASE_RESULTS_V0.1.0.md).

The matrix separates two scientific systems:

1. `mb_cg1d`: the released CG-BG one-dimensional PMF experiment. It is the
   regression test for the already successful full pipeline.
2. `mb2d_analytic`: the original two-dimensional Muller--Brown energy. It is
   asset-free and tests energy-only BMS, density calculation, and reweighting
   without a learned PMF or molecular geometry.

No result from `mb_cg1d` is relabelled as a two-dimensional result.

### 1.1 MB2D data identity and release status

There are now two deliberately separate endpoint-data identities:

| identity | endpoint law | role in the release |
|---|---|---|
| `biased_endpoint_stress_test` | the fixed full-support Gaussian mixture in `mb2d_analytic.yaml` | frozen legacy stress test; it tests whether importance reweighting can correct a deliberately biased proposal |
| `equilibrium_exact_v1` | samples from the normalized analytic MB2D Boltzmann density at `beta=1` | canonical data-assisted bridge and bridge-to-Energy matrix |

The existing output root
`outputs/mb2d_formal_matrix_20260724/` belongs to the first identity. It must
not be described as an equilibrium-data, MD-data, or reference-data bridge
experiment. Its cold Energy-BMS arm is independent of endpoint data and
remains a canonical data-free result. Its bridge-only and bridge-to-Energy
arms are retained, unchanged, as biased-endpoint stress tests.

The new equilibrium-data arms use a separate config, checkpoint identity, and
output root. Old and new checkpoints are never mixed. The complete endpoint
contract is specified in
[`MB2D_ENDPOINT_DATASETS.md`](MB2D_ENDPOINT_DATASETS.md).

## 2. Analytic MB2D contract

The controller and PF-ODE use standardized coordinates

\[
z\in\mathbb R^2,\qquad x=(25,25)+(10,10)\odot z.
\]

The source is exactly \(q_0(z)=\mathcal N(0,I_2)\). The formal target is

\[
\pi_\beta(z)\propto
\exp[-\beta U_{\rm MB}(x(z))]
\mathbf 1\{x(z)\in[0,50]^2\}.
\]

`kT` and `beta` are not applied twice: the runtime enforces
`kT == 1 / target.beta`. The affine determinant is constant and therefore
cancels in normalized importance weights, while PF `logq` is consistently
defined in `z`.

Cold rollouts can leave the formal box. Training therefore uses:

- a bounded C2 coordinate continuation outside the box, so energy and score
  remain finite;
- a flat-bottom restoring confinement term near/outside the boundary.

Both are recorded in the **training-target** signature. The confinement is
excluded from formal `U`, `logw`, and the analytic reference grid.

## 3. Legacy biased-endpoint stress test

`cg-bms-pretrain-flat-bridge` builds a reproducible endpoint bank from a
non-degenerate diagonal Gaussian mixture. Its default physical-space weights
are deliberately biased:

| component | weight | mean | scale |
|---|---:|---:|---:|
| right basin | 0.65 | `[48, 8]` | `[2.5, 2.5]` |
| central basin | 0.20 | `[32, 16]` | `[2.5, 2.5]` |
| upper basin | 0.13 | `[24, 32]` | `[2.5, 2.5]` |
| broad tail | 0.02 | `[25, 25]` | `[15, 15]` |

Every scale and weight is strictly positive, so the endpoint law has full
support on \(\mathbb R^2\). Its complete config and seed are canonically
hashed. That digest is stored as both `training_data_sha256` and
`warmstart_data_sha256`; an incompatible bridge checkpoint cannot initialize
an Energy-BMS run.

The completed 2026-07-24 matrix contains:

| release label | historical arm | forward controller | purpose |
|---|---|---|---|
| Biased bridge stress test | Bridge only | supervised biased bridge checkpoint | Does reweighting remove a known, intentionally constructed proposal bias? |
| Cold Energy-BMS | Cold Energy-BMS | Energy-BMS from `N(0,I2)` | Can data-free BMS learn the analytic target? This arm does not use the endpoint bank. |
| Biased bridge-to-Energy stress test | Bridge to Energy-BMS | biased bridge parameters, then a fresh Energy-BMS fixed-point run | Does energy refinement improve proposal quality or weight stability after biased initialization? |

Bridge initialization is parameters-only. Optimizer state, replay, RNG,
schedule, and Energy-BMS step count all restart.

These stress-test results are valid evidence about importance correction under
known proposal bias. They are not evidence that bridge matching learned an
equilibrium target from correct Boltzmann endpoints.

### 3.1 Canonical equilibrium bridge arms

The canonical data-assisted comparison replaces the synthetic mixture with a
versioned endpoint archive sampled from the same normalized analytic target
used by evaluation:

\[
p_{\rm endpoint}(x)=
\frac{\exp[-U_{\rm MB}(x)]\mathbf 1\{x\in[0,50]^2\}}
{\int_{[0,50]^2}\exp[-U_{\rm MB}(x')]\,dx'}.
\]

It adds two new arms:

| release label | initialization | Energy-BMS |
|---|---|---|
| Equilibrium bridge | `equilibrium_exact_v1` train split | none |
| Equilibrium bridge-to-Energy | same bridge checkpoint | fresh 100k-update Energy-BMS run |

The previously completed cold Energy-BMS arm is reused as the matched
data-free baseline because its training path never reads `bridge_data`.
All three arms receive independent matched backward training, PF-ODE
likelihood sampling, formal no-clip reweighting, and exact-grid evaluation.

## 4. Density and reweighting

For a matched forward/backward pair,

\[
f_{\rm PF}(t,z)=\tfrac12g(t)^2[u(t,z)-v(t,z)].
\]

Diffrax Dopri5 integrates coordinates and log density in the same state:

\[
\dot z=f_{\rm PF}(t,z),\qquad
\frac{d\log q_t}{dt}=-\nabla_z\cdot f_{\rm PF}(t,z).
\]

Formal weights are never clipped:

\[
\log w_{\rm raw}=-\beta U_{\rm MB}(x(z_1))-\log q_1(z_1),
\qquad
w_i=\operatorname{softmax}_i(\log w_{\rm raw}).
\]

Stored formal diagnostics include ESS/N, maximum normalized weight, top
0.1%/1% mass, log-weight variance/span, 2D JS, PMF RMSE, energy error, and
three-basin mass error.

## 5. Required matrix

### A. CG1D regression

- three independent forward/backward/PF seeds;
- 100k forward updates, 100k backward updates, 100k PF samples;
- formal no-clip metrics and mean/std across seeds;
- a separate CG-BG compatibility diagnostic that drops the highest 1% of raw
  log weights (`compat_clip_percentile=99.0`, `clip_mode=drop`).

The no-clip estimator remains the formal result. The clip1% view is reported
alongside it to expose weight-tail sensitivity, not to replace it.

Run:

```bash
bash scripts/run_mb_cg1d_three_seed.sh
```

On a host with two free GPUs, the race-safe parallel launcher assigns seeds
`0,2` to GPU 0 and seed `1` to GPU 2, then writes one summary only after both
workers succeed:

```bash
RUN_ROOT=outputs/mb_cg1d_three_seed_run1 \
GPU_A=0 GPU_B=2 \
bash scripts/run_mb_cg1d_three_seed_parallel.sh
```

### B. Analytic MB2D completed legacy matrix

- three independent controller seeds;
- 100k biased-bridge updates;
- cold and bridge-initialized Energy-BMS, each 100k updates;
- checkpoint every 5 outer iterations;
- matched 100k backward controller per arm;
- 100k PF samples per arm;
- proposal, reweighted, energy, marginal, basin, and weight diagnostics.

Run:

```bash
bash scripts/run_mb2d_formal_matrix.sh
```

Set `SEEDS=0` and `NUM_SAMPLES=20000` for a one-seed pilot. This is still a
formal no-clip run; it merely has lower statistical power.

This command reproduces the frozen `biased_endpoint_stress_test` matrix. It is
not the canonical equilibrium endpoint experiment.

### B.1 Analytic MB2D equilibrium endpoint matrix

Generate and validate the versioned exact endpoint archive first. The
generator must record the target signature, grid resolution, RNG seed,
coordinate transform, split sizes, file SHA-256, and empirical-versus-exact
distribution diagnostics. It must never label the archive as MD.

The v1 archive is already frozen in-tree. To create a future version, change
the output path and run:

```bash
cg-bms-generate-mb2d-equilibrium \
  output=data/<new-version>/endpoints.npz
```

The canonical matrix then runs:

- equilibrium bridge: 100k forward updates per seed;
- equilibrium bridge-to-Energy: the same bridge initializer followed by a
  fresh 100k-update Energy-BMS run per seed;
- 100k matched-backward updates and 100k PF samples for both arms;
- seeds `0, 1, 2`, formal no-clip weights, and 500 bootstrap replicates;
- the already completed cold Energy-BMS results as the data-free baseline.

The new result root is:

```text
outputs/mb2d_equilibrium_bridge_matrix_v1/
```

and the intended launcher is:

```bash
bash scripts/run_mb2d_equilibrium_bridge_matrix.sh
```

The equilibrium endpoint dataset and checkpoint digests must differ from the
legacy synthetic-mixture digests. A run is invalid if an old bridge
checkpoint is accepted under the equilibrium config.

### B.2 Robust all-sample visualization

Formal metrics always use every sample and the unmodified importance weights.
For readability only, the energy subplot uses the analytic target's
probability-weighted 0.5%--99.5% energy quantiles (plus 5% padding) as its
display range. Proposal and reweighted mass below, above, or non-finite with
respect to that range is reported explicitly; it is not discarded or
renormalized into the visible range.

One formal archive can be redrawn with:

```bash
cg-bms-mb2d-suite evaluate \
  --sample outputs/mb2d_formal_matrix/seed_0/cold_energy_pf100000.npz \
  --output-dir outputs/mb2d_formal_matrix/seed_0/cold_energy_evaluation_robust \
  --bins 160 --energy-bins 200
```

The corresponding drop-top-1% diagnostic is explicit:

```bash
cg-bms-mb2d-suite evaluate \
  --sample outputs/mb2d_formal_matrix/seed_0/cold_energy_pf100000.npz \
  --output-dir outputs/mb2d_formal_matrix/seed_0/cold_energy_clip1 \
  --bins 160 --energy-bins 200 \
  --clip-percentile 99 --clip-mode drop
```

For a higher-statistics figure, `pool` concatenates all 100k samples from each
of three seeds (300k samples per arm). Each seed's formal weights are
normalized independently and assigned equal total mass. This is an
equal-seed stratified visualization; the per-seed metrics remain the
authoritative formal estimates.

```bash
cg-bms-mb2d-suite pool \
  --archive seed0=outputs/mb2d_formal_matrix/seed_0/cold_energy_pf100000.npz \
  --archive seed1=outputs/mb2d_formal_matrix/seed_1/cold_energy_pf100000.npz \
  --archive seed2=outputs/mb2d_formal_matrix/seed_2/cold_energy_pf100000.npz \
  --output-dir outputs/mb2d_formal_matrix/pooled_300k/cold_energy \
  --bins 160 --energy-bins 240
```

For the pooled clip diagnostic, add
`--clip-percentile 99 --clip-mode drop`. Clipping is performed independently
inside each seed before assigning equal total mass to the three seeds.

### C. Forward-checkpoint sweep

Hold the backward budget, PF seed, sample count, and tolerance fixed. Compare
forward updates 10k, 25k, 50k, 75k, and 100k.

### D. Backward-budget sweep

Freeze one final forward checkpoint and compare backward 10k, 25k, 50k, and
100k. `train_backward.updates` is a root-level estimator-budget override, so
the experiment identity and forward checkpoint hash remain unchanged.

### E. Sample-size convergence

Use nested random prefixes of one 100k archive:

```text
1k, 2k, 5k, 10k, 20k, 50k, 100k
```

Repeat with five independent permutations and report mean/std. No prefix is
clipped.

### F. SDE versus PF terminal law

`cg-bms-mb2d-suite dynamics` compares unweighted forward-SDE and PF terminal
distributions through means, covariances, 2D histogram JS, and energy
histograms. This checks whether the learned backward field gives a PF whose
terminal law is consistent with the stochastic forward proposal.

### G. Integrated logq versus direct Jacobian

`cg-bms-validate-pf` is a small-batch dense audit. It differentiates the
coordinate-only flow map with `jax.jacrev` and checks

\[
\log q_1^{\rm direct}
=\log q_0-\log|\det(\partial z_1/\partial z_0)|
\]

against the integrated divergence result. It is deliberately limited to at
most four dimensions and is not a production sampler.

### H. ODE-tolerance sweep

Using bitwise-identical initial samples, compare:

| label | rtol | atol |
|---|---:|---:|
| loose | `1e-4` | `1e-5` |
| formal | `1e-5` | `1e-6` |
| strict | `1e-7` | `1e-8` |

Report terminal-coordinate RMSE, logq RMSE/max error, and weight total
variation against the strict archive.

### I. Optional temperature reweighting

Reuse a fixed proposal and evaluate beta
`0.5, 0.75, 1.0, 1.25, 1.5`. The analytic reference is recomputed for every
beta. This is a support/overlap stress test, not additional training.

The forward-checkpoint, backward-budget, and tolerance sweeps are launched by:

```bash
bash scripts/run_mb2d_controlled_sweeps.sh
```

## 6. Smoke sequence

Before a formal matrix:

```bash
cg-bms-pretrain-flat-bridge \
  experiment=mb2d_analytic_smoke \
  pretrain.updates=8 \
  pretrain.batch_size=32 \
  checkpoint_every=null \
  output_dir=outputs/mb2d_smoke/bridge

cg-bms-train-forward \
  experiment=mb2d_analytic_smoke \
  initialize_controller_from=outputs/mb2d_smoke/bridge/forward_pretrain_step_00000008 \
  output_dir=outputs/mb2d_smoke/forward

cg-bms-train-backward \
  experiment=mb2d_analytic_smoke \
  forward_checkpoint=outputs/mb2d_smoke/forward/forward_step_00000008 \
  updates=8 \
  output_dir=outputs/mb2d_smoke/backward

cg-bms-sample-reweight \
  experiment=mb2d_analytic_smoke \
  forward_checkpoint=outputs/mb2d_smoke/forward/forward_step_00000008 \
  backward_checkpoint=outputs/mb2d_smoke/backward/backward_step_00000008 \
  num_samples=256 batch_size=64 target_batch_size=64 \
  progress_every_batches=1 \
  output=outputs/mb2d_smoke/samples_and_weights.npz

cg-bms-evaluate \
  experiment=mb2d_analytic_smoke \
  samples=outputs/mb2d_smoke/samples_and_weights.npz \
  output_dir=outputs/mb2d_smoke/evaluation \
  bootstrap=false compat_clip_percentile=null
```

Passing a smoke run confirms software connectivity only. A scientific claim
requires the seed matrix, convergence checks, and the no-clip metrics above.

## 7. Interpretation rules

- Reweighting is useful when formal weighted errors improve consistently and
  ESS/N is not dominated by a few samples.
- The old MB2D bridge results answer a stress-test question: whether
  reweighting corrects an intentionally biased full-support proposal. They do
  not answer whether equilibrium endpoint training is accurate.
- The equilibrium bridge matrix is the only data-assisted MB2D matrix used
  for an equilibrium-learning claim. The cold Energy-BMS arm remains valid
  in both comparisons because it is data-free.
- Larger sample count is not expected to make every finite-sample curve
  monotone. Convergence should be judged across repeated nested prefixes.
- A good weighted result with a poor proposal still demonstrates a correct
  importance correction, but it does not establish a practical sampler.
- A good proposal with unstable weights points first to backward/logq quality;
  use the Jacobian and tolerance audits before changing forward dynamics.
- Clipped plots may be generated separately for diagnosis, but they cannot
  replace formal no-clip results.
