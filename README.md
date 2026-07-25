# cg-bms-jax

`cg-bms-jax` is a pure-JAX Bridge Matching Sampler (BMS) implementation with
exact-density PF-ODE reweighting for Boltzmann targets. The mixed Torch/JAX
`cg-bms` prototype is a numerical reference only; it is not a runtime
dependency.

The v0.1 benchmark/release scope contains three experiment families:

- `mb_cg1d`: the one-dimensional coarse variable of the Müller–Brown system,
  with an MLP controller;
- `mb2d_analytic`: an asset-free, analytic two-dimensional Muller--Brown
  target in a standardized `N(0,I_2)` state space, with an MLP controller and
  an explicit finite-box formal support;
- `ala2_ambient18_300k`: the six-bead Ala2 core-beta mapping in the complete
  CG-BG-style 18D auxiliary-COM space at 300 K, with the faithful JAX port of
  the BMS PaiNN controller and the CG-BG MACE PMF.

The repository retains some experimental 22-atom Ala2/OpenMM implementation
code for development continuity. It is **not** a supported v0.1 workflow and
no all-atom config, result, figure, or claim is included in the frozen release.

The formal density path is:

```text
frozen PMF -> forward BMS u -> backward matching v
           -> coupled PF-ODE state + exact log q
           -> PMF importance weights -> CG-BG-style evaluation
```

The forward SDE sampler is also exposed for unweighted diagnostics. Its samples
do not carry `logq` and the reweighting API deliberately refuses to treat them
as likelihood samples.

The complete MB replication and validation matrix is documented in
[`docs/MB_EXPERIMENT_MATRIX.md`](docs/MB_EXPERIMENT_MATRIX.md). All new MB2D
code is part of this pure-JAX repository; the older mixed Torch/JAX `cg-bms`
checkout remains an unchanged numerical reference.

The exact v0.1 result boundary, artifact layout, reporting rules, and
still-pending aggregation fields are recorded in
[`docs/RELEASE_RESULTS_V0.1.0.md`](docs/RELEASE_RESULTS_V0.1.0.md). A value
marked `PENDING_VERIFIED_AGGREGATION` is deliberately not a release number.

MB2D bridge results are data-versioned. The completed 2026-07-24 bridge and
bridge-to-Energy runs used a deliberately biased, full-support synthetic
endpoint mixture and are retained as stress tests of importance correction.
They are not labelled as equilibrium or MD endpoint training. The cold
Energy-BMS arm is endpoint-independent and remains a canonical data-free
result. Canonical data-assisted claims use the separately hashed
`equilibrium_exact_v1` endpoint archive and rerun only the two
endpoint-dependent arms. See
[`docs/MB2D_ENDPOINT_DATASETS.md`](docs/MB2D_ENDPOINT_DATASETS.md).

## Optional Ala2 bridge warm start

Cold energy-only Ala2 training can spend its early fixed-point iterations far
outside the molecular manifold.  The optional warm-start command learns only
an initial controller from the hash-pinned CG-BG `flow_b/data.npz` training
split, using the variance-weighted supervised bridge objective released in
WT-ASBS.  It never reads `flow_ub`, the implicit reference trajectory, or PMF
energies.

```bash
cg-bms-pretrain-bridge \
  experiment=ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3
```

The production defaults run 100000 updates with batch size 64 and save every
10000 updates.  Each data endpoint is geometrically centered, divided by the
exact CG-BG training-set standard deviation, Haar-SO(3) rotated, and augmented
with the shared `N(0,I/6)` COM variable.  The paired endpoint is independently
drawn from the same full-rank `N(0,I_18)` source used by energy BMS.

Warm checkpoints can be inspected before energy refinement with the explicitly
diagnostic sampler mode:

```bash
cg-bms-sample-sde \
  experiment=ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3 \
  controller_kind=forward_pretrain \
  forward_checkpoint=outputs/ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3/warmstart/forward_pretrain_step_00100000 \
  num_samples=20000 batch_size=64 \
  output=outputs/ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3/warmstart/proposal_20k.npz
```

This writes `sampler_kind=sde_pretrain`, contains no likelihood or weights,
and is accepted only by proposal/evaluation diagnostics.

Use the final warm checkpoint only as controller initialization:

```bash
cg-bms-train-forward \
  experiment=ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3 \
  initialize_controller_from=outputs/ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3/warmstart/forward_pretrain_step_00100000 \
  experiment.training.outer_iterations=20 \
  experiment.training.gradient_steps=1000 \
  output_dir=outputs/ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3/warm_energy_20k/forward
```

`initialize_controller_from` restores only controller parameters and immutable
Flax constants.  The optimizer, learning-rate schedule, replay buffer, random
stream, fixed-point iteration, and global step are rebuilt from scratch.  The
resulting energy checkpoint records the initializer SHA-256 and remains an
ordinary forward checkpoint for SDE diagnostics and later backward/PF-ODE
training.  This is deliberately different from the unimplemented `resume`
operation.

## Reproducibility contract

- BMS/PaiNN reference revision:
  `d19a27b854fd77387c43109e37b597bc35252e7d`.
- CG-BG PMF/evaluation reference revision:
  `948aaeff8a6b25de38b6e7b1112041c1cfd40573`.
- WT-ASBS warm-start reference revision:
  `3177ec826b1111a8ee47ee3b9d1683a3a55917c6`.
- ChemTrain revision: `a97ca2dd60c8327f574f269d02ec5edbccbae6b8`.
- CGPeptides asset revision:
  `39765bbcfee382e5f30445589d7fe28ebb6cfff8`.
- Runtime source contains no Torch import and needs only one CUDA GPU.
- Formal CG Ala2 reweighting uses the normalized auxiliary-COM target in
  ambient 18D. The historical CG-BG radial COM correction is available only
  for the CG diagnostic `cgbg_compat` mode.
- Experimental all-atom support is retained for development continuity but is
  excluded from the v0.1 benchmark and scientific claims.

Every checkpoint records the experiment digest, PMF revision and hash, mapping,
units, density convention, model signature, and SDE signature. Backward and
forward checkpoints must have matching metadata and exact parentage before a
PF-ODE can be assembled.

## Installation

The reference environment is Python 3.11 at
`/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax`. From a source checkout:

```bash
cd /ds/project/weilong/ke/cg-bms-jax
bash scripts/create_env.sh
source /ds/project/weilong/ke/miniconda3/bin/activate cg-bms-jax
python scripts/fetch_assets.py --all
python scripts/check_environment.py
pytest -q
```

`create_env.sh` is the authoritative Ala2 installation route. It installs the
pinned ChemTrain and ChemUtils sources without allowing their metadata to
downgrade the validated JAX 0.4.38 stack. A normal wheel contains the Python
package, Hydra configs, asset manifest, provenance, and license notices; it does
not contain downloaded PMF checkpoints or trajectories.

All relative asset and output paths resolve against the source checkout. When
running an installed wheel elsewhere, they resolve against the current working
directory unless `CG_BMS_PROJECT_ROOT` is set.

## Five-stage workflow

The commands below run the small Müller–Brown smoke configuration and make all
checkpoint paths explicit. Run them in a fresh output directory. To select a
GPU, set `CUDA_VISIBLE_DEVICES` before the first command.

```bash
cd /ds/project/weilong/ke/cg-bms-jax
source /ds/project/weilong/ke/miniconda3/bin/activate cg-bms-jax
export CUDA_VISIBLE_DEVICES=0
```

### 1. Train the forward BMS controller

```bash
cg-bms-train-forward experiment=mb_cg1d_smoke
```

This smoke config makes eight optimizer updates and writes:

```text
outputs/mb_cg1d_smoke/forward/forward_step_00000008
```

For the full configuration, use `experiment=mb_cg1d`; its default final step is
100000. Existing checkpoint destinations are never overwritten.

### 2. Generate an unweighted forward-SDE diagnostic

```bash
cg-bms-sample-sde \
  experiment=mb_cg1d_smoke \
  forward_checkpoint=outputs/mb_cg1d_smoke/forward/forward_step_00000008 \
  num_samples=1024 batch_size=256 \
  output=outputs/mb_cg1d_smoke/proposed_samples.npz
```

The archive has `sampler_kind=sde`, coordinates, PMF energies, and a validity
mask. It intentionally has no `logq`, `logw`, or formal weights.

### 3. Train the backward controller against the frozen forward controller

```bash
cg-bms-train-backward \
  experiment=mb_cg1d_smoke \
  forward_checkpoint=outputs/mb_cg1d_smoke/forward/forward_step_00000008
```

The resulting checkpoint is:

```text
outputs/mb_cg1d_smoke/backward/backward_step_00000008
```

Backward matching supplies the reverse score required to turn the pair of
stochastic BMS drifts into the deterministic probability-flow velocity. It is
not needed for the stage-2 SDE diagnostic; it is required for exact-density
sampling.

### 4. Integrate the PF-ODE, compute `logq`, and reweight

```bash
cg-bms-sample-reweight \
  experiment=mb_cg1d_smoke \
  forward_checkpoint=outputs/mb_cg1d_smoke/forward/forward_step_00000008 \
  backward_checkpoint=outputs/mb_cg1d_smoke/backward/backward_step_00000008 \
  num_samples=1024 batch_size=256 \
  progress_every_batches=10 \
  output=outputs/mb_cg1d_smoke/samples_and_weights.npz
```

Diffrax Dopri5 integrates coordinates and `logq` in the same ODE state using
the exact ambient divergence. The archive contains CG-BG-compatible
`R`, `logp`, `logw`, `weights`, and `U`, plus `logq_ambient`, `logw_raw`,
validity and support masks, solver diagnostics, and provenance. Formal importance weights
are normalized without clipping. Sampling prints the processed batch/sample
count, ODE steps, elapsed time, throughput, and ETA every
`progress_every_batches` batches, as well as on the first and final batch.

### 5. Produce CG-BG-style evaluation

```bash
cg-bms-evaluate \
  experiment=mb_cg1d_smoke \
  samples=outputs/mb_cg1d_smoke/samples_and_weights.npz \
  output_dir=outputs/mb_cg1d_smoke/evaluation \
  bootstrap=false
```

The `formal/` directory contains the unweighted/reweighted density and free
energy plot plus JSON metrics (JS divergence, PMF error, ESS, maximum weight,
and log-weight variance). A separately labelled CG-BG 99th-percentile clipping
view is diagnostic only and does not replace the formal weights.

## Ala2 at 300 K

Fetch the assets and replace `experiment=mb_cg1d_smoke` in all five commands
with either `experiment=ala2_ambient18_300k_smoke` or the production
`experiment=ala2_ambient18_300k`. Use the checkpoint names printed by the two
training commands. Ala2 evaluation adds:

- explicit/implicit/proposal/reweighted energy distributions;
- unweighted and weighted phi/psi marginals and one-dimensional FES;
- explicit, proposal, and reweighted Ramachandran FES;
- distribution, ESS, maximum-weight, log-weight, and energy-Wasserstein
  metrics.

Exact 18D divergence is substantially more expensive than the one-dimensional
Müller–Brown likelihood calculation, so validate with the smoke config before a
production run.

## Configuration and help

Each console command prints its fully composed Hydra config with `--help`.
Override values with normal Hydra syntax, for example:

```bash
cg-bms-sample-reweight --help
cg-bms-train-forward experiment=mb_cg1d seed=7 \
  experiment.training.outer_iterations=10
```

`resume` is reserved but not implemented in the alpha release because a valid
resume must restore optimizer state and scheduling exactly. Use a fresh output
path instead.

## Licensing and citation

The faithful BMS port is distributed under the FAIR Chemistry License and its
incorporated Acceptable Use Policy. CG-BG-derived portions retain their MIT
notice. See `LICENSE`, `LICENSES/`, `THIRD_PARTY_NOTICES.md`, and `UPSTREAM.md`.

Downloaded checkpoints and trajectory files are not committed or included in
the wheel. Before a public release, replace the placeholder repository URL and
add the actual project authors in `CITATION.cff`; those identities cannot be
inferred safely from the source tree.
