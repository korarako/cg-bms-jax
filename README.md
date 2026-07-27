# cg-bms-jax

## Introduction

`cg-bms-jax` is a pure-JAX implementation of Bridge Matching Samplers for
Boltzmann distributions. It provides forward stochastic bridge training,
backward matching, probability-flow ODE likelihoods, and importance
reweighting in one codebase. Controllers include MLP and E(3)-equivariant
PaiNN models; targets include analytic Müller--Brown systems and optional
coarse-grained molecular PMFs.

## Install

Python 3.11 is required.

```bash
git clone https://github.com/korarako/cg-bms-jax.git
cd cg-bms-jax
python -m pip install -e .
```

For CUDA and the optional CG PMF dependencies:

```bash
bash scripts/create_env.sh
conda activate cg-bms-jax
python scripts/fetch_assets.py --all
```

## Quickstart

The asset-free `mb2d_analytic_quickstart` experiment exposes three distinct
training routes. Each stage uses only eight updates, so these commands validate
the workflow rather than reproduce benchmark-scale results.

The Bridge Matching routes use the deterministic full-support Gaussian mixture
declared by `mb2d_analytic`. It is a biased-endpoint workflow example, not an
equilibrium reference dataset. Canonical analytic endpoints can instead be
created with `cg-bms-generate-mb2d-equilibrium`.

### Forward Energy-BMS + backward

Train a cold-start forward Energy-BMS controller, then its backward controller:

```bash
cg-bms-train-forward \
  experiment=mb2d_analytic_quickstart \
  output_dir=outputs/mb2d_analytic_quickstart/energy_bms/forward

cg-bms-sample-sde \
  experiment=mb2d_analytic_quickstart \
  controller_kind=forward \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/energy_bms/forward/forward_step_00000008 \
  num_samples=256 batch_size=64 \
  output=outputs/mb2d_analytic_quickstart/energy_bms/proposal.npz

cg-bms-train-backward \
  experiment=mb2d_analytic_quickstart \
  forward_controller_kind=forward \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/energy_bms/forward/forward_step_00000008 \
  updates=8 \
  output_dir=outputs/mb2d_analytic_quickstart/energy_bms/backward

cg-bms-sample-reweight \
  experiment=mb2d_analytic_quickstart \
  forward_controller_kind=forward \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/energy_bms/forward/forward_step_00000008 \
  backward_checkpoint=outputs/mb2d_analytic_quickstart/energy_bms/backward/backward_step_00000008 \
  num_samples=256 batch_size=64 \
  output=outputs/mb2d_analytic_quickstart/energy_bms/samples_and_weights.npz

cg-bms-evaluate \
  experiment=mb2d_analytic_quickstart \
  samples=outputs/mb2d_analytic_quickstart/energy_bms/samples_and_weights.npz \
  output_dir=outputs/mb2d_analytic_quickstart/energy_bms/evaluation \
  bootstrap=false compat_clip_percentile=null
```

### Bridge Matching + backward

Train the supervised Bridge Matching controller directly from endpoint data,
then fit its backward controller:

```bash
cg-bms-pretrain-flat-bridge \
  experiment=mb2d_analytic_quickstart \
  pretrain.updates=8 \
  pretrain.batch_size=32 \
  pretrain.learning_rate_warmup_updates=0 \
  pretrain.learning_rate_decay_updates=8 \
  checkpoint_every=null \
  save_endpoint_bank=false \
  output_dir=outputs/mb2d_analytic_quickstart/bridge_matching/forward_pretrain

cg-bms-sample-sde \
  experiment=mb2d_analytic_quickstart \
  controller_kind=forward_pretrain \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/bridge_matching/forward_pretrain/forward_pretrain_step_00000008 \
  num_samples=256 batch_size=64 \
  output=outputs/mb2d_analytic_quickstart/bridge_matching/proposal.npz

cg-bms-train-backward \
  experiment=mb2d_analytic_quickstart \
  forward_controller_kind=forward_pretrain \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/bridge_matching/forward_pretrain/forward_pretrain_step_00000008 \
  updates=8 \
  output_dir=outputs/mb2d_analytic_quickstart/bridge_matching/backward

cg-bms-sample-reweight \
  experiment=mb2d_analytic_quickstart \
  forward_controller_kind=forward_pretrain \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/bridge_matching/forward_pretrain/forward_pretrain_step_00000008 \
  backward_checkpoint=outputs/mb2d_analytic_quickstart/bridge_matching/backward/backward_step_00000008 \
  num_samples=256 batch_size=64 \
  output=outputs/mb2d_analytic_quickstart/bridge_matching/samples_and_weights.npz

cg-bms-evaluate \
  experiment=mb2d_analytic_quickstart \
  samples=outputs/mb2d_analytic_quickstart/bridge_matching/samples_and_weights.npz \
  output_dir=outputs/mb2d_analytic_quickstart/bridge_matching/evaluation \
  bootstrap=false compat_clip_percentile=null
```

The checkpoint role in this route is `forward_pretrain`. The explicit
controller-kind options prevent it from being mistaken for an Energy-BMS
checkpoint.

### Bridge Matching + forward Energy-BMS + backward

Initialize a fresh Energy-BMS run from the Bridge Matching controller above,
then train a new backward controller for the resulting forward checkpoint:

```bash
cg-bms-train-forward \
  experiment=mb2d_analytic_quickstart \
  initialize_controller_from=outputs/mb2d_analytic_quickstart/bridge_matching/forward_pretrain/forward_pretrain_step_00000008 \
  output_dir=outputs/mb2d_analytic_quickstart/bridge_energy/forward

cg-bms-sample-sde \
  experiment=mb2d_analytic_quickstart \
  controller_kind=forward \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/bridge_energy/forward/forward_step_00000008 \
  num_samples=256 batch_size=64 \
  output=outputs/mb2d_analytic_quickstart/bridge_energy/proposal.npz

cg-bms-train-backward \
  experiment=mb2d_analytic_quickstart \
  forward_controller_kind=forward \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/bridge_energy/forward/forward_step_00000008 \
  updates=8 \
  output_dir=outputs/mb2d_analytic_quickstart/bridge_energy/backward

cg-bms-sample-reweight \
  experiment=mb2d_analytic_quickstart \
  forward_controller_kind=forward \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/bridge_energy/forward/forward_step_00000008 \
  backward_checkpoint=outputs/mb2d_analytic_quickstart/bridge_energy/backward/backward_step_00000008 \
  num_samples=256 batch_size=64 \
  output=outputs/mb2d_analytic_quickstart/bridge_energy/samples_and_weights.npz

cg-bms-evaluate \
  experiment=mb2d_analytic_quickstart \
  samples=outputs/mb2d_analytic_quickstart/bridge_energy/samples_and_weights.npz \
  output_dir=outputs/mb2d_analytic_quickstart/bridge_energy/evaluation \
  bootstrap=false compat_clip_percentile=null
```

`initialize_controller_from` copies controller parameters and immutable
constants only. Energy-BMS starts with a fresh optimizer, replay buffer, random
stream, and step counter. SDE proposal archives are unweighted diagnostics;
`logq` and importance weights are produced by the paired forward/backward
PF-ODE command.

## Cite

Please cite the software metadata in [CITATION.cff](CITATION.cff), together
with the Bridge Matching Sampler and CG-BG works listed in
[UPSTREAM.md](UPSTREAM.md) when their components are used.

## License

This repository is distributed under the
[FAIR Chemistry License](LICENSE). Third-party notices and license texts are
provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSES](LICENSES).
