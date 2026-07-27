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

The following asset-free example trains a small analytic two-dimensional
model, generates an SDE proposal, learns the backward controller, and produces
PF-ODE samples with importance weights:

```bash
cg-bms-train-forward experiment=mb2d_analytic_quickstart

cg-bms-sample-sde \
  experiment=mb2d_analytic_quickstart \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/forward/forward_step_00000008 \
  num_samples=256 batch_size=64 \
  output=outputs/mb2d_analytic_quickstart/proposal.npz

cg-bms-train-backward \
  experiment=mb2d_analytic_quickstart \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/forward/forward_step_00000008

cg-bms-sample-reweight \
  experiment=mb2d_analytic_quickstart \
  forward_checkpoint=outputs/mb2d_analytic_quickstart/forward/forward_step_00000008 \
  backward_checkpoint=outputs/mb2d_analytic_quickstart/backward/backward_step_00000008 \
  num_samples=256 batch_size=64 \
  output=outputs/mb2d_analytic_quickstart/samples_and_weights.npz

cg-bms-evaluate \
  experiment=mb2d_analytic_quickstart \
  samples=outputs/mb2d_analytic_quickstart/samples_and_weights.npz \
  output_dir=outputs/mb2d_analytic_quickstart/evaluation \
  bootstrap=false
```

## Cite

Please cite the software metadata in [CITATION.cff](CITATION.cff), together
with the Bridge Matching Sampler and CG-BG works listed in
[UPSTREAM.md](UPSTREAM.md) when their components are used.

## License

This repository is distributed under the
[FAIR Chemistry License](LICENSE). Third-party notices and license texts are
provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSES](LICENSES).
