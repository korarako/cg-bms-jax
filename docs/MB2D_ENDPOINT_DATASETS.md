# MB2D endpoint datasets and provenance

## Purpose

This document prevents three distinct objects from being conflated:

1. the analytic MB2D Boltzmann target;
2. a deliberately biased synthetic endpoint law used for a stress test;
3. a finite endpoint archive sampled from the analytic target for supervised
   bridge matching.

The target energy, PF-ODE likelihood, and formal reweighting calculation are
the same across the MB2D experiments. Only the bridge endpoint source changes.

## Canonical target identity

The physical coordinate is \(x\in[0,50]^2\), the controller coordinate is

\[
z=(x-(25,25))/(10,10),
\]

and the source is \(q_0(z)=\mathcal N(0,I_2)\). At the released temperature,

\[
\pi(x)\propto\exp[-U_{\rm MB}(x)]\mathbf 1\{x\in[0,50]^2\},
\qquad \beta=kT=1.
\]

The endpoint generator, Energy-BMS target, reweighting target, and exact
evaluation grid must share this complete target identity. An endpoint archive
from ADTM or from a differently scaled Muller--Brown implementation is not
compatible merely because it is also called `mb2d`.

## Dataset registry

### `biased_endpoint_stress_test`

This is the legacy `full_support_diagonal_gaussian_mixture_v1` mapping in
`configs/experiment/mb2d_analytic.yaml`, with seed 17 and 100,000 endpoints.
Its component weights are intentionally non-equilibrium:

```text
right / central / upper / broad-tail = 0.65 / 0.20 / 0.13 / 0.02
```

The completed bridge-only and bridge-to-Energy results under
`outputs/mb2d_formal_matrix_20260724/` are frozen under this identity. They
are useful because the proposal is strongly biased while retaining full
support. They must not be called equilibrium bridge, MD bridge, or
reference-data bridge results.

The cold Energy-BMS arm in the same directory did not consume this endpoint
law. It remains a canonical data-free result.

### `equilibrium_exact_v1`

This is the canonical bridge endpoint dataset. It is sampled from a normalized
analytic grid at `beta=1`, with independent uniform jitter inside the selected
grid cell. The jitter removes lattice artifacts without changing the
piecewise-constant grid approximation.

Frozen release artifact:

```text
path: data/mb2d_equilibrium_exact_v1/endpoints.npz
sha256: f4c43d8feff619102e5cc1f8ad2dbd422d9a201453d356d4f59e48cb9032347d
grid: 1024 x 1024
seed: 20260725
train / validation / test: 100000 / 50000 / 100000
```

The accepted generator diagnostics are:

| diagnostic | value |
|---|---:|
| train histogram JS vs analytic grid | 0.00395496 |
| train histogram TV | 0.0323033 |
| three-basin L1 error | 0.000271461 |
| energy-mean error | 0.00249534 |
| 512-to-1024 energy-mean delta | 3.52532e-9 |
| 512-to-1024 basin L1 delta | 1.56045e-6 |

The complete target signature, split hashes, and acceptance thresholds are in
`data/mb2d_equilibrium_exact_v1/endpoints.manifest.json`.

The archive is called an **exact-grid Boltzmann endpoint dataset**, not an MD
dataset. Its manifest must contain:

- schema and dataset version;
- analytic target kind and target-signature SHA-256;
- `beta`, `kT`, physical domain, affine offset, and affine scale;
- grid resolution and cell area;
- RNG algorithm and seed;
- train/validation/test split sizes and split hashes;
- archive SHA-256;
- empirical energy mean and three-basin masses;
- empirical-versus-grid JS/TV diagnostics;
- a grid-resolution convergence diagnostic.

The formal split policy is:

```text
train:      100,000 endpoints
validation:  50,000 endpoints
test:       100,000 endpoints
```

Bridge training reads only `train`. Validation may choose checkpoints but may
not update parameters. `test` is reserved for endpoint-distribution auditing;
formal proposal evaluation continues to use the normalized analytic grid.

## Sampling algorithm

For an \(n\times n\) midpoint grid:

1. evaluate reduced energy \(\beta U(x_{ij})\);
2. compute cell masses with a stable log-sum-exp normalization;
3. draw categorical cell indices from those masses;
4. draw an independent uniform offset within each selected cell;
5. convert physical \(x\) to controller coordinate \(z\);
6. split deterministically and save the archive and manifest.

The implementation must verify that every physical coordinate is inside the
formal box and that every stored value is finite.

## Acceptance gates

Before any formal bridge run:

- archive SHA-256 matches its manifest;
- composed config target identity matches the manifest exactly;
- the same seed regenerates bitwise-identical arrays;
- empirical basin masses and energy moments agree with the normalized grid
  within their sampling uncertainty;
- the empirical 2D histogram converges toward the exact grid as sample count
  increases;
- changing beta, domain, affine transform, or file content is rejected;
- the legacy synthetic-mixture loader still reproduces its old identity.

## Formal rerun matrix

Only the endpoint-dependent arms need retraining:

| arm | forward | backward | PF samples | status |
|---|---:|---:|---:|---|
| Cold Energy-BMS | reuse completed 3-seed 100k result | reuse | reuse | canonical, endpoint-independent |
| Equilibrium bridge | 100k x 3 seeds | 100k x 3 | 100k x 3 | new |
| Equilibrium bridge-to-Energy | bridge initializer + fresh Energy-BMS 100k x 3 | 100k x 3 | 100k x 3 | new |

The new runs live under
`outputs/mb2d_equilibrium_bridge_matrix_v1/`. The old
`outputs/mb2d_formal_matrix_20260724/` tree is never overwritten.

Each result must report both proposal and no-clip reweighted metrics:

- 2D JS and PMF error;
- three-basin mass error;
- energy mean/error;
- ESS/N, maximum weight, top 0.1% and 1% weight mass;
- log-weight variance/span;
- finite, valid, and in-domain fractions.

Drop-top-1% views remain separately labelled diagnostics.

## Claim boundary

The intended release claim is:

> Exact-density PF-ODE reweighting improves target-distribution estimates for
> BMS proposals across data-free, deliberately biased, and
> equilibrium-endpoint initialization regimes, subject to reported
> overlap/ESS diagnostics.

The legacy biased endpoint experiment supports the “deliberately biased”
part of this statement. It cannot substitute for the new equilibrium bridge
matrix. The equilibrium matrix is considered successful only when the
no-clip weighted errors improve consistently across seeds without being
dominated by a negligible effective sample size.
