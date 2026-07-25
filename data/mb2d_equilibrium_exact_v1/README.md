# MB2D exact-grid Boltzmann endpoints v1

- Target: analytic CG-BG Muller--Brown energy on `[0, 50]^2`
- Temperature: `beta=1`, `kT=1`
- Controller map: `z = (x - [25, 25]) / [10, 10]`
- Generator: 1024 x 1024 midpoint quadrature with uniform within-cell jitter
- Seed: `20260725`
- Splits: 100,000 train / 50,000 validation / 100,000 test
- Archive SHA-256:
  `f4c43d8feff619102e5cc1f8ad2dbd422d9a201453d356d4f59e48cb9032347d`

`endpoints.manifest.json` contains the target and coordinate signatures,
split hashes, empirical target diagnostics, and grid-convergence checks.

Generate a new versioned dataset with:

```bash
cg-bms-generate-mb2d-equilibrium \
  output=data/<new-version>/endpoints.npz
```

The generator refuses to overwrite an existing archive unless
`overwrite=true` is supplied explicitly. Never overwrite this v1 archive:
create a new dataset version and a new experiment config instead.
