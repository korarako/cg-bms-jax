# Publishing `cg-bms-jax` v0.1.0 safely

This repository has a large experimental working tree. The initial public
commit is therefore built from an explicit allowlist rather than from the
whole directory.

The publishable benchmark scope is:

- MB CG1D Energy-BMS plus exact-density reweighting;
- analytic MB2D cold Energy-BMS, equilibrium bridge, and
  bridge-to-Energy, each with exact-density reweighting;
- the labelled biased-endpoint MB2D stress test;
- six-bead CG Ala2 selected positive, mixed, and failure diagnostics.

Full-atom Ala2 benchmark configs, run scripts, datasets, result documents, and
artifacts are deliberately excluded. Generic source modules and tests may
remain for development continuity, but they are not part of the v0.1.0
benchmark claim.

## Two independent gates

After the atomic bundle has been built at `artifacts/release_v0.1.0`, run:

```bash
python scripts/validate_release_scope.py artifacts/release_v0.1.0
python scripts/validate_publication_tree.py --mode worktree
```

The first command verifies every frozen result file against `MANIFEST.json`.
The second command checks the wider Git candidate set for:

- files outside `release/publication_allowlist_v0.1.0.txt`;
- files larger than 25 MiB;
- checkpoints, logs, caches, archives, and historical probe artifacts;
- likely credentials or private keys;
- unresolved release placeholders;
- broken relative links in `README.md`;
- full-atom benchmark materials in configs, data, docs, scripts, or results.

Both gates are read-only.

## Safe initial staging

Never use `git add -A`, `git add .`, or `git add artifacts`.
From the repository root, stage only these paths:

```bash
git add -- \
  .gitattributes .gitignore CITATION.cff environment.yml LICENSE \
  pyproject.toml README.md THIRD_PARTY_NOTICES.md UPSTREAM.md \
  LICENSES src tests assets/manifest.yaml \
  configs/*.yaml \
  configs/experiment/mb*.yaml \
  configs/experiment/ala2_ambient18*.yaml \
  configs/experiment/ala2_cg*.yaml \
  data/mb2d_equilibrium_exact_v1 \
  docs/MB2D_ENDPOINT_DATASETS.md \
  docs/MB_EXPERIMENT_MATRIX.md \
  docs/PUBLISHING_V0.1.0.md \
  docs/RELEASE_RESULTS_V0.1.0.md \
  docs/periodic_forward_diagnostics.md \
  scripts/*.py \
  scripts/create_env.sh \
  scripts/finalize_mb2d_equilibrium_bridge_matrix.sh \
  scripts/run_additive_v3_20k_validation.sh \
  scripts/run_ala2_cg_release_pf10k.sh \
  scripts/run_ala2_cold_clip_probe.sh \
  scripts/run_ala2_energy20k_backward20k_pf20k_eval.sh \
  scripts/run_ala2_warm100k_energy100k.sh \
  scripts/run_mb2d_controlled_sweeps.sh \
  scripts/run_mb2d_equilibrium_bridge_matrix.sh \
  scripts/run_mb2d_formal_matrix.sh \
  scripts/run_mb_cg1d_three_seed.sh \
  scripts/run_mb_cg1d_three_seed_parallel.sh \
  scripts/run_smoke.sh \
  scripts/run_warm_vs_energy_b20k_pf20k.sh \
  scripts/watch_warm10k_e20k_then_eval.sh \
  release/publication_allowlist_v0.1.0.txt \
  artifacts/release_v0.1.0
```

Now prove that the index is exactly the allowlisted candidate set:

```bash
python scripts/validate_publication_tree.py --mode staged
git diff --cached --check
git diff --cached --stat
git status --short
```

Only after inspecting those outputs should the initial commit be created:

```bash
git commit -m "release: freeze cg-bms-jax v0.1.0 benchmarks"
```

No command in this document creates a remote, pushes a branch, or publishes a
release.
