# cg-bms-jax frozen result bundle

This is a lightweight, reproducible result freeze. Large PF archives and checkpoints are **not copied**; their independently verified SHA-256 digests, manifests, array inventories and PF metadata are stored under `provenance/`.

## Scientific scope

- MB CG1D: three-seed cold Energy-BMS plus formal reweighting.
- MB2D: three-seed cold Energy-BMS, exact-equilibrium bridge, and exact-equilibrium bridge-to-Energy.
- MB2D legacy: bridge and bridge-to-Energy on deliberately biased synthetic full-support endpoints. These are stress tests; they are **not MD** and **not equilibrium endpoint** experiments.
- CG Ala2: the strict release-tolerance W10k-to-E20k-to-B20k PF10k result and the in-root W10k-to-E100k-to-B100k PF20k mixed/low-ESS result.
- CG Ala2 target: six-bead core-beta CG-BG MACE PMF plus fixed support terms at 300 K in ambient18 auxiliary-COM coordinates; warm endpoints are SHA-pinned CG-BG explicit core-beta data.
- The CG Ala2 explicit `flow_ub` asset is a structural plotting reference. It is not claimed to be an exact equilibrium sample from the full PMF-plus-support formal target.
- All-atom Ala2 is deliberately excluded.

## Formal estimator and plotting

Every stored result passed an explicit audit that `logw_raw` equals `-target_reduced_energy-logq_ambient` and that `weights` equal the softmax of that recomputed formula over finite valid/support samples. Invalid/out-of-support samples retain zero formal weight. No clipping, capping or PSIS replacement is part of the formal result.

Energy plots use the reference 0.5%--99.5% energy-quantile window plus 5% padding only to avoid an unreadable x axis. All generated samples are accounted for; only finite valid/support samples have nonzero formal weight. Omitted display mass is recorded in each evaluator metric JSON.

Each PF arm also has a standalone formal no-clip cumulative weight-tail panel annotated with ESS/N, maximum weight, top-tail mass, and raw-log-weight variance/span. A separately labelled `diagnostic_drop_top_1_percent/` view drops the top 1% of finite raw weights and renormalizes. It is diagnostic only and is never used in the headline benchmark tables.

All MB panels obey `VISUAL_CONTRACT.json`: exact/true is black, a distinct reference is green `#8FD18B`, proposal is orange `#F2A174`, and reweighted uses a light-blue `#B9CBEA` fill with a blue `#4472C4` outline. Reference/proposal/reweighted one-dimensional marginals and energy distributions are translucent filled densities with crisp outlines; exact/true remains a black line. Every MB2D FES uses the same viridis map, physical [0,50] x [0,50] extent, and fixed 0--12 kT color scale labelled `Free energy`. CG Ala2 reference, proposal and reweighted Ramachandran maps are saved as separate standalone panels without overlays.

## Benchmark

| system | arm | role | data identity | N seeds | proposal JS | reweighted JS | ESS/N |
|---|---|---|---|---:|---:|---:|---:|
| ala2_cg | mixed_low_ess_warm10k_energy100k_pf20k | mixed_negative_ablation | six-bead core-beta CG-BG MACE PMF + fixed support terms, 300 K, ambient18 auxiliary-COM; warm endpoints hash-pinned CG-BG explicit core-beta data | 1 | 0.0786754 | 0.109545 | 0.0140783 |
| ala2_cg | positive_warm10k_energy20k_pf10k | primary | six-bead core-beta CG-BG MACE PMF + fixed support terms, 300 K, ambient18 auxiliary-COM; warm endpoints hash-pinned CG-BG explicit core-beta data | 1 | 0.114469 | 0.0879012 | 0.135554 |
| mb2d | biased_bridge_energy | biased_endpoint_stress_test | deliberately_biased_synthetic_full_support_v1 | 3 | 0.405559 | 0.0136739 | 0.144835 |
| mb2d | biased_bridge_only | biased_endpoint_stress_test | deliberately_biased_synthetic_full_support_v1 | 3 | 0.502579 | 0.0340462 | 0.0395534 |
| mb2d | cold_energy_only | primary | endpoint_independent_energy_bms | 3 | 0.0260363 | 0.00544036 | 0.814894 |
| mb2d | equilibrium_bridge_energy | primary | equilibrium_exact_v1 | 3 | 0.007999 | 0.00698289 | 0.975432 |
| mb2d | equilibrium_bridge_only | primary | equilibrium_exact_v1 | 3 | 0.00799362 | 0.00632629 | 0.985633 |
| mb_cg1d | energy_only | primary | endpoint_independent_energy_bms | 3 | 0.0848956 | 0.000117778 | 0.565384 |

Per-run authoritative values are in `per_seed.csv` and `per_seed.json`; aggregate mean/std/min/max values are in `summary.csv` and `summary.json`. Standalone panels are below `figures/`; formal and diagnostic estimators are in separate subdirectories. Every panel directory contains `plot_data/` CSV/JSON with the exact derived grid, histogram, marginal, energy-window or weight-tail values used by that panel. `SOURCE_RESULTS_INDEX.csv` and `.json` identify every NAS checkpoint, PF archive, reference, PMF and endpoint dataset by path, byte size and SHA-256; those large sources are not copied into Git. `SHA256SUMS` covers every payload file except itself and the subsequently generated self-describing `MANIFEST.json`.
