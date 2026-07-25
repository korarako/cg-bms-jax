# Third-party notices

This distribution is licensed under the FAIR Chemistry License reproduced in
`LICENSE`. The incorporated FAIR Chemistry Acceptable Use Policy is snapshotted
in `LICENSES/FAIR-CHEMISTRY-AUP.md`.

## Bridge Matching Sampler

The BMS mathematics, process implementation, controller architecture, and the
faithful JAX PaiNN port are derived from
[`DenisBless/BridgeMatchingSampler`](https://github.com/DenisBless/BridgeMatchingSampler)
revision `d19a27b854fd77387c43109e37b597bc35252e7d`.
The all-atom Ala2 topology file and the two chirality-restraint definitions
are retained from the same pinned revision.

Copyright (c) Meta Platforms, Inc. and affiliates.

The root `LICENSE` is content-identical to the pinned upstream file after the
repository's configured LF line-ending normalization.

## WT-ASBS

The optional supervised warm-start objective and its variance weighting are
adapted from [`facebookresearch/wt-asbs`](https://github.com/facebookresearch/wt-asbs)
revision `3177ec826b1111a8ee47ee3b9d1683a3a55917c6`.

Copyright (c) Meta Platforms, Inc. and affiliates. WT-ASBS is distributed
under the FAIR Chemistry License already reproduced by this distribution.

## CG-BG

PMF reconstruction, ambient center-of-mass conventions, archive fields, and
evaluation conventions are adapted from
[`tummfm/cg-bg`](https://github.com/tummfm/cg-bg) revision
`948aaeff8a6b25de38b6e7b1112041c1cfd40573`.

Copyright (c) 2026 Multiscale Modeling of Fluid Materials.

CG-BG is distributed under the MIT License, reproduced in
`LICENSES/MIT-CG-BG.txt`.

## External runtime dependencies

ChemTrain is installed from
[`tummfm/chemtrain`](https://github.com/tummfm/chemtrain) revision
`a97ca2dd60c8327f574f269d02ec5edbccbae6b8`. ChemUtils and its MACE-JAX
implementation are installed directly from the pinned CG-BG revision. They are
external dependencies, not copied into this wheel; their own distributions
retain their upstream notices and license files. In particular, the MACE-JAX
subtree records:

- Copyright (c) 2022 mace-jax
- Copyright (c) 2025 tummfm
- MIT License

## Downloaded PMF and reference assets

The fetch script downloads files from
[`bojuntum/CGPeptides`](https://huggingface.co/datasets/bojuntum/CGPeptides) at
revision `39765bbcfee382e5f30445589d7fe28ebb6cfff8`. The pinned manifest records
the dataset card license as MIT. Checkpoints and trajectory files are verified
cache artifacts and are not redistributed in the source archive or wheel.
