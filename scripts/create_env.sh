#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CG_BMS_CONDA_ROOT:-/ds/project/weilong/ke/miniconda3}"
PREFIX="${CG_BMS_ENV_PREFIX:-${CONDA_ROOT}/envs/cg-bms-jax}"

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "conda not found at ${CONDA_ROOT}/bin/conda" >&2
  exit 2
fi

if [[ ! -x "${PREFIX}/bin/python" ]]; then
  "${CONDA_ROOT}/bin/conda" create --yes --override-channels --channel conda-forge \
    --prefix "${PREFIX}" python=3.11 pip
fi

"${PREFIX}/bin/python" -m pip install --upgrade pip wheel
"${PREFIX}/bin/python" -m pip install \
  jax==0.4.38 jaxlib==0.4.38 \
  jax-cuda12-plugin==0.4.38 jax-cuda12-pjrt==0.4.38
"${PREFIX}/bin/python" -m pip install -e "${ROOT}[pmf,eval,dev]"
# The pinned ChemTrain metadata caps JAX at 0.4.37, while the pinned Diffrax
# requires JAX >=0.4.38.  The code is used with JAX 0.4.38 in the validated
# CG-BG environment, so install the fixed source without allowing pip to
# downgrade the already locked JAX stack.
"${PREFIX}/bin/python" -m pip install --no-deps \
  "git+https://github.com/tummfm/chemtrain.git@a97ca2dd60c8327f574f269d02ec5edbccbae6b8"
"${PREFIX}/bin/python" -m pip install --no-deps \
  "chemutils @ git+https://github.com/tummfm/cg-bg.git@948aaeff8a6b25de38b6e7b1112041c1cfd40573#subdirectory=external/chemutils"
"${PREFIX}/bin/python" -m pip install --force-reinstall --no-deps \
  jax==0.4.38 jaxlib==0.4.38 \
  jax-cuda12-plugin==0.4.38 jax-cuda12-pjrt==0.4.38

echo "created ${PREFIX}"
