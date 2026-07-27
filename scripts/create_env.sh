#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_EXE="${CG_BMS_CONDA_EXE:-$(command -v conda || true)}"
ENV_NAME="${CG_BMS_ENV_NAME:-cg-bms-jax}"
PREFIX="${CG_BMS_ENV_PREFIX:-}"

if [[ -z "${CONDA_EXE}" || ! -x "${CONDA_EXE}" ]]; then
  echo "conda was not found on PATH; set CG_BMS_CONDA_EXE if needed" >&2
  exit 2
fi

if [[ -n "${PREFIX}" ]]; then
  ENV_ARGS=(--prefix "${PREFIX}")
  ACTIVATE_TARGET="${PREFIX}"
else
  ENV_ARGS=(--name "${ENV_NAME}")
  ACTIVATE_TARGET="${ENV_NAME}"
fi

if ! "${CONDA_EXE}" run "${ENV_ARGS[@]}" python -c \
  'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' \
  >/dev/null 2>&1; then
  "${CONDA_EXE}" create --yes --override-channels --channel conda-forge \
    "${ENV_ARGS[@]}" python=3.11 pip
fi

run_python() {
  "${CONDA_EXE}" run "${ENV_ARGS[@]}" python "$@"
}

run_python -m pip install --upgrade pip wheel
run_python -m pip install \
  jax==0.4.38 jaxlib==0.4.38 \
  jax-cuda12-plugin==0.4.38 jax-cuda12-pjrt==0.4.38
run_python -m pip install -e "${ROOT}[pmf,eval]"
# The pinned ChemTrain metadata caps JAX at 0.4.37, while the pinned Diffrax
# requires JAX >=0.4.38.  The code is used with JAX 0.4.38 in the validated
# CG-BG environment, so install the fixed source without allowing pip to
# downgrade the already locked JAX stack.
run_python -m pip install --no-deps \
  "git+https://github.com/tummfm/chemtrain.git@a97ca2dd60c8327f574f269d02ec5edbccbae6b8"
run_python -m pip install --no-deps \
  "chemutils @ git+https://github.com/tummfm/cg-bg.git@948aaeff8a6b25de38b6e7b1112041c1cfd40573#subdirectory=external/chemutils"
run_python -m pip install --force-reinstall --no-deps \
  jax==0.4.38 jaxlib==0.4.38 \
  jax-cuda12-plugin==0.4.38 jax-cuda12-pjrt==0.4.38

echo "created Conda environment ${ACTIVATE_TARGET}"
echo "activate with: conda activate ${ACTIVATE_TARGET}"
