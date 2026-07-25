#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CG_BMS_PYTHON:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax/bin/python}"
RUN_ID="${CG_BMS_SMOKE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_ROOT="${CG_BMS_SMOKE_OUTPUT:-${ROOT}/outputs/smoke/${RUN_ID}}"
NUM_SAMPLES="${CG_BMS_SMOKE_NUM_SAMPLES:-64}"
BATCH_SIZE="${CG_BMS_SMOKE_BATCH_SIZE:-16}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="${ROOT}/${RUN_ROOT}"
fi

cd "${ROOT}"
echo "cg-bms-jax smoke output: ${RUN_ROOT}"
"${PYTHON}" scripts/check_environment.py
"${PYTHON}" -m pytest -q

"${PYTHON}" -m cg_bms_jax.experiment.train_forward \
  experiment=mb_cg1d_smoke \
  "output_dir=${RUN_ROOT}/forward"

shopt -s nullglob
forward_checkpoints=("${RUN_ROOT}/forward"/forward_step_*)
if (( ${#forward_checkpoints[@]} != 1 )); then
  echo "Expected exactly one forward checkpoint, found ${#forward_checkpoints[@]}" >&2
  exit 1
fi
FORWARD_CHECKPOINT="${forward_checkpoints[0]}"

# This stochastic sample is proposal diagnostics only.  It deliberately has no
# logq or importance weights attached.
"${PYTHON}" -m cg_bms_jax.experiment.sample_sde \
  experiment=mb_cg1d_smoke \
  "forward_checkpoint=${FORWARD_CHECKPOINT}" \
  "num_samples=${NUM_SAMPLES}" \
  "batch_size=${BATCH_SIZE}" \
  "output=${RUN_ROOT}/sde_proposal.npz"

"${PYTHON}" -m cg_bms_jax.experiment.train_backward \
  experiment=mb_cg1d_smoke \
  "forward_checkpoint=${FORWARD_CHECKPOINT}" \
  "output_dir=${RUN_ROOT}/backward"

backward_checkpoints=("${RUN_ROOT}/backward"/backward_step_*)
if (( ${#backward_checkpoints[@]} != 1 )); then
  echo "Expected exactly one backward checkpoint, found ${#backward_checkpoints[@]}" >&2
  exit 1
fi
BACKWARD_CHECKPOINT="${backward_checkpoints[0]}"

# PF-ODE coordinates and logq are integrated as one augmented state, then the
# PMF target is evaluated and normalized importance weights are written.
"${PYTHON}" -m cg_bms_jax.experiment.sample_reweight \
  experiment=mb_cg1d_smoke \
  "forward_checkpoint=${FORWARD_CHECKPOINT}" \
  "backward_checkpoint=${BACKWARD_CHECKPOINT}" \
  "num_samples=${NUM_SAMPLES}" \
  "batch_size=${BATCH_SIZE}" \
  "output=${RUN_ROOT}/samples_and_weights.npz"

"${PYTHON}" -m cg_bms_jax.experiment.evaluate \
  experiment=mb_cg1d_smoke \
  "samples=${RUN_ROOT}/samples_and_weights.npz" \
  "output_dir=${RUN_ROOT}/evaluation" \
  bootstrap=false \
  n_bootstraps=1

echo "cg-bms-jax smoke passed: ${RUN_ROOT}"
