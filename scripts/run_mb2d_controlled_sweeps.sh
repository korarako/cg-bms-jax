#!/usr/bin/env bash
set -Eeuo pipefail

# Controlled sweeps use one already-completed seed from run_mb2d_formal_matrix.sh.
# They never change the experiment identity: only the forward snapshot,
# backward estimator budget, or ODE tolerance changes.

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
EXP="${EXP:-mb2d_analytic}"
MATRIX_ROOT="${MATRIX_ROOT:-${ROOT}/outputs/mb2d_formal_matrix}"
RUN="${MATRIX_ROOT}/seed_${SEED}"
OUT="${RUN}/controlled_sweeps"
BIN="${ENV_ROOT}/bin"
PF_SAMPLES="${PF_SAMPLES:-20000}"
PF_BATCH_SIZE="${PF_BATCH_SIZE:-1024}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-10000 25000 50000 75000 100000}"
CHECKPOINT_BWD_UPDATES="${CHECKPOINT_BWD_UPDATES:-100000}"
BACKWARD_BUDGETS="${BACKWARD_BUDGETS:-10000 25000 50000 100000}"

mkdir -p "${OUT}"
cd "${ROOT}"
unset LD_LIBRARY_PATH
unset LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONUNBUFFERED=1
export PATH="${BIN}:${PATH}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

verify_checkpoint() {
  "${BIN}/python" -c \
    'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "$1"
}

run_backward_pf() {
  local label="$1"
  local forward="$2"
  local updates="$3"
  local directory="${OUT}/${label}"
  local step_tag
  step_tag="$(printf '%08d' "${updates}")"
  local backward="${directory}/backward/backward_step_${step_tag}"
  local samples="${directory}/samples_and_weights_${PF_SAMPLES}.npz"
  mkdir -p "${directory}"
  verify_checkpoint "${forward}" >/dev/null
  if [[ ! -d "${backward}" ]]; then
    "${BIN}/cg-bms-train-backward" \
      "experiment=${EXP}" \
      forward_controller_kind=forward \
      "forward_checkpoint=${forward}" \
      "updates=${updates}" \
      "seed=$((SEED + 501))" \
      "output_dir=${directory}/backward" \
      progress_every=100
  fi
  verify_checkpoint "${backward}" >/dev/null
  if [[ ! -f "${samples}" ]]; then
    "${BIN}/cg-bms-sample-reweight" \
      "experiment=${EXP}" \
      forward_controller_kind=forward \
      "forward_checkpoint=${forward}" \
      "backward_checkpoint=${backward}" \
      "seed=$((SEED + 601))" \
      "num_samples=${PF_SAMPLES}" \
      "batch_size=${PF_BATCH_SIZE}" \
      "target_batch_size=${PF_BATCH_SIZE}" \
      progress_every_batches=10 \
      likelihood.dt0=1.0e-3 \
      likelihood.rtol=1.0e-5 \
      likelihood.atol=1.0e-6 \
      likelihood.max_steps=16384 \
      reweight.density_mode=exact_ambient \
      reweight.clip=null \
      "output=${samples}"
  fi
  "${BIN}/cg-bms-evaluate" \
    "experiment=${EXP}" \
    "samples=${samples}" \
    "output_dir=${directory}/evaluation" \
    bootstrap=false \
    compat_clip_percentile=null
}

# 1. Forward checkpoint sweep: same bridge-initialized energy trajectory and
# the same backward budget, seeds, PF sample count, and likelihood tolerance.
FORWARD_COMPARE=()
for step in ${CHECKPOINT_STEPS}; do
  step_tag="$(printf '%08d' "${step}")"
  forward="${RUN}/bridge_energy_forward/forward_step_${step_tag}"
  label="forward_${step_tag}_backward_$(printf '%08d' "${CHECKPOINT_BWD_UPDATES}")"
  run_backward_pf "${label}" "${forward}" "${CHECKPOINT_BWD_UPDATES}"
  FORWARD_COMPARE+=(--archive "${step}=${OUT}/${label}/samples_and_weights_${PF_SAMPLES}.npz")
done
"${BIN}/cg-bms-mb2d-suite" compare \
  "${FORWARD_COMPARE[@]}" \
  --output-dir "${OUT}/forward_checkpoint_comparison"

# 2. Backward budget sweep: one frozen final forward controller.
FINAL_FORWARD="${RUN}/bridge_energy_forward/forward_step_00100000"
BACKWARD_COMPARE=()
for updates in ${BACKWARD_BUDGETS}; do
  label="final_forward_backward_$(printf '%08d' "${updates}")"
  run_backward_pf "${label}" "${FINAL_FORWARD}" "${updates}"
  BACKWARD_COMPARE+=(--archive "${updates}=${OUT}/${label}/samples_and_weights_${PF_SAMPLES}.npz")
done
"${BIN}/cg-bms-mb2d-suite" compare \
  "${BACKWARD_COMPARE[@]}" \
  --output-dir "${OUT}/backward_budget_comparison"

# 3. ODE tolerance sweep: identical initial source samples and controller pair.
FINAL_BACKWARD="${RUN}/bridge_energy_backward100k/backward_step_00100000"
declare -A RTOL=( [loose]=1.0e-4 [formal]=1.0e-5 [strict]=1.0e-7 )
declare -A ATOL=( [loose]=1.0e-5 [formal]=1.0e-6 [strict]=1.0e-8 )
TOLERANCE_ARGUMENTS=()
for label in loose formal strict; do
  archive="${OUT}/tolerance_${label}.npz"
  if [[ ! -f "${archive}" ]]; then
    "${BIN}/cg-bms-sample-reweight" \
      "experiment=${EXP}" \
      forward_controller_kind=forward \
      "forward_checkpoint=${FINAL_FORWARD}" \
      "backward_checkpoint=${FINAL_BACKWARD}" \
      "seed=$((SEED + 701))" \
      "num_samples=${PF_SAMPLES}" \
      "batch_size=${PF_BATCH_SIZE}" \
      "target_batch_size=${PF_BATCH_SIZE}" \
      progress_every_batches=10 \
      likelihood.dt0=1.0e-3 \
      "likelihood.rtol=${RTOL[${label}]}" \
      "likelihood.atol=${ATOL[${label}]}" \
      likelihood.max_steps=32768 \
      reweight.density_mode=exact_ambient \
      reweight.clip=null \
      "output=${archive}"
  fi
  TOLERANCE_ARGUMENTS+=(--archive "${label}=${archive}")
done
"${BIN}/cg-bms-mb2d-suite" tolerance \
  "${TOLERANCE_ARGUMENTS[@]}" \
  --reference strict \
  --output "${OUT}/ode_tolerance_comparison.json"

printf 'MB2D controlled sweeps complete: %s\n' "${OUT}"

