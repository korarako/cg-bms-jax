#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
GPU="${GPU:-0}"
TRAIN_PID="${TRAIN_PID:?TRAIN_PID must identify the running E20k process}"

EXP=ala2_ambient18_300k_bms_eta10_official_cadence_additive_v3
RUN="${ROOT}/outputs/ala2_bms_official_cadence_additive_v3/warm10k_then_e20k_s0_run1"
WARM="${RUN}/warm/forward_pretrain_step_00010000"
FWD="${RUN}/forward/forward_step_00020000"
W_SAMPLES="${RUN}/proposal_warm10k_sde20k.npz"
WE_SAMPLES="${RUN}/proposal_warm10k_e20k_sde20k.npz"
W_EVAL="${RUN}/evaluation_warm10k_proposal20k"
WE_EVAL="${RUN}/evaluation_warm10k_e20k_proposal20k"
STATUS="${RUN}/postprocess_status.txt"
BIN="${ENV_ROOT}/bin"

mkdir -p "${RUN}" "${ROOT}/logs"

write_status() {
  local phase="$1"
  shift
  {
    printf 'phase=%s\n' "${phase}"
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'watcher_pid=%s\n' "$$"
    printf 'training_pid=%s\n' "${TRAIN_PID}"
    printf 'gpu=%s\n' "${GPU}"
    printf '%s\n' "$@"
  } >"${STATUS}.tmp"
  mv "${STATUS}.tmp" "${STATUS}"
}

on_error() {
  local code=$?
  write_status failed \
    "exit_code=${code}" \
    "failed_line=${BASH_LINENO[0]}"
  exit "${code}"
}
trap on_error ERR

cd "${ROOT}"
unset LD_LIBRARY_PATH
unset LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONUNBUFFERED=1
export PATH="${BIN}:${PATH}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

if [[ ! -r "/proc/${TRAIN_PID}/cmdline" ]]; then
  write_status failed 'reason=training_pid_not_running_at_watcher_start'
  exit 1
fi
if ! tr '\0' ' ' <"/proc/${TRAIN_PID}/cmdline" | grep -Fq \
  'warm10k_then_e20k_s0_run1/forward'; then
  write_status failed 'reason=training_pid_command_mismatch'
  exit 1
fi

write_status waiting_energy 'energy_step=running/20000'
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 15
done

write_status verify_energy 'energy_process=exited'
test -d "${WARM}"
test -d "${FWD}"
WARM_SHA=$("${BIN}/python" -c \
  'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
  "${WARM}")
FWD_SHA=$("${BIN}/python" -c \
  'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
  "${FWD}")

write_status sample_w \
  "warm_sha256=${WARM_SHA}" \
  "forward_sha256=${FWD_SHA}" \
  'samples=0/20000'
"${BIN}/cg-bms-sample-sde" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  "forward_checkpoint=${WARM}" \
  controller_kind=forward_pretrain \
  seed=2 \
  num_samples=20000 \
  batch_size=64 \
  progress_every_batches=10 \
  "output=${W_SAMPLES}"

write_status sample_we \
  "warm_samples=${W_SAMPLES}" \
  'samples=0/20000'
"${BIN}/cg-bms-sample-sde" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  "forward_checkpoint=${FWD}" \
  controller_kind=forward \
  seed=2 \
  num_samples=20000 \
  batch_size=64 \
  progress_every_batches=10 \
  "output=${WE_SAMPLES}"

write_status eval_w "warm_samples=${W_SAMPLES}"
"${BIN}/cg-bms-evaluate" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  "samples=${W_SAMPLES}" \
  "output_dir=${W_EVAL}" \
  bootstrap=true \
  n_bootstraps=500

write_status eval_we "warm_energy_samples=${WE_SAMPLES}"
"${BIN}/cg-bms-evaluate" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  "samples=${WE_SAMPLES}" \
  "output_dir=${WE_EVAL}" \
  bootstrap=true \
  n_bootstraps=500

write_status complete \
  "warm_sha256=${WARM_SHA}" \
  "forward_sha256=${FWD_SHA}" \
  "warm_samples=${W_SAMPLES}" \
  "warm_energy_samples=${WE_SAMPLES}" \
  "warm_evaluation=${W_EVAL}" \
  "warm_energy_evaluation=${WE_EVAL}"
printf 'postprocess complete: %s\n' "${RUN}"
