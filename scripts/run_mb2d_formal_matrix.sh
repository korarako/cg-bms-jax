#!/usr/bin/env bash
set -Eeuo pipefail

# Pure-JAX analytic MB2D matrix:
#   biased full-support bridge, cold Energy-BMS, bridge -> Energy-BMS
#   -> matched backward controllers -> PF/logq -> formal no-clip reweighting.
#
# Environment overrides:
#   ROOT, ENV_ROOT, GPU, SEEDS ("0 1 2" or "0,1,2"), NUM_SAMPLES,
#   RUN_DIAGNOSTICS

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
PF_BATCH_SIZE="${PF_BATCH_SIZE:-1024}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
EXP="${EXP:-mb2d_analytic}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/mb2d_formal_matrix}"
BIN="${ENV_ROOT}/bin"
# A worker-specific status path allows disjoint seed sets to run concurrently
# in the same RUN_ROOT without racing on one monitoring file.
STATUS="${STATUS:-${RUN_ROOT}/status.txt}"

mkdir -p "${RUN_ROOT}" "${ROOT}/logs"
cd "${ROOT}"
unset LD_LIBRARY_PATH
unset LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONUNBUFFERED=1
export PATH="${BIN}:${PATH}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

write_status() {
  local phase="$1"
  shift
  {
    printf 'phase=%s\n' "${phase}"
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'pid=%s\n' "$$"
    printf 'gpu=%s\n' "${GPU}"
    printf '%s\n' "$@"
  } >"${STATUS}.tmp"
  mv "${STATUS}.tmp" "${STATUS}"
}

on_error() {
  local code=$?
  write_status failed "exit_code=${code}" "failed_line=${BASH_LINENO[0]}"
  exit "${code}"
}
trap on_error ERR

verify_checkpoint() {
  "${BIN}/python" -c \
    'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "$1"
}

for seed in ${SEEDS//,/ }; do
  RUN="${RUN_ROOT}/seed_${seed}"
  BRIDGE_DIR="${RUN}/bridge"
  COLD_DIR="${RUN}/cold_forward"
  WARM_ENERGY_DIR="${RUN}/bridge_energy_forward"
  BRIDGE="${BRIDGE_DIR}/forward_pretrain_step_00100000"
  COLD="${COLD_DIR}/forward_step_00100000"
  WARM_ENERGY="${WARM_ENERGY_DIR}/forward_step_00100000"
  mkdir -p "${RUN}"

  write_status bridge_pretrain "seed=${seed}" "step=0/100000"
  if [[ ! -d "${BRIDGE}" ]]; then
    "${BIN}/cg-bms-pretrain-flat-bridge" \
      "experiment=${EXP}" \
      "seed=${seed}" \
      "output_dir=${BRIDGE_DIR}" \
      progress_every=100 \
      checkpoint_every=10000
  fi
  BRIDGE_SHA="$(verify_checkpoint "${BRIDGE}")"

  write_status cold_forward \
    "seed=${seed}" "bridge_sha256=${BRIDGE_SHA}" "step=0/100000"
  if [[ ! -d "${COLD}" ]]; then
    "${BIN}/cg-bms-train-forward" \
      "experiment=${EXP}" \
      "seed=${seed}" \
      "output_dir=${COLD_DIR}" \
      save_checkpoint_interval_outer=5 \
      progress_every=100
  fi
  COLD_SHA="$(verify_checkpoint "${COLD}")"

  write_status bridge_energy_forward \
    "seed=${seed}" "cold_sha256=${COLD_SHA}" "step=0/100000"
  if [[ ! -d "${WARM_ENERGY}" ]]; then
    "${BIN}/cg-bms-train-forward" \
      "experiment=${EXP}" \
      "seed=${seed}" \
      "initialize_controller_from=${BRIDGE}" \
      "output_dir=${WARM_ENERGY_DIR}" \
      save_checkpoint_interval_outer=5 \
      progress_every=100
  fi
  WARM_ENERGY_SHA="$(verify_checkpoint "${WARM_ENERGY}")"

  declare -A FORWARD_PATHS=(
    [bridge_only]="${BRIDGE}"
    [cold_energy]="${COLD}"
    [bridge_energy]="${WARM_ENERGY}"
  )
  declare -A FORWARD_KINDS=(
    [bridge_only]="forward_pretrain"
    [cold_energy]="forward"
    [bridge_energy]="forward"
  )
  COMPARE_ARGUMENTS=()

  for arm in bridge_only cold_energy bridge_energy; do
    FWD="${FORWARD_PATHS[${arm}]}"
    KIND="${FORWARD_KINDS[${arm}]}"
    BWD_DIR="${RUN}/${arm}_backward100k"
    BWD="${BWD_DIR}/backward_step_00100000"
    PF="${RUN}/${arm}_pf${NUM_SAMPLES}.npz"
    SDE="${RUN}/${arm}_sde20000.npz"
    EVAL="${RUN}/${arm}_evaluation"

    write_status backward \
      "seed=${seed}" "arm=${arm}" "step=0/100000"
    if [[ ! -d "${BWD}" ]]; then
      "${BIN}/cg-bms-train-backward" \
        "experiment=${EXP}" \
        "forward_controller_kind=${KIND}" \
        "forward_checkpoint=${FWD}" \
        updates=100000 \
        "seed=$((seed + 100))" \
        "output_dir=${BWD_DIR}" \
        progress_every=100
    fi
    BWD_SHA="$(verify_checkpoint "${BWD}")"

    write_status pf_reweight \
      "seed=${seed}" "arm=${arm}" "backward_sha256=${BWD_SHA}" \
      "samples=0/${NUM_SAMPLES}"
    if [[ ! -f "${PF}" ]]; then
      "${BIN}/cg-bms-sample-reweight" \
        "experiment=${EXP}" \
        "forward_controller_kind=${KIND}" \
        "forward_checkpoint=${FWD}" \
        "backward_checkpoint=${BWD}" \
        "seed=$((seed + 200))" \
        "num_samples=${NUM_SAMPLES}" \
        "batch_size=${PF_BATCH_SIZE}" \
        "target_batch_size=${PF_BATCH_SIZE}" \
        progress_every_batches=10 \
        likelihood.dt0=1.0e-3 \
        likelihood.rtol=1.0e-5 \
        likelihood.atol=1.0e-6 \
        likelihood.max_steps=16384 \
        reweight.density_mode=exact_ambient \
        reweight.clip=null \
        reuse_pf_stage=true \
        "output=${PF}"
    fi

    write_status evaluate "seed=${seed}" "arm=${arm}" "samples=${PF}"
    "${BIN}/cg-bms-evaluate" \
      "experiment=${EXP}" \
      "samples=${PF}" \
      "output_dir=${EVAL}" \
      bootstrap=false \
      compat_clip_percentile=null
    COMPARE_ARGUMENTS+=(--archive "${arm}=${PF}")

    if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
      if [[ ! -f "${SDE}" ]]; then
        "${BIN}/cg-bms-sample-sde" \
          "experiment=${EXP}" \
          "controller_kind=${KIND}" \
          "forward_checkpoint=${FWD}" \
          "seed=$((seed + 300))" \
          num_samples=20000 \
          batch_size=1024 \
          progress_every_batches=10 \
          "output=${SDE}"
      fi
      "${BIN}/cg-bms-mb2d-suite" dynamics \
        --sde "${SDE}" \
        --pf "${PF}" \
        --output-dir "${RUN}/${arm}_sde_pf_alignment"
    fi
  done

  write_status compare_arms "seed=${seed}"
  "${BIN}/cg-bms-mb2d-suite" compare \
    "${COMPARE_ARGUMENTS[@]}" \
    --output-dir "${RUN}/arm_comparison"

  if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    "${BIN}/cg-bms-mb2d-suite" sample-size \
      --sample "${RUN}/bridge_energy_pf${NUM_SAMPLES}.npz" \
      --output-dir "${RUN}/sample_size_convergence" \
      --sizes 1000,2000,5000,10000,20000,50000,100000 \
      --repeats 5 \
      --seed "${seed}"
    "${BIN}/cg-bms-mb2d-suite" temperature \
      --sample "${RUN}/bridge_energy_pf${NUM_SAMPLES}.npz" \
      --output-dir "${RUN}/temperature_reweighting" \
      --betas 0.5,0.75,1.0,1.25,1.5
    "${BIN}/cg-bms-validate-pf" \
      "experiment=${EXP}" \
      "forward_checkpoint=${WARM_ENERGY}" \
      "backward_checkpoint=${RUN}/bridge_energy_backward100k/backward_step_00100000" \
      "seed=$((seed + 400))" \
      num_samples=8 \
      "output_dir=${RUN}/pf_jacobian_validation"
  fi
  unset FORWARD_PATHS FORWARD_KINDS
done

write_status complete \
  "seeds=${SEEDS}" \
  "num_samples=${NUM_SAMPLES}" \
  "run_root=${RUN_ROOT}"
printf 'MB2D formal matrix complete: %s\n' "${RUN_ROOT}"
