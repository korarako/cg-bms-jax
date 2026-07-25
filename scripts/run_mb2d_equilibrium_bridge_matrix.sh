#!/usr/bin/env bash
set -Eeuo pipefail

# Exact-equilibrium MB2D endpoint matrix:
#
#   equilibrium bridge-only  -> backward -> PF/logq -> no-clip reweight/eval
#   equilibrium bridge -> Energy-BMS -> backward -> PF/logq -> no-clip eval
#
# The legacy scripts/run_mb2d_formal_matrix.sh and its outputs are intentionally
# untouched: they remain the deliberately biased-endpoint stress test.
# Cold Energy-BMS is data-free and is also not retrained here; reuse the frozen
# cold arm from the legacy matrix when assembling the release comparison.
#
# Prerequisite (performed explicitly, never inside this formal runner):
#
#   cg-bms-generate-mb2d-equilibrium \
#     output=data/mb2d_equilibrium_exact_v1/endpoints.npz
#
# The release config pins the resulting dataset digest. The runner verifies
# both the archive and its accepted manifest before starting any training.
#
#   bash scripts/run_mb2d_equilibrium_bridge_matrix.sh
#
# Environment overrides:
#   ROOT, ENV_ROOT, GPU, SEEDS ("0 1 2" or "0,1,2"), NUM_SAMPLES,
#   PF_BATCH_SIZE, RUN_DIAGNOSTICS, RUN_POOLED, EXP, RUN_ROOT, STATUS.

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
PF_BATCH_SIZE="${PF_BATCH_SIZE:-1024}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
RUN_POOLED="${RUN_POOLED:-1}"
EXP="${EXP:-mb2d_analytic_equilibrium_bridge}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/mb2d_equilibrium_bridge_matrix_v1}"
BIN="${ENV_ROOT}/bin"
STATUS="${STATUS:-${RUN_ROOT}/status.txt}"
DATASET_RELATIVE_PATH="data/mb2d_equilibrium_exact_v1/endpoints.npz"
DATASET="${ROOT}/${DATASET_RELATIVE_PATH}"
DATASET_MANIFEST="${ROOT}/data/mb2d_equilibrium_exact_v1/endpoints.manifest.json"
FROZEN_DATASET_SHA256="f4c43d8feff619102e5cc1f8ad2dbd422d9a201453d356d4f59e48cb9032347d"

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
    printf 'experiment=%s\n' "${EXP}"
    printf 'dataset=%s\n' "${DATASET_RELATIVE_PATH}"
    printf 'dataset_sha256=%s\n' "${FROZEN_DATASET_SHA256}"
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

for executable in \
  cg-bms-pretrain-flat-bridge \
  cg-bms-train-forward \
  cg-bms-train-backward \
  cg-bms-sample-reweight \
  cg-bms-evaluate \
  cg-bms-mb2d-suite; do
  if [[ ! -x "${BIN}/${executable}" ]]; then
    echo "Missing executable: ${BIN}/${executable}" >&2
    exit 127
  fi
done

if [[ ! -f "${DATASET}" ]]; then
  echo "Missing exact MB2D endpoint dataset: ${DATASET}" >&2
  echo "Generate and validate it before launching the formal matrix." >&2
  exit 2
fi
if [[ ! -f "${DATASET_MANIFEST}" ]]; then
  echo "Missing exact MB2D endpoint manifest: ${DATASET_MANIFEST}" >&2
  exit 2
fi
ACTUAL_DATASET_SHA256="$(sha256sum "${DATASET}" | awk '{print $1}')"
if [[ "${ACTUAL_DATASET_SHA256,,}" != "${FROZEN_DATASET_SHA256}" ]]; then
  echo "Exact MB2D endpoint dataset checksum mismatch." >&2
  echo "expected=${FROZEN_DATASET_SHA256}" >&2
  echo "actual=${ACTUAL_DATASET_SHA256}" >&2
  exit 2
fi
"${BIN}/python" -c \
  'import json,sys; m=json.load(open(sys.argv[1], encoding="utf-8")); expected=sys.argv[2]; assert m["dataset_sha256"] == expected; assert m["validation"]["accepted"] is True' \
  "${DATASET_MANIFEST}" "${FROZEN_DATASET_SHA256}"

verify_checkpoint() {
  "${BIN}/python" -c \
    'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "$1"
}

declare -a POOL_BRIDGE_ARGUMENTS=()
declare -a POOL_ENERGY_ARGUMENTS=()

for seed in ${SEEDS//,/ }; do
  RUN="${RUN_ROOT}/seed_${seed}"
  BRIDGE_DIR="${RUN}/equilibrium_bridge"
  ENERGY_DIR="${RUN}/equilibrium_bridge_energy_forward"
  BRIDGE="${BRIDGE_DIR}/forward_pretrain_step_00100000"
  ENERGY="${ENERGY_DIR}/forward_step_00100000"
  mkdir -p "${RUN}"

  write_status bridge_pretrain \
    "seed=${seed}" "arm=equilibrium_bridge_only" "step=0/100000"
  if [[ ! -d "${BRIDGE}" ]]; then
    "${BIN}/cg-bms-pretrain-flat-bridge" \
      "experiment=${EXP}" \
      "seed=${seed}" \
      "output_dir=${BRIDGE_DIR}" \
      progress_every=100 \
      checkpoint_every=10000
  fi
  BRIDGE_SHA="$(verify_checkpoint "${BRIDGE}")"

  write_status bridge_energy_forward \
    "seed=${seed}" "arm=equilibrium_bridge_energy" \
    "bridge_sha256=${BRIDGE_SHA}" "step=0/100000"
  if [[ ! -d "${ENERGY}" ]]; then
    "${BIN}/cg-bms-train-forward" \
      "experiment=${EXP}" \
      "seed=${seed}" \
      "initialize_controller_from=${BRIDGE}" \
      "output_dir=${ENERGY_DIR}" \
      save_checkpoint_interval_outer=5 \
      progress_every=100
  fi
  ENERGY_SHA="$(verify_checkpoint "${ENERGY}")"

  declare -A FORWARD_PATHS=(
    [equilibrium_bridge_only]="${BRIDGE}"
    [equilibrium_bridge_energy]="${ENERGY}"
  )
  declare -A FORWARD_KINDS=(
    [equilibrium_bridge_only]="forward_pretrain"
    [equilibrium_bridge_energy]="forward"
  )
  declare -A FORWARD_SHAS=(
    [equilibrium_bridge_only]="${BRIDGE_SHA}"
    [equilibrium_bridge_energy]="${ENERGY_SHA}"
  )
  COMPARE_ARGUMENTS=()

  for arm in equilibrium_bridge_only equilibrium_bridge_energy; do
    FWD="${FORWARD_PATHS[${arm}]}"
    KIND="${FORWARD_KINDS[${arm}]}"
    FWD_SHA="${FORWARD_SHAS[${arm}]}"
    BWD_DIR="${RUN}/${arm}_backward100k"
    BWD="${BWD_DIR}/backward_step_00100000"
    PF="${RUN}/${arm}_pf${NUM_SAMPLES}.npz"
    SDE="${RUN}/${arm}_sde20000.npz"
    EVAL="${RUN}/${arm}_evaluation"

    write_status backward \
      "seed=${seed}" "arm=${arm}" "forward_sha256=${FWD_SHA}" \
      "step=0/100000"
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
    # A same-name archive from an interrupted or older run must never be
    # silently relabelled as the current equilibrium-data result.
    "${BIN}/python" scripts/validate_pf_archive.py \
      --archive "${PF}" \
      --forward-sha256 "${FWD_SHA}" \
      --backward-sha256 "${BWD_SHA}" \
      --forward-controller-kind "${KIND}" \
      --seed "$((seed + 200))" \
      --num-samples "${NUM_SAMPLES}" \
      --density-mode ambient_exact \
      --endpoint-distribution equilibrium_endpoint_npz_v1 \
      --likelihood-dt0 1.0e-3 \
      --likelihood-rtol 1.0e-5 \
      --likelihood-atol 1.0e-6 \
      --likelihood-max-steps 16384

    write_status evaluate "seed=${seed}" "arm=${arm}" "samples=${PF}"
    "${BIN}/cg-bms-evaluate" \
      "experiment=${EXP}" \
      "samples=${PF}" \
      "output_dir=${EVAL}" \
      bootstrap=true \
      n_bootstraps=500 \
      compat_clip_percentile=99.0
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

  POOL_BRIDGE_ARGUMENTS+=(
    --archive "seed${seed}=${RUN}/equilibrium_bridge_only_pf${NUM_SAMPLES}.npz"
  )
  POOL_ENERGY_ARGUMENTS+=(
    --archive "seed${seed}=${RUN}/equilibrium_bridge_energy_pf${NUM_SAMPLES}.npz"
  )
  unset FORWARD_PATHS FORWARD_KINDS FORWARD_SHAS
done

if [[ "${RUN_POOLED}" == "1" ]]; then
  write_status pooled_visualization "seeds=${SEEDS}" "samples_per_seed=${NUM_SAMPLES}"
  "${BIN}/cg-bms-mb2d-suite" pool \
    "${POOL_BRIDGE_ARGUMENTS[@]}" \
    --output-dir "${RUN_ROOT}/pooled/equilibrium_bridge_only" \
    --bins 160 \
    --energy-bins 240
  "${BIN}/cg-bms-mb2d-suite" pool \
    "${POOL_ENERGY_ARGUMENTS[@]}" \
    --output-dir "${RUN_ROOT}/pooled/equilibrium_bridge_energy" \
    --bins 160 \
    --energy-bins 240
fi

write_status complete \
  "seeds=${SEEDS}" \
  "num_samples=${NUM_SAMPLES}" \
  "pooled=${RUN_POOLED}" \
  "run_root=${RUN_ROOT}"
printf 'MB2D equilibrium bridge matrix complete: %s\n' "${RUN_ROOT}"
