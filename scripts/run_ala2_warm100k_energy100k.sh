#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
GPU="${GPU:-1}"
NAME="ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3"
RUN_TAG="${RUN_TAG:-warm100k_energy100k_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/${NAME}/${RUN_TAG}}"
WARM_DIR="${RUN_DIR}/warm"
ENERGY_DIR="${RUN_DIR}/forward"
STATUS="${RUN_DIR}/status.txt"

mkdir -p "${RUN_DIR}"

write_status() {
    local phase="$1"
    shift
    {
        printf 'phase=%s\n' "${phase}"
        printf 'pid=%s\n' "$$"
        printf 'gpu=%s\n' "${GPU}"
        printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
        printf '%s\n' "$@"
    } > "${STATUS}.tmp"
    mv "${STATUS}.tmp" "${STATUS}"
}

on_error() {
    local exit_code=$?
    write_status failed \
        "exit_code=${exit_code}" \
        "failed_line=${BASH_LINENO[0]}"
    exit "${exit_code}"
}
trap on_error ERR

cd "${ROOT}"
unset LD_LIBRARY_PATH
unset LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONUNBUFFERED=1

BIN="${ENV_ROOT}/bin"
WARM_CHECKPOINT="${WARM_DIR}/forward_pretrain_step_00100000"
ENERGY_CHECKPOINT="${ENERGY_DIR}/forward_step_00100000"

write_status warm \
    'warm_step=0/100000' \
    'energy_step=0/100000'

"${BIN}/cg-bms-pretrain-bridge" \
    experiment="${NAME}" \
    seed=0 \
    output_dir="${WARM_DIR}" \
    pretrain.updates=100000 \
    pretrain.batch_size=64 \
    pretrain.learning_rate_warmup_updates=5000 \
    pretrain.learning_rate_decay_updates=100000 \
    checkpoint_every=null \
    progress_every=100

WARM_SHA=$("${BIN}/python" -c \
    'from cg_bms_jax.checkpoint.io import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "${WARM_CHECKPOINT}")

write_status energy \
    'warm_step=100000/100000' \
    "warm_checkpoint=${WARM_CHECKPOINT}" \
    "warm_sha256=${WARM_SHA}" \
    'energy_step=0/100000'

"${BIN}/cg-bms-train-forward" \
    experiment="${NAME}" \
    seed=0 \
    resume=null \
    initialize_controller_from="${WARM_CHECKPOINT}" \
    output_dir="${ENERGY_DIR}" \
    experiment.training.outer_iterations=100 \
    experiment.training.gradient_steps=1000 \
    progress_every=100

ENERGY_SHA=$("${BIN}/python" -c \
    'from cg_bms_jax.checkpoint.io import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "${ENERGY_CHECKPOINT}")

write_status complete \
    'warm_step=100000/100000' \
    "warm_checkpoint=${WARM_CHECKPOINT}" \
    "warm_sha256=${WARM_SHA}" \
    'energy_step=100000/100000' \
    "energy_checkpoint=${ENERGY_CHECKPOINT}" \
    "energy_sha256=${ENERGY_SHA}"

printf 'Warm checkpoint: %s\n' "${WARM_CHECKPOINT}"
printf 'Warm SHA256: %s\n' "${WARM_SHA}"
printf 'Energy checkpoint: %s\n' "${ENERGY_CHECKPOINT}"
printf 'Energy SHA256: %s\n' "${ENERGY_SHA}"
