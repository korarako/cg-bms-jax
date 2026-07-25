#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
GPU="${GPU:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
NAME="ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3"
SOURCE_RUN="${ROOT}/outputs/${NAME}/energy_from_warm10k_formal20k_20260719_2132"
RUN_TAG="${RUN_TAG:-pf${NUM_SAMPLES}_eval_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${SOURCE_RUN}/${RUN_TAG}"
STATUS="${RUN_DIR}/status.txt"

FWD="${SOURCE_RUN}/forward/forward_step_00020000"
BWD="${SOURCE_RUN}/backward_formal20k_run1/backward_step_00020000"
SAMPLES="${RUN_DIR}/samples_and_weights_${NUM_SAMPLES}.npz"
EVALUATION="${RUN_DIR}/evaluation"

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
FWD_SHA=$("${BIN}/python" -c \
    'from cg_bms_jax.checkpoint.io import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "${FWD}")
BWD_SHA=$("${BIN}/python" -c \
    'from cg_bms_jax.checkpoint.io import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "${BWD}")

write_status sampling \
    "samples=0/${NUM_SAMPLES}" \
    "forward_sha256=${FWD_SHA}" \
    "backward_sha256=${BWD_SHA}"

"${BIN}/cg-bms-sample-reweight" \
    experiment="${NAME}" \
    experiment.training.outer_iterations=20 \
    experiment.training.gradient_steps=1000 \
    forward_checkpoint="${FWD}" \
    backward_checkpoint="${BWD}" \
    seed=3 \
    num_samples="${NUM_SAMPLES}" \
    batch_size=16 \
    target_batch_size=16 \
    reuse_pf_stage=true \
    progress_every_batches=10 \
    ++likelihood.dt0=0.005 \
    likelihood.rtol=1.0e-5 \
    likelihood.atol=1.0e-5 \
    likelihood.max_steps=16384 \
    likelihood.divergence=exact \
    reweight.density_mode=exact_ambient \
    output="${SAMPLES}"

write_status evaluating \
    "samples=${NUM_SAMPLES}/${NUM_SAMPLES}" \
    "samples_path=${SAMPLES}" \
    "forward_sha256=${FWD_SHA}" \
    "backward_sha256=${BWD_SHA}"

"${BIN}/cg-bms-evaluate" \
    experiment="${NAME}" \
    samples="${SAMPLES}" \
    output_dir="${EVALUATION}" \
    bootstrap=true \
    n_bootstraps=500 \
    compat_clip_percentile=99.0

write_status complete \
    "samples=${NUM_SAMPLES}/${NUM_SAMPLES}" \
    "samples_path=${SAMPLES}" \
    "evaluation_path=${EVALUATION}" \
    "forward_sha256=${FWD_SHA}" \
    "backward_sha256=${BWD_SHA}"

printf 'Samples: %s\n' "${SAMPLES}"
printf 'Evaluation: %s\n' "${EVALUATION}"
