#!/usr/bin/env bash
set -Eeuo pipefail

GPU="${1:?GPU index is required}"
EXPERIMENT="${2:?Hydra experiment name is required}"
TAG="${3:?unique run tag is required}"
TOTAL_STEPS="${4:-100}"
NUM_SAMPLES="${5:-10000}"

ROOT=/ds/project/weilong/ke/cg-bms-jax
ENV_ROOT=/ds/project/weilong/ke/miniconda3
RUN_DIR="${ROOT}/outputs/${EXPERIMENT}/${TAG}"
STATUS="${RUN_DIR}/status.txt"
REFERENCE="${ROOT}/assets/cache/Ac-Ala-NHMe/implicit/data.npz"

mkdir -p "${RUN_DIR}"

write_status() {
  printf 'phase=%s\ntime=%s\ngpu=%s\nexperiment=%s\ntotal_steps=%s\nnum_samples=%s\nrun_dir=%s\n' \
    "$1" "$(date --iso-8601=seconds)" "${GPU}" "${EXPERIMENT}" \
    "${TOTAL_STEPS}" "${NUM_SAMPLES}" "${RUN_DIR}" \
    > "${STATUS}"
}

on_error() {
  rc=$?
  printf 'phase=failed\ntime=%s\nexit_code=%s\ngpu=%s\nexperiment=%s\ntotal_steps=%s\nnum_samples=%s\nrun_dir=%s\n' \
    "$(date --iso-8601=seconds)" "${rc}" "${GPU}" "${EXPERIMENT}" \
    "${TOTAL_STEPS}" "${NUM_SAMPLES}" "${RUN_DIR}" \
    > "${STATUS}"
  exit "${rc}"
}
trap on_error ERR

source "${ENV_ROOT}/etc/profile.d/conda.sh"
conda activate cg-bms-jax
cd "${ROOT}"

unset LD_LIBRARY_PATH
unset LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONUNBUFFERED=1

write_status training
cg-bms-train-forward \
  "experiment=${EXPERIMENT}" \
  "output_dir=${RUN_DIR}/forward" \
  progress_every=10

printf -v STEP_LABEL '%08d' "${TOTAL_STEPS}"
CHECKPOINT="${RUN_DIR}/forward/forward_step_${STEP_LABEL}"
python -c "from cg_bms_jax.checkpoint import verify_checkpoint; print(verify_checkpoint('${CHECKPOINT}'))"

write_status sampling
PROPOSAL="${RUN_DIR}/proposal_${NUM_SAMPLES}.npz"
cg-bms-sample-sde \
  "experiment=${EXPERIMENT}" \
  "forward_checkpoint=${CHECKPOINT}" \
  seed=17 \
  "num_samples=${NUM_SAMPLES}" \
  batch_size=256 \
  "output=${PROPOSAL}"

write_status audit
python scripts/audit_ala2_proposal.py \
  "${PROPOSAL}" \
  --reference "${REFERENCE}" \
  --reference-variant implicit \
  --reference-units auto \
  --output "${RUN_DIR}/proposal_geometry_audit.json" \
  --no-fail

write_status evaluation
cg-bms-evaluate \
  "experiment=${EXPERIMENT}" \
  "samples=${PROPOSAL}" \
  "output_dir=${RUN_DIR}/evaluation" \
  "implicit_reference_override=${REFERENCE}" \
  bootstrap=false \
  compat_clip_percentile=null

write_status complete
printf 'Completed %s on GPU %s\n' "${EXPERIMENT}" "${GPU}"
