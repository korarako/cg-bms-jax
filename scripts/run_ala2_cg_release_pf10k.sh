#!/usr/bin/env bash
set -Eeuo pipefail

# Recompute the strongest frozen CG Ala2 positive arm with the release PF-ODE
# tolerances.  Training is not repeated: both verified checkpoints and their
# fully resolved compatibility config remain immutable.

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
GPU="${GPU:-0}"
BIN="${ENV_ROOT}/bin"
EXP="${EXP:-ala2_cg_legacy_positive_release}"
BASE="${ROOT}/outputs/ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3"
RUN="${BASE}/energy_from_warm10k_formal20k_20260719_2132"
FWD="${RUN}/forward/forward_step_00020000"
BWD="${RUN}/backward_formal20k_run1/backward_step_00020000"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN}/pf10k_release_tol_20260726}"
PF="${OUTPUT_ROOT}/samples_and_weights_10000.npz"
EVALUATION="${OUTPUT_ROOT}/evaluation"
STATUS="${OUTPUT_ROOT}/status.txt"

mkdir -p "${OUTPUT_ROOT}" "${ROOT}/logs"
cd "${ROOT}"
unset LD_LIBRARY_PATH
unset LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONUNBUFFERED=1
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

FWD_SHA="$(
  "${BIN}/python" -c \
    'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "${FWD}"
)"
BWD_SHA="$(
  "${BIN}/python" -c \
    'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "${BWD}"
)"

write_status pf_reweight \
  "forward_sha256=${FWD_SHA}" \
  "backward_sha256=${BWD_SHA}" \
  "samples=0/10000"

if [[ ! -f "${PF}" ]]; then
  "${BIN}/cg-bms-sample-reweight" \
    "experiment=${EXP}" \
    forward_controller_kind=forward \
    "forward_checkpoint=${FWD}" \
    "backward_checkpoint=${BWD}" \
    seed=3 \
    num_samples=10000 \
    batch_size=16 \
    target_batch_size=64 \
    progress_every_batches=10 \
    likelihood.dt0=1.0e-3 \
    likelihood.rtol=1.0e-5 \
    likelihood.atol=1.0e-6 \
    likelihood.max_steps=16384 \
    likelihood.divergence=exact \
    reweight.density_mode=exact_ambient \
    reweight.clip=null \
    reuse_pf_stage=true \
    "output=${PF}"
fi

write_status evaluate "samples=${PF}"
"${BIN}/cg-bms-evaluate" \
  "experiment=${EXP}" \
  "samples=${PF}" \
  "output_dir=${EVALUATION}" \
  bootstrap=true \
  n_bootstraps=500 \
  compat_clip_percentile=99.0

write_status complete \
  "forward_sha256=${FWD_SHA}" \
  "backward_sha256=${BWD_SHA}" \
  "samples=${PF}" \
  "evaluation=${EVALUATION}"
printf 'CG Ala2 release PF/evaluation complete: %s\n' "${OUTPUT_ROOT}"
