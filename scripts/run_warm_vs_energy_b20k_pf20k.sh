#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
GPU="${GPU:-0}"
BIN="${ENV_ROOT}/bin"
EXP=ala2_ambient18_300k_bms_eta10_official_cadence_additive_v3
RUN="${ROOT}/outputs/ala2_bms_official_cadence_additive_v3/warm10k_then_e20k_s0_run1"
STATUS="${RUN}/b20k_pf20k_comparison_status.txt"

WARM="${RUN}/warm/forward_pretrain_step_00010000"
FWD="${RUN}/forward/forward_step_00020000"
B_W_DIR="${RUN}/backward_warm10k_b20k"
B_WE_DIR="${RUN}/backward_warm10k_e20k_b20k"
B_W="${B_W_DIR}/backward_step_00020000"
B_WE="${B_WE_DIR}/backward_step_00020000"

PF_W="${RUN}/pf_reweight_warm10k_20k.npz"
PF_WE="${RUN}/pf_reweight_warm10k_e20k_20k.npz"
EVAL_W="${RUN}/evaluation_pf_reweight_warm10k_20k"
EVAL_WE="${RUN}/evaluation_pf_reweight_warm10k_e20k_20k"
COMPARISON="${RUN}/pf_reweight_warm_vs_energy_comparison.json"

mkdir -p "${RUN}" "${ROOT}/logs"

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

test -d "${WARM}"
test -d "${FWD}"
WARM_SHA=$("${BIN}/python" -c \
  'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
  "${WARM}")
FWD_SHA=$("${BIN}/python" -c \
  'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
  "${FWD}")

write_status backward_we \
  "warm_sha256=${WARM_SHA}" \
  "forward_sha256=${FWD_SHA}" \
  'step=0/20000'
"${BIN}/cg-bms-train-backward" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  experiment.training.gradient_steps=100 \
  forward_controller_kind=forward \
  "forward_checkpoint=${FWD}" \
  seed=1 \
  progress_every=100 \
  "output_dir=${B_WE_DIR}"

B_WE_SHA=$("${BIN}/python" -c \
  'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
  "${B_WE}")

write_status backward_w \
  "backward_we_sha256=${B_WE_SHA}" \
  'step=0/20000'
"${BIN}/cg-bms-train-backward" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  experiment.training.gradient_steps=100 \
  forward_controller_kind=forward_pretrain \
  "forward_checkpoint=${WARM}" \
  seed=1 \
  progress_every=100 \
  "output_dir=${B_W_DIR}"

B_W_SHA=$("${BIN}/python" -c \
  'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
  "${B_W}")

# Four-sample formal-tolerance probes catch stiffness before the expensive PF20k runs.
write_status pf_smoke_w "backward_w_sha256=${B_W_SHA}"
"${BIN}/cg-bms-sample-reweight" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  experiment.training.gradient_steps=100 \
  forward_controller_kind=forward_pretrain \
  "forward_checkpoint=${WARM}" \
  "backward_checkpoint=${B_W}" \
  seed=3 num_samples=4 batch_size=1 target_batch_size=1 \
  progress_every_batches=1 \
  ++likelihood.dt0=0.001 \
  likelihood.rtol=1.0e-5 likelihood.atol=1.0e-6 \
  likelihood.max_steps=16384 likelihood.divergence=exact \
  reweight.density_mode=exact_ambient \
  "output=${RUN}/pf_smoke_warm10k_4.npz"

write_status pf_smoke_we "backward_we_sha256=${B_WE_SHA}"
"${BIN}/cg-bms-sample-reweight" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  experiment.training.gradient_steps=100 \
  forward_controller_kind=forward \
  "forward_checkpoint=${FWD}" \
  "backward_checkpoint=${B_WE}" \
  seed=3 num_samples=4 batch_size=1 target_batch_size=1 \
  progress_every_batches=1 \
  ++likelihood.dt0=0.001 \
  likelihood.rtol=1.0e-5 likelihood.atol=1.0e-6 \
  likelihood.max_steps=16384 likelihood.divergence=exact \
  reweight.density_mode=exact_ambient \
  "output=${RUN}/pf_smoke_warm10k_e20k_4.npz"

write_status pf_w \
  "backward_w_sha256=${B_W_SHA}" \
  'samples=0/20000'
"${BIN}/cg-bms-sample-reweight" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  experiment.training.gradient_steps=100 \
  forward_controller_kind=forward_pretrain \
  "forward_checkpoint=${WARM}" \
  "backward_checkpoint=${B_W}" \
  seed=3 num_samples=20000 batch_size=16 target_batch_size=16 \
  progress_every_batches=10 \
  ++likelihood.dt0=0.001 \
  likelihood.rtol=1.0e-5 likelihood.atol=1.0e-6 \
  likelihood.max_steps=16384 likelihood.divergence=exact \
  reweight.density_mode=exact_ambient reweight.clip=null \
  reuse_pf_stage=true \
  "output=${PF_W}"

write_status pf_we \
  "backward_we_sha256=${B_WE_SHA}" \
  'samples=0/20000'
"${BIN}/cg-bms-sample-reweight" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  experiment.training.gradient_steps=100 \
  forward_controller_kind=forward \
  "forward_checkpoint=${FWD}" \
  "backward_checkpoint=${B_WE}" \
  seed=3 num_samples=20000 batch_size=16 target_batch_size=16 \
  progress_every_batches=10 \
  ++likelihood.dt0=0.001 \
  likelihood.rtol=1.0e-5 likelihood.atol=1.0e-6 \
  likelihood.max_steps=16384 likelihood.divergence=exact \
  reweight.density_mode=exact_ambient reweight.clip=null \
  reuse_pf_stage=true \
  "output=${PF_WE}"

write_status eval_w "samples=${PF_W}"
"${BIN}/cg-bms-evaluate" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  experiment.training.gradient_steps=100 \
  "samples=${PF_W}" \
  "output_dir=${EVAL_W}" \
  bootstrap=true n_bootstraps=500

write_status eval_we "samples=${PF_WE}"
"${BIN}/cg-bms-evaluate" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  experiment.training.gradient_steps=100 \
  "samples=${PF_WE}" \
  "output_dir=${EVAL_WE}" \
  bootstrap=true n_bootstraps=500

write_status compare
"${BIN}/python" scripts/compare_pf_reweight_arms.py \
  --warm "${PF_W}" \
  --warm-energy "${PF_WE}" \
  --warm-metrics "${EVAL_W}/formal/ala2_cb_metrics.json" \
  --warm-energy-metrics "${EVAL_WE}/formal/ala2_cb_metrics.json" \
  --output "${COMPARISON}"

write_status complete \
  "warm_sha256=${WARM_SHA}" \
  "forward_sha256=${FWD_SHA}" \
  "backward_w_sha256=${B_W_SHA}" \
  "backward_we_sha256=${B_WE_SHA}" \
  "pf_w=${PF_W}" \
  "pf_we=${PF_WE}" \
  "comparison=${COMPARISON}"
printf 'comparison pipeline complete: %s\n' "${COMPARISON}"
