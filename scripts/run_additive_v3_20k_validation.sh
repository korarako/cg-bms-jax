#!/usr/bin/env bash
set -Eeuo pipefail

GPU="${1:-1}"
TAG="${2:-direct_D_20k_$(date +%Y%m%d_%H%M%S)}"

ROOT=/ds/project/weilong/ke/cg-bms-jax
ENV_ROOT=/ds/project/weilong/ke/miniconda3
BIN="${ENV_ROOT}/envs/cg-bms-jax/bin"
EXP=ala2_ambient18_300k_bms_eta10_s1_canonical_additive_v3
RUN="${ROOT}/outputs/${EXP}/${TAG}"
STATUS="${RUN}/status.txt"
REFERENCE="${ROOT}/assets/cache/Ac-Ala-NHMe/explicit/core_beta/pmf_b/flow_ub/data.npz"
FWD="${RUN}/forward/forward_step_00020000"
PROPOSAL="${RUN}/proposal_sde_20000.npz"

mkdir -p "${RUN}/forward" "${RUN}/evaluation"

write_status() {
  {
    printf 'phase=%s\n' "$1"
    printf 'time=%s\n' "$(date --iso-8601=seconds)"
    printf 'host=%s\n' "$(hostname)"
    printf 'gpu=%s\n' "${GPU}"
    printf 'experiment=%s\n' "${EXP}"
    printf 'run=%s\n' "${RUN}"
    printf 'checkpoint=%s\n' "${FWD}"
    printf 'proposal=%s\n' "${PROPOSAL}"
  } >"${STATUS}"
}

on_exit() {
  code=$?
  if [[ ${code} -ne 0 ]]; then
    {
      printf 'phase=failed\n'
      printf 'time=%s\n' "$(date --iso-8601=seconds)"
      printf 'exit_code=%s\n' "${code}"
      printf 'host=%s\n' "$(hostname)"
      printf 'gpu=%s\n' "${GPU}"
      printf 'experiment=%s\n' "${EXP}"
      printf 'run=%s\n' "${RUN}"
    } >"${STATUS}"
  fi
}
trap on_exit EXIT

source "${ENV_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_ROOT}/envs/cg-bms-jax"
cd "${ROOT}"

unset LD_LIBRARY_PATH
unset LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONUNBUFFERED=1
export PATH="${BIN}:${PATH}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

test -f "${REFERENCE}"

{
  printf 'started=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'physical_gpu=%s\n' "${GPU}"
  printf 'experiment=%s\n' "${EXP}"
  printf 'target=U_topo+U_CB+U_PMF\n'
  printf 'outer_iterations=200\n'
  printf 'updates_per_outer=100\n'
  printf 'total_updates=20000\n'
  printf 'proposal_samples=20000\n'
  sha256sum "configs/experiment/${EXP}.yaml"
  "${BIN}/python" scripts/check_environment.py
} >"${RUN}/run_metadata.txt"

write_status forward
printf '[1/4] forward start %s\n' "$(date --iso-8601=seconds)"
"${BIN}/cg-bms-train-forward" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  experiment.training.gradient_steps=100 \
  "output_dir=${RUN}/forward" \
  progress_every=100

write_status verify_forward
"${BIN}/python" -c \
  "from cg_bms_jax.checkpoint import verify_checkpoint; print('[verify] forward sha256=' + verify_checkpoint('${FWD}'), flush=True)"

write_status proposal
printf '[2/4] proposal start %s\n' "$(date --iso-8601=seconds)"
"${BIN}/cg-bms-sample-sde" \
  "experiment=${EXP}" \
  experiment.training.outer_iterations=200 \
  experiment.training.gradient_steps=100 \
  "forward_checkpoint=${FWD}" \
  seed=17 \
  num_samples=20000 \
  batch_size=256 \
  progress_every_batches=5 \
  "output=${PROPOSAL}"

write_status audit
printf '[3/4] proposal audit start %s\n' "$(date --iso-8601=seconds)"
"${BIN}/python" scripts/audit_ala2_proposal.py \
  "${PROPOSAL}" \
  --reference "${REFERENCE}" \
  --reference-variant cg \
  --reference-units auto \
  --reference-max-frames 100000 \
  --output "${RUN}/proposal_geometry_audit.json" \
  --no-fail

write_status evaluation
printf '[4/4] evaluation start %s\n' "$(date --iso-8601=seconds)"
"${BIN}/cg-bms-evaluate" \
  "experiment=${EXP}" \
  "samples=${PROPOSAL}" \
  "output_dir=${RUN}/evaluation" \
  bootstrap=false \
  compat_clip_percentile=null

write_status complete
printf 'validation complete %s\n' "$(date --iso-8601=seconds)"
printf 'run=%s\n' "${RUN}"
