#!/usr/bin/env bash
set -Eeuo pipefail

# Reproduce the established CG-BG MB marginal experiment with three independent
# pure-JAX BMS seeds and formal no-clip PF reweighting.

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0,1,2}"
EXP="${EXP:-mb_cg1d}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/mb_cg1d_three_seed}"
BIN="${ENV_ROOT}/bin"
STATUS="${STATUS:-${RUN_ROOT}/status.txt}"
SUMMARY="${SUMMARY:-1}"

mkdir -p "${RUN_ROOT}"
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

for executable in \
  cg-bms-train-forward \
  cg-bms-train-backward \
  cg-bms-sample-reweight \
  cg-bms-evaluate; do
  if [[ ! -x "${BIN}/${executable}" ]]; then
    echo "Missing executable: ${BIN}/${executable}" >&2
    exit 127
  fi
done
for asset in \
  assets/cache/MB/pmf_b/flow_b/energy_params.pkl \
  assets/cache/MB/pmf_b/flow_b/data.npz \
  assets/cache/MB/pmf_b/flow_ub/data.npz; do
  if [[ ! -f "${asset}" ]]; then
    echo "Missing MB CG1D asset: ${ROOT}/${asset}" >&2
    exit 2
  fi
done

verify_checkpoint() {
  "${BIN}/python" -c \
    'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "$1"
}

for seed in ${SEEDS//,/ }; do
  RUN="${RUN_ROOT}/seed_${seed}"
  FWD="${RUN}/forward/forward_step_00100000"
  BWD="${RUN}/backward/backward_step_00100000"
  PF="${RUN}/samples_and_weights_100000.npz"
  mkdir -p "${RUN}"
  write_status forward "seed=${seed}" "step=0/100000"
  if [[ ! -d "${FWD}" ]]; then
    "${BIN}/cg-bms-train-forward" \
      "experiment=${EXP}" \
      "seed=${seed}" \
      "output_dir=${RUN}/forward" \
      save_checkpoint_interval_outer=10 \
      progress_every=100
  fi
  verify_checkpoint "${FWD}" >/dev/null
  write_status backward "seed=${seed}" "step=0/100000"
  if [[ ! -d "${BWD}" ]]; then
    "${BIN}/cg-bms-train-backward" \
      "experiment=${EXP}" \
      "forward_checkpoint=${FWD}" \
      "seed=$((seed + 100))" \
      updates=100000 \
      "output_dir=${RUN}/backward" \
      progress_every=100
  fi
  verify_checkpoint "${BWD}" >/dev/null
  write_status pf_reweight "seed=${seed}" "samples=0/100000"
  if [[ ! -f "${PF}" ]]; then
    "${BIN}/cg-bms-sample-reweight" \
      "experiment=${EXP}" \
      "forward_checkpoint=${FWD}" \
      "backward_checkpoint=${BWD}" \
      "seed=$((seed + 200))" \
      num_samples=100000 \
      batch_size=1024 \
      target_batch_size=4096 \
      progress_every_batches=10 \
      likelihood.dt0=1.0e-3 \
      likelihood.rtol=1.0e-5 \
      likelihood.atol=1.0e-6 \
      likelihood.max_steps=16384 \
      reweight.density_mode=exact_ambient \
      reweight.clip=null \
      "output=${PF}"
  fi
  write_status evaluate "seed=${seed}" "samples=${PF}"
  "${BIN}/cg-bms-evaluate" \
    "experiment=${EXP}" \
    "samples=${PF}" \
    "output_dir=${RUN}/evaluation" \
    bootstrap=true \
    n_bootstraps=500 \
    compat_clip_percentile=99.0
done

if [[ "${SUMMARY}" == "1" ]]; then
  read -r -a SEED_ARRAY <<<"${SEEDS//,/ }"
  "${BIN}/python" scripts/summarize_mb_cg1d_seeds.py \
    --run-root "${RUN_ROOT}" \
    --seeds "${SEED_ARRAY[@]}" \
    --output "${RUN_ROOT}/three_seed_summary.json"
fi
write_status complete "seeds=${SEEDS}" "run_root=${RUN_ROOT}" "summary=${SUMMARY}"
printf 'MB CG1D three-seed matrix complete: %s\n' "${RUN_ROOT}"
