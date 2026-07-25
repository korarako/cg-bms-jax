#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/mb2d_equilibrium_bridge_matrix_v1}"
SEEDS="${SEEDS:-0 1 2}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
BIN="${ENV_ROOT}/bin"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

verify_checkpoint() {
  "${BIN}/python" -c \
    'from cg_bms_jax.checkpoint import verify_checkpoint; import sys; print(verify_checkpoint(sys.argv[1]))' \
    "$1"
}

validate_pf() {
  local archive="$1"
  local forward_sha="$2"
  local backward_sha="$3"
  local controller_kind="$4"
  local seed="$5"
  "${BIN}/python" scripts/validate_pf_archive.py \
    --archive "${archive}" \
    --forward-sha256 "${forward_sha}" \
    --backward-sha256 "${backward_sha}" \
    --forward-controller-kind "${controller_kind}" \
    --seed "$((seed + 200))" \
    --num-samples "${NUM_SAMPLES}" \
    --density-mode ambient_exact \
    --endpoint-distribution equilibrium_endpoint_npz_v1 \
    --likelihood-dt0 1.0e-3 \
    --likelihood-rtol 1.0e-5 \
    --likelihood-atol 1.0e-6 \
    --likelihood-max-steps 16384
}

declare -a BRIDGE_ARCHIVES=()
declare -a ENERGY_ARCHIVES=()
declare -a SEED_ARGUMENTS=()
for seed in ${SEEDS//,/ }; do
  RUN="${RUN_ROOT}/seed_${seed}"
  BRIDGE="${RUN}/equilibrium_bridge_only_pf${NUM_SAMPLES}.npz"
  ENERGY="${RUN}/equilibrium_bridge_energy_pf${NUM_SAMPLES}.npz"
  test -s "${BRIDGE}"
  test -s "${ENERGY}"
  test -s "${RUN}/equilibrium_bridge_only_evaluation/formal/mb2d_metrics.json"
  test -s "${RUN}/equilibrium_bridge_energy_evaluation/formal/mb2d_metrics.json"

  BRIDGE_FWD="${RUN}/equilibrium_bridge/forward_pretrain_step_00100000"
  ENERGY_FWD="${RUN}/equilibrium_bridge_energy_forward/forward_step_00100000"
  BRIDGE_BWD="${RUN}/equilibrium_bridge_only_backward100k/backward_step_00100000"
  ENERGY_BWD="${RUN}/equilibrium_bridge_energy_backward100k/backward_step_00100000"
  BRIDGE_FWD_SHA="$(verify_checkpoint "${BRIDGE_FWD}")"
  ENERGY_FWD_SHA="$(verify_checkpoint "${ENERGY_FWD}")"
  BRIDGE_BWD_SHA="$(verify_checkpoint "${BRIDGE_BWD}")"
  ENERGY_BWD_SHA="$(verify_checkpoint "${ENERGY_BWD}")"
  validate_pf \
    "${BRIDGE}" "${BRIDGE_FWD_SHA}" "${BRIDGE_BWD_SHA}" forward_pretrain "${seed}"
  validate_pf \
    "${ENERGY}" "${ENERGY_FWD_SHA}" "${ENERGY_BWD_SHA}" forward "${seed}"

  BRIDGE_ARCHIVES+=(--archive "seed${seed}=${BRIDGE}")
  ENERGY_ARCHIVES+=(--archive "seed${seed}=${ENERGY}")
  SEED_ARGUMENTS+=("${seed}")
done

"${BIN}/cg-bms-mb2d-suite" pool \
  "${BRIDGE_ARCHIVES[@]}" \
  --output-dir "${RUN_ROOT}/pooled/equilibrium_bridge_only" \
  --bins 160 \
  --energy-bins 240

"${BIN}/cg-bms-mb2d-suite" pool \
  "${ENERGY_ARCHIVES[@]}" \
  --output-dir "${RUN_ROOT}/pooled/equilibrium_bridge_energy" \
  --bins 160 \
  --energy-bins 240

"${BIN}/python" scripts/summarize_mb2d_equilibrium_seeds.py \
  --run-root "${RUN_ROOT}" \
  --seeds "${SEED_ARGUMENTS[@]}" \
  --num-samples "${NUM_SAMPLES}" \
  --output "${RUN_ROOT}/three_seed_summary.json"

{
  printf 'phase=complete\n'
  printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'seeds=%s\n' "${SEEDS}"
  printf 'num_samples=%s\n' "${NUM_SAMPLES}"
  printf 'summary=%s\n' "${RUN_ROOT}/three_seed_summary.json"
} >"${RUN_ROOT}/finalize_status.txt"

printf 'MB2D equilibrium matrix finalized: %s\n' "${RUN_ROOT}"
