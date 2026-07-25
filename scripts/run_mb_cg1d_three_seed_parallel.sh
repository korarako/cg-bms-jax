#!/usr/bin/env bash
set -Eeuo pipefail

# Run the formal MB CG1D three-seed matrix on two GPUs without allowing the
# per-worker summary writers to race.  The single final summary is produced
# only after both workers finish successfully.

ROOT="${ROOT:-/ds/project/weilong/ke/cg-bms-jax}"
ENV_ROOT="${ENV_ROOT:-/ds/project/weilong/ke/miniconda3/envs/cg-bms-jax}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/mb_cg1d_three_seed_parallel}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-2}"
SEEDS_A="${SEEDS_A:-0,2}"
SEEDS_B="${SEEDS_B:-1}"
ALL_SEEDS="${ALL_SEEDS:-0,1,2}"
STATUS="${STATUS:-${RUN_ROOT}/parallel_status.txt}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
BIN="${ENV_ROOT}/bin"
RUN_TAG="${RUN_TAG:-$(basename "${RUN_ROOT}")}"
LOCK_DIR="${RUN_ROOT}.parallel.lock"
PID_A=""
PID_B=""

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
cd "${ROOT}"
unset LD_LIBRARY_PATH
unset LIBRARY_PATH
export PATH="${BIN}:${PATH}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

if [[ ! -x "${BIN}/python" ]]; then
  echo "Missing Python executable: ${BIN}/python" >&2
  exit 127
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "Missing required executable: setsid" >&2
  exit 127
fi
"${BIN}/python" -c '
import sys

def parse(text):
    values = text.replace(",", " ").split()
    if not values or any(not value.isdigit() for value in values):
        raise SystemExit(f"Invalid seed list: {text!r}")
    numbers = [int(value) for value in values]
    if len(numbers) != len(set(numbers)):
        raise SystemExit(f"Seed list contains duplicates: {text!r}")
    return numbers

left, right, expected = map(parse, sys.argv[1:])
if set(left) & set(right):
    raise SystemExit("SEEDS_A and SEEDS_B must be disjoint")
if set(left) | set(right) != set(expected):
    raise SystemExit("Worker seed union must equal ALL_SEEDS")
' "${SEEDS_A}" "${SEEDS_B}" "${ALL_SEEDS}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another parallel matrix owns lock: ${LOCK_DIR}" >&2
  exit 73
fi

cleanup_lock() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup_lock EXIT

write_status() {
  local phase="$1"
  shift
  {
    printf 'phase=%s\n' "${phase}"
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'pid=%s\n' "$$"
    printf 'run_root=%s\n' "${RUN_ROOT}"
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

on_signal() {
  local signal="$1"
  trap - ERR INT TERM
  for pid in "${PID_A:-}" "${PID_B:-}"; do
    if [[ -n "${pid}" ]]; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${PID_A:-}" "${PID_B:-}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  write_status cancelled "signal=${signal}"
  exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

launch_worker() {
  local label="$1"
  local gpu="$2"
  local seeds="$3"
  local worker_status="${RUN_ROOT}/${label}_status.txt"
  local worker_log="${LOG_DIR}/${RUN_TAG}_${label}.log"
  setsid env \
    ROOT="${ROOT}" \
    ENV_ROOT="${ENV_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    GPU="${gpu}" \
    SEEDS="${seeds}" \
    STATUS="${worker_status}" \
    SUMMARY=0 \
    bash scripts/run_mb_cg1d_three_seed.sh \
    >"${worker_log}" 2>&1 &
  LAST_PID="$!"
}

write_status starting \
  "gpu_a=${GPU_A}" "seeds_a=${SEEDS_A}" \
  "gpu_b=${GPU_B}" "seeds_b=${SEEDS_B}"

launch_worker worker_a "${GPU_A}" "${SEEDS_A}"
PID_A="${LAST_PID}"
launch_worker worker_b "${GPU_B}" "${SEEDS_B}"
PID_B="${LAST_PID}"
printf '%s\n' "${PID_A}" >"${RUN_ROOT}/worker_a.pid"
printf '%s\n' "${PID_B}" >"${RUN_ROOT}/worker_b.pid"
write_status running \
  "worker_a_pid=${PID_A}" "worker_a_gpu=${GPU_A}" "worker_a_seeds=${SEEDS_A}" \
  "worker_a_status=${RUN_ROOT}/worker_a_status.txt" \
  "worker_a_log=${LOG_DIR}/${RUN_TAG}_worker_a.log" \
  "worker_b_pid=${PID_B}" "worker_b_gpu=${GPU_B}" "worker_b_seeds=${SEEDS_B}" \
  "worker_b_status=${RUN_ROOT}/worker_b_status.txt" \
  "worker_b_log=${LOG_DIR}/${RUN_TAG}_worker_b.log"

set +e
wait "${PID_A}"
CODE_A=$?
wait "${PID_B}"
CODE_B=$?
set -e
if [[ "${CODE_A}" -ne 0 || "${CODE_B}" -ne 0 ]]; then
  write_status failed \
    "worker_a_pid=${PID_A}" "worker_a_exit=${CODE_A}" \
    "worker_b_pid=${PID_B}" "worker_b_exit=${CODE_B}"
  exit 1
fi

read -r -a SEED_ARRAY <<<"${ALL_SEEDS//,/ }"
"${BIN}/python" scripts/summarize_mb_cg1d_seeds.py \
  --run-root "${RUN_ROOT}" \
  --seeds "${SEED_ARRAY[@]}" \
  --output "${RUN_ROOT}/three_seed_summary.json"

rm -f "${RUN_ROOT}/worker_a.pid" "${RUN_ROOT}/worker_b.pid"
write_status complete \
  "worker_a_exit=${CODE_A}" "worker_b_exit=${CODE_B}" \
  "summary=${RUN_ROOT}/three_seed_summary.json"
printf 'MB CG1D parallel three-seed matrix complete: %s\n' "${RUN_ROOT}"
