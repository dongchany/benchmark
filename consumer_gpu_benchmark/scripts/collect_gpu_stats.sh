#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_OUTPUT_DIR="${BENCH_ROOT}/results/raw"

INTERVAL_SEC=1
OUTPUT_FILE=""
DURATION_SEC=0
QUERY_FIELDS="timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,temperature.gpu,power.draw,power.limit"

log() {
  printf '[collect_gpu_stats] %s\n' "$*"
}

usage() {
  cat <<'EOF'
Usage:
  bash consumer_gpu_benchmark/scripts/collect_gpu_stats.sh [options]

Options:
  --output <file>       Output CSV path. If omitted, a timestamped file is created in results/raw/
  --interval <sec>      Sampling interval in seconds (default: 1)
  --duration <sec>      Optional fixed duration. If 0, run until interrupted (default: 0)
  -h, --help            Show this help message

Examples:
  bash consumer_gpu_benchmark/scripts/collect_gpu_stats.sh
  bash consumer_gpu_benchmark/scripts/collect_gpu_stats.sh --interval 2 --duration 120
  bash consumer_gpu_benchmark/scripts/collect_gpu_stats.sh --output ./consumer_gpu_benchmark/results/raw/vllm_gpu.csv

Output columns:
  timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,temperature.gpu,power.draw,power.limit
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output)
        shift
        [[ $# -gt 0 ]] || { log "ERROR: --output requires a value"; exit 1; }
        OUTPUT_FILE="$1"
        ;;
      --interval)
        shift
        [[ $# -gt 0 ]] || { log "ERROR: --interval requires a value"; exit 1; }
        INTERVAL_SEC="$1"
        ;;
      --duration)
        shift
        [[ $# -gt 0 ]] || { log "ERROR: --duration requires a value"; exit 1; }
        DURATION_SEC="$1"
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        log "ERROR: unknown argument: $1"
        usage
        exit 1
        ;;
    esac
    shift
  done
}

validate_number() {
  local value="$1"
  local name="$2"
  if ! [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    log "ERROR: ${name} must be a positive number, got '${value}'"
    exit 1
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: required command not found: $1"
    exit 1
  fi
}

ensure_output_file() {
  mkdir -p "${DEFAULT_OUTPUT_DIR}"
  if [[ -z "${OUTPUT_FILE}" ]]; then
    OUTPUT_FILE="${DEFAULT_OUTPUT_DIR}/gpu_stats_$(date +%Y%m%d_%H%M%S).csv"
  fi
  mkdir -p "$(dirname -- "${OUTPUT_FILE}")"
}

write_header() {
  if [[ ! -f "${OUTPUT_FILE}" || ! -s "${OUTPUT_FILE}" ]]; then
    printf '%s\n' "${QUERY_FIELDS}" >"${OUTPUT_FILE}"
  fi
}

sample_once() {
  nvidia-smi \
    --query-gpu="${QUERY_FIELDS}" \
    --format=csv,noheader,nounits >>"${OUTPUT_FILE}"
}

main() {
  parse_args "$@"
  require_cmd nvidia-smi
  validate_number "${INTERVAL_SEC}" "interval"
  validate_number "${DURATION_SEC}" "duration"
  ensure_output_file
  write_header

  log "Writing GPU samples to ${OUTPUT_FILE}"
  log "Sampling interval: ${INTERVAL_SEC}s"

  if [[ "${DURATION_SEC}" == "0" || "${DURATION_SEC}" == "0.0" ]]; then
    log "Running until interrupted..."
    while true; do
      sample_once
      sleep "${INTERVAL_SEC}"
    done
  else
    local_end_time=$(python3 - <<PY
import time
print(time.time() + float(${DURATION_SEC}))
PY
)
    while true; do
      should_continue="$(python3 - <<PY
import time
print(1 if time.time() < float(${local_end_time}) else 0)
PY
)"
      if [[ "${should_continue}" != "1" ]]; then
        break
      fi
      sample_once
      sleep "${INTERVAL_SEC}"
    done
    log "Completed fixed-duration GPU sampling."
  fi
}

main "$@"
