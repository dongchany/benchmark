#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${BENCH_ROOT}/configs"
LOG_DIR="${BENCH_ROOT}/logs"
RESULTS_DIR="${BENCH_ROOT}/results"
RAW_RESULTS_DIR="${RESULTS_DIR}/raw"
NORMALIZED_RESULTS_DIR="${RESULTS_DIR}/normalized"
REPORT_DIR="${BENCH_ROOT}/reports"
ENV_DIR="${BENCH_ROOT}/envs"
TOOLKIT_ENV_SCRIPT="${SCRIPT_DIR}/toolkit_env.sh"

MODELS_JSON="${CONFIG_DIR}/models.json"
SCENARIOS_JSON="${CONFIG_DIR}/scenarios.json"

VLLM_REPO="/home/dong/workspace/vllm"
SGLANG_REPO="/home/dong/workspace/sglang"

DEFAULT_HOST="127.0.0.1"
DEFAULT_VLLM_PORT="18000"
DEFAULT_SGLANG_PORT="19000"
DEFAULT_SINGLE_GPU="1"
DEFAULT_DUAL_GPU="0,1"

log() {
  printf '[benchmark] %s\n' "$*"
}

warn() {
  printf '[benchmark][warn] %s\n' "$*" >&2
}

die() {
  printf '[benchmark][error] %s\n' "$*" >&2
  exit 1
}

ensure_dir() {
  mkdir -p "$1"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "required directory not found: $1"
}

timestamp() {
  date +%Y%m%d_%H%M%S
}

framework_env_dir() {
  case "$1" in
    vllm) printf '%s\n' "${VLLM_ENV_DIR_OVERRIDE:-${ENV_DIR}/vllm}" ;;
    sglang) printf '%s\n' "${SGLANG_ENV_DIR_OVERRIDE:-${ENV_DIR}/sglang}" ;;
    *) die "unknown framework: $1" ;;
  esac
}

framework_python() {
  local env_dir
  env_dir="$(framework_env_dir "$1")"
  printf '%s\n' "${env_dir}/bin/python"
}

framework_repo() {
  case "$1" in
    vllm) printf '%s\n' "${VLLM_REPO}" ;;
    sglang) printf '%s\n' "${SGLANG_REPO}" ;;
    *) die "unknown framework: $1" ;;
  esac
}

framework_log_dir() {
  case "$1" in
    vllm) printf '%s\n' "${LOG_DIR}/vllm" ;;
    sglang) printf '%s\n' "${LOG_DIR}/sglang" ;;
    *) die "unknown framework: $1" ;;
  esac
}

model_field() {
  local model_id="$1"
  local field_name="$2"
  python3 - <<PY
import json
from pathlib import Path
path = Path(${MODELS_JSON@Q})
with path.open() as f:
    data = json.load(f)
for model in data["models"]:
    if model["id"] == ${model_id@Q}:
        value = model.get(${field_name@Q})
        if isinstance(value, (list, dict)):
            import json as _json
            print(_json.dumps(value))
        elif value is None:
            print("")
        else:
            print(value)
        break
else:
    raise SystemExit(f"model id not found: {${model_id@Q}}")
PY
}

scenario_json() {
  local scenario_id="$1"
  python3 - <<PY
import json
from pathlib import Path
path = Path(${SCENARIOS_JSON@Q})
with path.open() as f:
    data = json.load(f)
for scenario in data["scenarios"]:
    if scenario["id"] == ${scenario_id@Q}:
        print(json.dumps(scenario))
        break
else:
    raise SystemExit(f"scenario id not found: {${scenario_id@Q}}")
PY
}

wait_for_http_ok() {
  local url="$1"
  local timeout_sec="${2:-600}"
  local started_at
  started_at="$(date +%s)"
  while true; do
    if python3 - <<PY
import sys
import urllib.request
url = ${url@Q}
try:
    with urllib.request.urlopen(url, timeout=3) as resp:
        sys.exit(0 if 200 <= resp.status < 300 else 1)
except Exception:
    sys.exit(1)
PY
    then
      log "endpoint ready: ${url}"
      return 0
    fi
    if (( $(date +%s) - started_at >= timeout_sec )); then
      return 1
    fi
    sleep 1
  done
}

find_free_port() {
  python3 - <<'PY'
import socket
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
}

start_gpu_sampler() {
  local output_file="$1"
  local interval="${2:-1}"
  ensure_dir "$(dirname -- "${output_file}")"
  bash "${SCRIPT_DIR}/collect_gpu_stats.sh" --output "${output_file}" --interval "${interval}" >/dev/null 2>&1 &
  echo $!
}

stop_process_if_running() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

json_escape_file() {
  python3 - <<PY
import json
from pathlib import Path
path = Path(${1@Q})
print(json.dumps(str(path)))
PY
}

assert_framework_env_ready() {
  local framework="$1"
  local pybin
  pybin="$(framework_python "${framework}")"
  [[ -x "${pybin}" ]] || die "framework environment is not ready: ${pybin}. Run setup_envs.sh first."
}

apply_framework_toolkit_env() {
  local framework="$1"
  local pybin

  pybin="$(framework_python "${framework}")"
  if [[ -f "${TOOLKIT_ENV_SCRIPT}" ]]; then
    # shellcheck source=/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/scripts/toolkit_env.sh
    source "${TOOLKIT_ENV_SCRIPT}"
    benchmark_toolkit_apply_framework_env "${framework}" "${pybin}"
  fi
}

make_run_prefix() {
  local framework="$1"
  local model_id="$2"
  local scenario_id="$3"
  printf '%s_%s_%s_%s\n' "${framework}" "${model_id}" "${scenario_id}" "$(timestamp)"
}
