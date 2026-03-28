#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=consumer_gpu_benchmark/scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_ID=""
HOST="${DEFAULT_HOST}"
PORT="${DEFAULT_SGLANG_PORT}"
DEVICES="${DEFAULT_SINGLE_GPU}"
TP_SIZE=1
MEM_FRACTION_STATIC=""
MAX_RUNNING_REQUESTS=""
TRUST_REMOTE_CODE=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash consumer_gpu_benchmark/scripts/run_sglang.sh --model-id <model-id> [options] [-- extra sglang serve args]

Required:
  --model-id <id>              Model id from configs/models.json, e.g. qwen3-0.6b

Options:
  --host <host>                Bind host (default: 127.0.0.1)
  --port <port>                Bind port (default: 19000)
  --devices <ids>              CUDA_VISIBLE_DEVICES value (default: 1)
  --tp-size <n>                Tensor parallel size (default: 1)
  --mem-fraction-static <f>    Optional explicit static memory fraction for SGLang
  --max-running-requests <n>   Optional explicit max running requests
  --trust-remote-code          Pass --trust-remote-code to SGLang
  -h, --help                   Show this help message

Any arguments after "--" are appended directly to `sglang serve`.

Outputs:
  - Service logs under consumer_gpu_benchmark/logs/sglang/
  - A run metadata file under consumer_gpu_benchmark/results/raw/
  - The script prints PID, log path, and base URL once the service is ready
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-id)
        shift
        [[ $# -gt 0 ]] || die "--model-id requires a value"
        MODEL_ID="$1"
        ;;
      --host)
        shift
        [[ $# -gt 0 ]] || die "--host requires a value"
        HOST="$1"
        ;;
      --port)
        shift
        [[ $# -gt 0 ]] || die "--port requires a value"
        PORT="$1"
        ;;
      --devices)
        shift
        [[ $# -gt 0 ]] || die "--devices requires a value"
        DEVICES="$1"
        ;;
      --tp-size)
        shift
        [[ $# -gt 0 ]] || die "--tp-size requires a value"
        TP_SIZE="$1"
        ;;
      --mem-fraction-static)
        shift
        [[ $# -gt 0 ]] || die "--mem-fraction-static requires a value"
        MEM_FRACTION_STATIC="$1"
        ;;
      --max-running-requests)
        shift
        [[ $# -gt 0 ]] || die "--max-running-requests requires a value"
        MAX_RUNNING_REQUESTS="$1"
        ;;
      --trust-remote-code)
        TRUST_REMOTE_CODE=1
        ;;
      --)
        shift
        EXTRA_ARGS=("$@")
        break
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
    shift
  done
}

validate_args() {
  [[ -n "${MODEL_ID}" ]] || die "--model-id is required"
  assert_framework_env_ready sglang
  require_file "${MODELS_JSON}"
  require_file "${SCENARIOS_JSON}"
}

resolve_sglang_cuda_home() {
  local nvcc_path=""
  local resolved_nvcc=""
  local candidate=""

  if [[ -n "${SGLANG_CUDA_HOME_OVERRIDE:-}" ]]; then
    printf '%s\n' "${SGLANG_CUDA_HOME_OVERRIDE}"
    return 0
  fi

  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    printf '%s\n' "${CUDA_HOME}"
    return 0
  fi

  nvcc_path="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "${nvcc_path}" ]]; then
    resolved_nvcc="$(readlink -f "${nvcc_path}" 2>/dev/null || printf '%s' "${nvcc_path}")"
    candidate="$(cd -- "$(dirname -- "${resolved_nvcc}")/.." && pwd)"
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  return 1
}

main() {
  parse_args "$@"
  validate_args

  local pybin sglang_bin model_path hf_name log_dir run_prefix log_file meta_file base_url cuda_home cudacxx
  pybin="$(framework_python sglang)"
  apply_framework_toolkit_env sglang
  sglang_bin="$(dirname -- "${pybin}")/sglang"
  model_path="$(model_field "${MODEL_ID}" "local_path")"
  hf_name="$(model_field "${MODEL_ID}" "hf_name")"
  log_dir="$(framework_log_dir sglang)"
  run_prefix="$(make_run_prefix sglang "${MODEL_ID}" serve)"
  log_file="${log_dir}/${run_prefix}.log"
  meta_file="${RAW_RESULTS_DIR}/${run_prefix}.meta.json"
  base_url="http://${HOST}:${PORT}"
  cuda_home="$(resolve_sglang_cuda_home || true)"
  cudacxx=""
  if [[ -n "${cuda_home}" ]]; then
    cudacxx="${cuda_home}/bin/nvcc"
  fi

  require_dir "${SGLANG_REPO}"
  require_dir "${log_dir}"
  require_dir "${RAW_RESULTS_DIR}"
  require_dir "${model_path}"

  if ! python3 - <<PY
port = int(${PORT@Q})
if port <= 0 or port > 65535:
    raise SystemExit(1)
PY
  then
    die "invalid port: ${PORT}"
  fi

  [[ -x "${sglang_bin}" ]] || die "sglang console script not found: ${sglang_bin}"

  local cmd=(
    "${sglang_bin}" serve
    --model-path "${model_path}"
    --host "${HOST}"
    --port "${PORT}"
    --tp-size "${TP_SIZE}"
  )

  if [[ -n "${MEM_FRACTION_STATIC}" ]]; then
    cmd+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
  fi

  if [[ -n "${MAX_RUNNING_REQUESTS}" ]]; then
    cmd+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
  fi

  if [[ ${TRUST_REMOTE_CODE} -eq 1 ]]; then
    cmd+=(--trust-remote-code)
  fi

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  log "Starting SGLang service for model_id=${MODEL_ID}"
  log "Resolved local model path: ${model_path}"
  log "Resolved HF model name: ${hf_name}"
  log "CUDA_VISIBLE_DEVICES=${DEVICES}"
  log "Log file: ${log_file}"
  if [[ -n "${cuda_home}" ]]; then
    log "Resolved CUDA_HOME=${cuda_home} for SGLang runtime JIT builds"
  fi

  if [[ -n "${cuda_home}" ]]; then
    CUDA_VISIBLE_DEVICES="${DEVICES}" CUDA_HOME="${cuda_home}" CUDACXX="${cudacxx}" nohup "${cmd[@]}" >"${log_file}" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${DEVICES}" nohup "${cmd[@]}" >"${log_file}" 2>&1 &
  fi
  local server_pid=$!

  cat >"${meta_file}" <<EOF
{
  "framework": "sglang",
  "model_id": "${MODEL_ID}",
  "hf_name": "${hf_name}",
  "model_path": "${model_path}",
  "host": "${HOST}",
  "port": ${PORT},
  "base_url": "${base_url}",
  "devices": "${DEVICES}",
  "tp_size": ${TP_SIZE},
  "mem_fraction_static": "${MEM_FRACTION_STATIC}",
  "max_running_requests": "${MAX_RUNNING_REQUESTS}",
  "cuda_home": "${cuda_home}",
  "cudacxx": "${cudacxx}",
  "trust_remote_code": ${TRUST_REMOTE_CODE},
  "pid": ${server_pid},
  "log_file": "${log_file}",
  "started_at": "$(date -Iseconds)"
}
EOF

  if ! wait_for_http_ok "${base_url}/health" 900; then
    warn "SGLang /health did not become ready in time; tailing recent logs"
    tail -n 80 "${log_file}" || true
    die "SGLang server failed to become healthy"
  fi

  if ! wait_for_http_ok "${base_url}/v1/models" 120; then
    warn "SGLang /v1/models did not become ready in time; tailing recent logs"
    tail -n 80 "${log_file}" || true
    die "SGLang server failed model discovery readiness"
  fi

  printf 'SGLANG_PID=%s\n' "${server_pid}"
  printf 'SGLANG_LOG=%s\n' "${log_file}"
  printf 'SGLANG_META=%s\n' "${meta_file}"
  printf 'SGLANG_BASE_URL=%s\n' "${base_url}"
}

main "$@"
