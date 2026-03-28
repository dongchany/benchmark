#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=consumer_gpu_benchmark/scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_ID=""
HOST="${DEFAULT_HOST}"
PORT="${DEFAULT_VLLM_PORT}"
DEVICES="${DEFAULT_SINGLE_GPU}"
TP_SIZE=1
GPU_MEMORY_UTILIZATION="0.90"
MAX_MODEL_LEN=""
ENABLE_PREFIX_CACHING=1
TRUST_REMOTE_CODE=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash consumer_gpu_benchmark/scripts/run_vllm.sh --model-id <model-id> [options] [-- extra vllm serve args]

Required:
  --model-id <id>              Model id from configs/models.json, e.g. qwen3-0.6b

Options:
  --host <host>                Bind host (default: 127.0.0.1)
  --port <port>                Bind port (default: 18000)
  --devices <ids>              CUDA_VISIBLE_DEVICES value (default: 1)
  --tp-size <n>                Tensor parallel size (default: 1)
  --gpu-memory-utilization <f> vLLM GPU memory utilization (default: 0.90)
  --max-model-len <n>          Optional explicit max model length
  --disable-prefix-caching     Disable prefix caching for fairness experiments
  --trust-remote-code          Pass --trust-remote-code to vLLM
  Env override: VLLM_ENV_DIR_OVERRIDE=/path/to/venv to use a non-default vLLM environment
  -h, --help                   Show this help message

Any arguments after "--" are appended directly to `vllm serve`.

Outputs:
  - Service logs under consumer_gpu_benchmark/logs/vllm/
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
      --gpu-memory-utilization)
        shift
        [[ $# -gt 0 ]] || die "--gpu-memory-utilization requires a value"
        GPU_MEMORY_UTILIZATION="$1"
        ;;
      --max-model-len)
        shift
        [[ $# -gt 0 ]] || die "--max-model-len requires a value"
        MAX_MODEL_LEN="$1"
        ;;
      --disable-prefix-caching)
        ENABLE_PREFIX_CACHING=0
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
  assert_framework_env_ready vllm
  require_file "${MODELS_JSON}"
  require_file "${SCENARIOS_JSON}"
}

main() {
  parse_args "$@"
  validate_args

  local pybin model_path hf_name log_dir run_prefix log_file meta_file base_url
  pybin="$(framework_python vllm)"
  apply_framework_toolkit_env vllm
  model_path="$(model_field "${MODEL_ID}" "local_path")"
  hf_name="$(model_field "${MODEL_ID}" "hf_name")"
  log_dir="$(framework_log_dir vllm)"
  run_prefix="$(make_run_prefix vllm "${MODEL_ID}" serve)"
  log_file="${log_dir}/${run_prefix}.log"
  meta_file="${RAW_RESULTS_DIR}/${run_prefix}.meta.json"
  base_url="http://${HOST}:${PORT}"

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

  local cmd=(
    "${pybin}" -m vllm.entrypoints.cli.main serve
    "${model_path}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --no-enable-log-requests
  )

  if [[ -n "${MAX_MODEL_LEN}" ]]; then
    cmd+=(--max-model-len "${MAX_MODEL_LEN}")
  fi

  if [[ ${ENABLE_PREFIX_CACHING} -eq 0 ]]; then
    cmd+=(--no-enable-prefix-caching)
  fi

  if [[ ${TRUST_REMOTE_CODE} -eq 1 ]]; then
    cmd+=(--trust-remote-code)
  fi

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  log "Starting vLLM service for model_id=${MODEL_ID}"
  log "Resolved local model path: ${model_path}"
  log "Resolved HF model name: ${hf_name}"
  log "CUDA_VISIBLE_DEVICES=${DEVICES}"
  log "Log file: ${log_file}"

  CUDA_VISIBLE_DEVICES="${DEVICES}" nohup "${cmd[@]}" >"${log_file}" 2>&1 &
  local server_pid=$!

  cat >"${meta_file}" <<EOF
{
  "framework": "vllm",
  "model_id": "${MODEL_ID}",
  "hf_name": "${hf_name}",
  "model_path": "${model_path}",
  "host": "${HOST}",
  "port": ${PORT},
  "base_url": "${base_url}",
  "devices": "${DEVICES}",
  "tp_size": ${TP_SIZE},
  "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION},
  "enable_prefix_caching": ${ENABLE_PREFIX_CACHING},
  "trust_remote_code": ${TRUST_REMOTE_CODE},
  "pid": ${server_pid},
  "log_file": "${log_file}",
  "started_at": "$(date -Iseconds)"
}
EOF

  if ! wait_for_http_ok "${base_url}/health" 900; then
    warn "vLLM /health did not become ready in time; tailing recent logs"
    tail -n 80 "${log_file}" || true
    die "vLLM server failed to become healthy"
  fi

  if ! wait_for_http_ok "${base_url}/v1/models" 120; then
    warn "vLLM /v1/models did not become ready in time; tailing recent logs"
    tail -n 80 "${log_file}" || true
    die "vLLM server failed model discovery readiness"
  fi

  printf 'VLLM_PID=%s\n' "${server_pid}"
  printf 'VLLM_LOG=%s\n' "${log_file}"
  printf 'VLLM_META=%s\n' "${meta_file}"
  printf 'VLLM_BASE_URL=%s\n' "${base_url}"
}

main "$@"
