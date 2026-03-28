#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=consumer_gpu_benchmark/scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

FRAMEWORK=""
MODEL_ID=""
SCENARIO_ID=""
BASE_URL=""
REQUEST_RATE=""
MAX_CONCURRENCY=""
REPEAT_INDEX=1
GPU_SAMPLING_INTERVAL=1
DISABLE_GPU_SAMPLER=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash consumer_gpu_benchmark/scripts/bench_online.sh \
    --framework <vllm|sglang> \
    --model-id <id> \
    --scenario-id <id> \
    [options] [-- extra benchmark args]

Required:
  --framework <name>           One of: vllm, sglang
  --model-id <id>              Model id from configs/models.json
  --scenario-id <id>           Scenario id from configs/scenarios.json

Options:
  --base-url <url>             Base URL of an already running service. If omitted, framework default is used.
  --request-rate <float>       Override scenario request rate with a single value
  --max-concurrency <int>      Override scenario max concurrency with a single value
  --repeat-index <int>         Optional repeat index for result naming (default: 1)
  --gpu-sampling-interval <s>  GPU metric sampling interval in seconds (default: 1)
  --disable-gpu-sampler        Skip background GPU metric collection
  -h, --help                   Show this help message

Arguments after "--" are passed directly to the framework benchmark CLI.

Behavior:
  - Reads the online scenario definition from configs/scenarios.json
  - Runs one request-rate x concurrency combination per invocation
  - Writes raw benchmark output and GPU samples into results/raw/
  - Keeps framework-specific native outputs for later normalization

Examples:
  bash consumer_gpu_benchmark/scripts/bench_online.sh \
    --framework vllm --model-id qwen3-0.6b --scenario-id online_short_smoke \
    --base-url http://127.0.0.1:18000 --request-rate 1 --max-concurrency 1

  bash consumer_gpu_benchmark/scripts/bench_online.sh \
    --framework sglang --model-id qwen3-1.7b --scenario-id online_medium_balance \
    --base-url http://127.0.0.1:19000 --request-rate 4 --max-concurrency 8
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --framework)
        shift
        [[ $# -gt 0 ]] || die "--framework requires a value"
        FRAMEWORK="$1"
        ;;
      --model-id)
        shift
        [[ $# -gt 0 ]] || die "--model-id requires a value"
        MODEL_ID="$1"
        ;;
      --scenario-id)
        shift
        [[ $# -gt 0 ]] || die "--scenario-id requires a value"
        SCENARIO_ID="$1"
        ;;
      --base-url)
        shift
        [[ $# -gt 0 ]] || die "--base-url requires a value"
        BASE_URL="$1"
        ;;
      --request-rate)
        shift
        [[ $# -gt 0 ]] || die "--request-rate requires a value"
        REQUEST_RATE="$1"
        ;;
      --max-concurrency)
        shift
        [[ $# -gt 0 ]] || die "--max-concurrency requires a value"
        MAX_CONCURRENCY="$1"
        ;;
      --repeat-index)
        shift
        [[ $# -gt 0 ]] || die "--repeat-index requires a value"
        REPEAT_INDEX="$1"
        ;;
      --gpu-sampling-interval)
        shift
        [[ $# -gt 0 ]] || die "--gpu-sampling-interval requires a value"
        GPU_SAMPLING_INTERVAL="$1"
        ;;
      --disable-gpu-sampler)
        DISABLE_GPU_SAMPLER=1
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
  [[ -n "${FRAMEWORK}" ]] || die "--framework is required"
  [[ -n "${MODEL_ID}" ]] || die "--model-id is required"
  [[ -n "${SCENARIO_ID}" ]] || die "--scenario-id is required"
  [[ "${FRAMEWORK}" == "vllm" || "${FRAMEWORK}" == "sglang" ]] || die "unsupported framework: ${FRAMEWORK}"
  assert_framework_env_ready "${FRAMEWORK}"
  require_file "${SCENARIOS_JSON}"
  require_file "${MODELS_JSON}"
}

scenario_field() {
  local scenario_id="$1"
  local expr="$2"
  python3 - <<PY
import json
from pathlib import Path
path = Path(${SCENARIOS_JSON@Q})
with path.open() as f:
    data = json.load(f)
scenario = next((s for s in data["scenarios"] if s["id"] == ${scenario_id@Q}), None)
if scenario is None:
    raise SystemExit(f"scenario not found: {${scenario_id@Q}}")
value = ${expr}
if isinstance(value, (dict, list)):
    print(json.dumps(value))
elif value is None:
    print("")
else:
    print(value)
PY
}

resolve_base_url() {
  if [[ -n "${BASE_URL}" ]]; then
    printf '%s\n' "${BASE_URL}"
    return
  fi
  case "${FRAMEWORK}" in
    vllm) printf 'http://%s:%s\n' "${DEFAULT_HOST}" "${DEFAULT_VLLM_PORT}" ;;
    sglang) printf 'http://%s:%s\n' "${DEFAULT_HOST}" "${DEFAULT_SGLANG_PORT}" ;;
  esac
}

main() {
  parse_args "$@"
  validate_args

  local category dataset_json dataset_type input_len output_len num_prompts
  category="$(scenario_field "${SCENARIO_ID}" 'scenario.get("category")')"
  [[ "${category}" == "online" ]] || die "scenario ${SCENARIO_ID} is not an online scenario"

  dataset_json="$(scenario_field "${SCENARIO_ID}" 'scenario.get("dataset", {})')"
  dataset_type="$(python3 - <<PY
import json
print(json.loads(${dataset_json@Q}).get("type", "random"))
PY
)"
  input_len="$(python3 - <<PY
import json
print(json.loads(${dataset_json@Q}).get("input_len", ""))
PY
)"
  output_len="$(python3 - <<PY
import json
print(json.loads(${dataset_json@Q}).get("output_len", ""))
PY
)"
  num_prompts="$(python3 - <<PY
import json
print(json.loads(${dataset_json@Q}).get("num_prompts", 100))
PY
)"

  if [[ -z "${REQUEST_RATE}" ]]; then
    REQUEST_RATE="$(scenario_field "${SCENARIO_ID}" 'scenario.get("request_rates", [float("inf")])[0]')"
  fi
  if [[ -z "${MAX_CONCURRENCY}" ]]; then
    MAX_CONCURRENCY="$(scenario_field "${SCENARIO_ID}" 'scenario.get("max_concurrency", [None])[0]')"
  fi
  if [[ "${MAX_CONCURRENCY}" == "None" ]]; then
    MAX_CONCURRENCY=""
  fi

  local pybin base_url run_prefix native_result_file wrapper_meta_file gpu_csv_file sampler_pid=""
  pybin="$(framework_python "${FRAMEWORK}")"
  apply_framework_toolkit_env "${FRAMEWORK}"
  base_url="$(resolve_base_url)"
  run_prefix="$(make_run_prefix "${FRAMEWORK}" "${MODEL_ID}" "${SCENARIO_ID}")_r${REPEAT_INDEX}_rr${REQUEST_RATE}_mc${MAX_CONCURRENCY:-na}"
  native_result_file="${RAW_RESULTS_DIR}/${run_prefix}.jsonl"
  wrapper_meta_file="${RAW_RESULTS_DIR}/${run_prefix}.meta.json"
  gpu_csv_file="${RAW_RESULTS_DIR}/${run_prefix}.gpu.csv"

  ensure_dir "${RAW_RESULTS_DIR}"
  ensure_dir "${NORMALIZED_RESULTS_DIR}"

  log "Running online benchmark: framework=${FRAMEWORK}, model_id=${MODEL_ID}, scenario_id=${SCENARIO_ID}"
  log "Base URL: ${base_url}"
  log "Scenario dataset type: ${dataset_type}"
  log "Request rate: ${REQUEST_RATE}"
  log "Max concurrency: ${MAX_CONCURRENCY:-unset}"
  log "Native result file: ${native_result_file}"

  if ! wait_for_http_ok "${base_url}/v1/models" 30; then
    die "service is not ready at ${base_url}/v1/models"
  fi

  if [[ ${DISABLE_GPU_SAMPLER} -eq 0 ]]; then
    sampler_pid="$(start_gpu_sampler "${gpu_csv_file}" "${GPU_SAMPLING_INTERVAL}")"
    log "Started GPU sampler with pid=${sampler_pid}"
  fi

  trap 'stop_process_if_running "${sampler_pid}"' EXIT

  local cmd=()
  if [[ "${FRAMEWORK}" == "vllm" ]]; then
    cmd=(
      "${pybin}" -m vllm.entrypoints.cli.main bench serve
      --backend openai
      --base-url "${base_url}"
      --endpoint /v1/completions
      --dataset-name "${dataset_type}"
      --num-prompts "${num_prompts}"
      --request-rate "${REQUEST_RATE}"
      --result-dir "${RAW_RESULTS_DIR}"
      --result-filename "$(basename -- "${native_result_file}")"
      --save-result
      --disable-tqdm
      --metadata framework="${FRAMEWORK}" model_id="${MODEL_ID}" scenario_id="${SCENARIO_ID}" repeat_index="${REPEAT_INDEX}"
    )
    if [[ -n "${MAX_CONCURRENCY}" ]]; then
      cmd+=(--max-concurrency "${MAX_CONCURRENCY}")
    fi
    if [[ "${dataset_type}" == "random" ]]; then
      cmd+=(--input-len "${input_len}" --output-len "${output_len}")
    fi
  else
    cmd=(
      "${pybin}" -m sglang.bench_serving
      --backend sglang
      --base-url "${base_url}"
      --dataset-name "${dataset_type}"
      --num-prompts "${num_prompts}"
      --request-rate "${REQUEST_RATE}"
      --output-file "${native_result_file}"
      --disable-tqdm
      --ready-check-timeout-sec 10
    )
    if [[ -n "${MAX_CONCURRENCY}" ]]; then
      cmd+=(--max-concurrency "${MAX_CONCURRENCY}")
    fi
    if [[ "${dataset_type}" == "random" ]]; then
      cmd+=(--random-input-len "${input_len}" --random-output-len "${output_len}")
    elif [[ "${dataset_type}" == "generated-shared-prefix" ]]; then
      local gsp_num_groups gsp_system_prompt_len gsp_question_len gsp_output_len
      gsp_num_groups="$(python3 - <<PY
import json
print(json.loads(${dataset_json@Q}).get("gsp_num_groups", 1))
PY
)"
      gsp_system_prompt_len="$(python3 - <<PY
import json
print(json.loads(${dataset_json@Q}).get("gsp_system_prompt_len", 2048))
PY
)"
      gsp_question_len="$(python3 - <<PY
import json
print(json.loads(${dataset_json@Q}).get("gsp_question_len", 128))
PY
)"
      gsp_output_len="$(python3 - <<PY
import json
print(json.loads(${dataset_json@Q}).get("gsp_output_len", 128))
PY
)"
      cmd+=(
        --gsp-num-groups "${gsp_num_groups}"
        --gsp-system-prompt-len "${gsp_system_prompt_len}"
        --gsp-question-len "${gsp_question_len}"
        --random-output-len "${gsp_output_len}"
      )
    fi
  fi

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  log "Executing benchmark command"
  printf '%s\n' "${cmd[*]}"
  "${cmd[@]}"

  stop_process_if_running "${sampler_pid}"
  sampler_pid=""
  trap - EXIT

  cat >"${wrapper_meta_file}" <<EOF
{
  "framework": "${FRAMEWORK}",
  "model_id": "${MODEL_ID}",
  "scenario_id": "${SCENARIO_ID}",
  "category": "online",
  "base_url": "${base_url}",
  "request_rate": "${REQUEST_RATE}",
  "max_concurrency": "${MAX_CONCURRENCY}",
  "dataset_type": "${dataset_type}",
  "input_len": "${input_len}",
  "output_len": "${output_len}",
  "num_prompts": "${num_prompts}",
  "repeat_index": ${REPEAT_INDEX},
  "native_result_file": "${native_result_file}",
  "gpu_result_file": "${gpu_csv_file}",
  "recorded_at": "$(date -Iseconds)"
}
EOF

  printf 'BENCH_ONLINE_RESULT=%s\n' "${native_result_file}"
  printf 'BENCH_ONLINE_META=%s\n' "${wrapper_meta_file}"
  printf 'BENCH_ONLINE_GPU=%s\n' "${gpu_csv_file}"
}

main "$@"
