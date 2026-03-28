#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=consumer_gpu_benchmark/scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

FRAMEWORK=""
MODEL_ID=""
BASE_URL=""
SCENARIO_ID="micro_prefill_decode"
BATCH_SIZES=""
INPUT_LENGTHS=""
OUTPUT_LENGTHS=""
REPEAT_INDEX=1
GPU_SAMPLING_INTERVAL=1
DISABLE_GPU_SAMPLER=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash consumer_gpu_benchmark/scripts/bench_micro.sh \
    --framework <vllm|sglang> \
    --model-id <id> \
    --base-url <url> \
    [options] [-- extra benchmark args]

Required:
  --framework <name>           One of: vllm, sglang
  --model-id <id>              Model id from configs/models.json
  --base-url <url>             Base URL of an already running service

Options:
  --scenario-id <id>           Scenario id from configs/scenarios.json (default: micro_prefill_decode)
  --batch-sizes <csv>          Override scenario batch sizes, e.g. 1,2,4,8
  --input-lengths <csv>        Override scenario input lengths, e.g. 256,1024,4096
  --output-lengths <csv>       Override scenario output lengths, e.g. 1,16,64
  --repeat-index <int>         Optional repeat index for result naming (default: 1)
  --gpu-sampling-interval <s>  GPU metric sampling interval in seconds (default: 1)
  --disable-gpu-sampler        Skip background GPU metric collection
  -h, --help                   Show this help message

Arguments after "--" are passed directly to `python -m sglang.bench_one_batch_server`.

Behavior:
  - Uses one common micro-benchmark harness for both frameworks
  - Talks to the already-running server via --base-url
  - Stores raw JSONL plus GPU telemetry into results/raw/

Example:
  bash consumer_gpu_benchmark/scripts/bench_micro.sh \
    --framework vllm --model-id qwen3-1.7b --base-url http://127.0.0.1:18000
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
      --base-url)
        shift
        [[ $# -gt 0 ]] || die "--base-url requires a value"
        BASE_URL="$1"
        ;;
      --scenario-id)
        shift
        [[ $# -gt 0 ]] || die "--scenario-id requires a value"
        SCENARIO_ID="$1"
        ;;
      --batch-sizes)
        shift
        [[ $# -gt 0 ]] || die "--batch-sizes requires a value"
        BATCH_SIZES="$1"
        ;;
      --input-lengths)
        shift
        [[ $# -gt 0 ]] || die "--input-lengths requires a value"
        INPUT_LENGTHS="$1"
        ;;
      --output-lengths)
        shift
        [[ $# -gt 0 ]] || die "--output-lengths requires a value"
        OUTPUT_LENGTHS="$1"
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
  [[ -n "${BASE_URL}" ]] || die "--base-url is required"
  [[ "${FRAMEWORK}" == "vllm" || "${FRAMEWORK}" == "sglang" ]] || die "unsupported framework: ${FRAMEWORK}"
  assert_framework_env_ready sglang
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

list_to_space_separated() {
  python3 - <<PY
import json
value = ${1@Q}
if value.strip().startswith('['):
    items = json.loads(value)
else:
    items = [x for x in value.split(',') if x]
print(' '.join(str(x) for x in items))
PY
}

main() {
  parse_args "$@"
  validate_args

  local category pybin model_path hf_name batch_values input_values output_values
  category="$(scenario_field "${SCENARIO_ID}" 'scenario.get("category")')"
  [[ "${category}" == "micro" ]] || die "scenario ${SCENARIO_ID} is not a micro scenario"

  pybin="$(framework_python sglang)"
  model_path="$(model_field "${MODEL_ID}" "local_path")"
  hf_name="$(model_field "${MODEL_ID}" "hf_name")"

  if [[ -z "${BATCH_SIZES}" ]]; then
    BATCH_SIZES="$(scenario_field "${SCENARIO_ID}" 'scenario.get("batch_sizes", [1])')"
  fi
  if [[ -z "${INPUT_LENGTHS}" ]]; then
    INPUT_LENGTHS="$(scenario_field "${SCENARIO_ID}" 'scenario.get("input_lengths", [1024])')"
  fi
  if [[ -z "${OUTPUT_LENGTHS}" ]]; then
    OUTPUT_LENGTHS="$(scenario_field "${SCENARIO_ID}" 'scenario.get("output_lengths", [16])')"
  fi

  batch_values="$(list_to_space_separated "${BATCH_SIZES}")"
  input_values="$(list_to_space_separated "${INPUT_LENGTHS}")"
  output_values="$(list_to_space_separated "${OUTPUT_LENGTHS}")"

  local run_prefix native_result_file wrapper_meta_file gpu_csv_file sampler_pid=""
  run_prefix="$(make_run_prefix "${FRAMEWORK}" "${MODEL_ID}" "${SCENARIO_ID}")_r${REPEAT_INDEX}"
  native_result_file="${RAW_RESULTS_DIR}/${run_prefix}.jsonl"
  wrapper_meta_file="${RAW_RESULTS_DIR}/${run_prefix}.meta.json"
  gpu_csv_file="${RAW_RESULTS_DIR}/${run_prefix}.gpu.csv"

  ensure_dir "${RAW_RESULTS_DIR}"
  require_dir "${model_path}"

  log "Running micro benchmark: framework=${FRAMEWORK}, model_id=${MODEL_ID}, scenario_id=${SCENARIO_ID}"
  log "Base URL: ${BASE_URL}"
  log "Resolved model path: ${model_path}"
  log "Resolved HF model name: ${hf_name}"
  log "Batch sizes: ${batch_values}"
  log "Input lengths: ${input_values}"
  log "Output lengths: ${output_values}"
  log "Native result file: ${native_result_file}"

  if ! wait_for_http_ok "${BASE_URL}/v1/models" 30; then
    die "service is not ready at ${BASE_URL}/v1/models"
  fi

  if [[ ${DISABLE_GPU_SAMPLER} -eq 0 ]]; then
    sampler_pid="$(start_gpu_sampler "${gpu_csv_file}" "${GPU_SAMPLING_INTERVAL}")"
    log "Started GPU sampler with pid=${sampler_pid}"
  fi

  trap 'stop_process_if_running "${sampler_pid}"' EXIT

  local cmd=(
    "${pybin}" -m sglang.bench_one_batch_server
    --base-url "${BASE_URL}"
    --dataset-name random
    --backend "${FRAMEWORK}"
    --model-path "${model_path}"
    --batch-size
  )

  # append list arguments carefully
  for value in ${batch_values}; do
    cmd+=("${value}")
  done
  cmd+=(--input-len)
  for value in ${input_values}; do
    cmd+=("${value}")
  done
  cmd+=(--output-len)
  for value in ${output_values}; do
    cmd+=("${value}")
  done
  cmd+=(
    --result-filename "${native_result_file}"
    --no-append-to-github-summary
    --skip-warmup
  )

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
  "category": "micro",
  "base_url": "${BASE_URL}",
  "model_path": "${model_path}",
  "hf_name": "${hf_name}",
  "batch_sizes": "${batch_values}",
  "input_lengths": "${input_values}",
  "output_lengths": "${output_values}",
  "repeat_index": ${REPEAT_INDEX},
  "native_result_file": "${native_result_file}",
  "gpu_result_file": "${gpu_csv_file}",
  "recorded_at": "$(date -Iseconds)"
}
EOF

  printf 'BENCH_MICRO_RESULT=%s\n' "${native_result_file}"
  printf 'BENCH_MICRO_META=%s\n' "${wrapper_meta_file}"
  printf 'BENCH_MICRO_GPU=%s\n' "${gpu_csv_file}"
}

main "$@"
