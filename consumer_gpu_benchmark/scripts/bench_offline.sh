#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=consumer_gpu_benchmark/scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

FRAMEWORK=""
MODEL_ID=""
SCENARIO_ID=""
DEVICES="${DEFAULT_SINGLE_GPU}"
TP_SIZE=1
GPU_MEMORY_UTILIZATION=""
REPEAT_INDEX=1
GPU_SAMPLING_INTERVAL=1
DISABLE_GPU_SAMPLER=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash consumer_gpu_benchmark/scripts/bench_offline.sh \
    --framework <vllm|sglang> \
    --model-id <id> \
    --scenario-id <id> \
    [options] [-- extra benchmark args]

Required:
  --framework <name>           One of: vllm, sglang
  --model-id <id>              Model id from configs/models.json
  --scenario-id <id>           Scenario id from configs/scenarios.json

Options:
  --devices <ids>              CUDA_VISIBLE_DEVICES value (default: 1)
  --tp-size <n>                Tensor parallel size for engine execution (default: 1)
  --gpu-memory-utilization <f> Optional vLLM engine memory cap for offline runs
  --repeat-index <int>         Optional repeat index for result naming (default: 1)
  --gpu-sampling-interval <s>  GPU metric sampling interval in seconds (default: 1)
  --disable-gpu-sampler        Skip background GPU metric collection
  -h, --help                   Show this help message

Arguments after "--" are passed directly to the framework benchmark CLI.

Behavior:
  - Reads the offline scenario definition from configs/scenarios.json
  - Uses each framework's native offline benchmark path
  - Writes raw benchmark output and GPU samples into results/raw/
  - Keeps framework-specific native outputs for later normalization

Examples:
  bash consumer_gpu_benchmark/scripts/bench_offline.sh \
    --framework vllm --model-id qwen3-1.7b --scenario-id offline_medium

  bash consumer_gpu_benchmark/scripts/bench_offline.sh \
    --framework sglang --model-id qwen3.5-4b --scenario-id offline_long --devices 1
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

main() {
  parse_args "$@"
  validate_args

  local category dataset_json dataset_type input_len output_len num_prompts
  category="$(scenario_field "${SCENARIO_ID}" 'scenario.get("category")')"
  [[ "${category}" == "offline" ]] || die "scenario ${SCENARIO_ID} is not an offline scenario"

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

  local pybin model_path hf_name run_prefix native_result_file wrapper_meta_file gpu_csv_file sampler_pid=""
  pybin="$(framework_python "${FRAMEWORK}")"
  apply_framework_toolkit_env "${FRAMEWORK}"
  model_path="$(model_field "${MODEL_ID}" "local_path")"
  hf_name="$(model_field "${MODEL_ID}" "hf_name")"
  run_prefix="$(make_run_prefix "${FRAMEWORK}" "${MODEL_ID}" "${SCENARIO_ID}")_r${REPEAT_INDEX}_tp${TP_SIZE}"
  native_result_file="${RAW_RESULTS_DIR}/${run_prefix}.jsonl"
  wrapper_meta_file="${RAW_RESULTS_DIR}/${run_prefix}.meta.json"
  gpu_csv_file="${RAW_RESULTS_DIR}/${run_prefix}.gpu.csv"

  ensure_dir "${RAW_RESULTS_DIR}"
  ensure_dir "${NORMALIZED_RESULTS_DIR}"
  require_dir "${model_path}"

  log "Running offline benchmark: framework=${FRAMEWORK}, model_id=${MODEL_ID}, scenario_id=${SCENARIO_ID}"
  log "Resolved local model path: ${model_path}"
  log "Resolved HF model name: ${hf_name}"
  log "Dataset type: ${dataset_type}"
  log "Input length: ${input_len}"
  log "Output length: ${output_len}"
  log "Num prompts: ${num_prompts}"
  log "CUDA_VISIBLE_DEVICES=${DEVICES}"
  log "Tensor parallel size=${TP_SIZE}"
  if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
    log "vLLM GPU memory utilization=${GPU_MEMORY_UTILIZATION}"
  fi
  log "Native result file: ${native_result_file}"

  if [[ ${DISABLE_GPU_SAMPLER} -eq 0 ]]; then
    sampler_pid="$(start_gpu_sampler "${gpu_csv_file}" "${GPU_SAMPLING_INTERVAL}")"
    log "Started GPU sampler with pid=${sampler_pid}"
  fi

  trap 'stop_process_if_running "${sampler_pid}"' EXIT

  local cmd=()
  if [[ "${FRAMEWORK}" == "vllm" ]]; then
    cmd=(
      "${pybin}" -m vllm.entrypoints.cli.main bench throughput
      --backend vllm
      --dataset-name "${dataset_type}"
      --num-prompts "${num_prompts}"
      --output-json "${native_result_file}"
      --model "${model_path}"
      --tensor-parallel-size "${TP_SIZE}"
      --tokenizer "${model_path}"
    )
    if [[ "${dataset_type}" == "random" ]]; then
      cmd+=(--random-input-len "${input_len}" --random-output-len "${output_len}")
    fi
    if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
      cmd+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
    fi
  else
    cmd=(
      "${pybin}" -m sglang.bench_offline_throughput
      --dataset-name "${dataset_type}"
      --num-prompts "${num_prompts}"
      --result-filename "${native_result_file}"
      --model-path "${model_path}"
      --tp-size "${TP_SIZE}"
      --skip-warmup
    )
    if [[ "${dataset_type}" == "random" ]]; then
      cmd+=(--random-input-len "${input_len}" --random-output-len "${output_len}")
    elif [[ "${dataset_type}" == "generated-shared-prefix" ]]; then
      local gsp_num_groups gsp_prompts_per_group gsp_system_prompt_len gsp_question_len gsp_output_len
      gsp_num_groups="$(python3 - <<PY
import json
print(json.loads(${dataset_json@Q}).get("gsp_num_groups", 1))
PY
)"
      gsp_prompts_per_group="$(python3 - <<PY
import json
print(json.loads(${dataset_json@Q}).get("gsp_prompts_per_group", 16))
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
        --gsp-prompts-per-group "${gsp_prompts_per_group}"
        --gsp-system-prompt-len "${gsp_system_prompt_len}"
        --gsp-question-len "${gsp_question_len}"
        --gsp-output-len "${gsp_output_len}"
      )
    fi
  fi

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  log "Executing benchmark command"
  printf '%s\n' "${cmd[*]}"
  CUDA_VISIBLE_DEVICES="${DEVICES}" "${cmd[@]}"

  stop_process_if_running "${sampler_pid}"
  sampler_pid=""
  trap - EXIT

  cat >"${wrapper_meta_file}" <<EOF
{
  "framework": "${FRAMEWORK}",
  "model_id": "${MODEL_ID}",
  "scenario_id": "${SCENARIO_ID}",
  "category": "offline",
  "model_path": "${model_path}",
  "hf_name": "${hf_name}",
  "devices": "${DEVICES}",
  "tp_size": ${TP_SIZE},
  "gpu_memory_utilization": "${GPU_MEMORY_UTILIZATION}",
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

  printf 'BENCH_OFFLINE_RESULT=%s\n' "${native_result_file}"
  printf 'BENCH_OFFLINE_META=%s\n' "${wrapper_meta_file}"
  printf 'BENCH_OFFLINE_GPU=%s\n' "${gpu_csv_file}"
}

main "$@"
