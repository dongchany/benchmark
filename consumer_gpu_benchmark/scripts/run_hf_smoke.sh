#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=consumer_gpu_benchmark/scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_ID=""
DEVICES="${DEFAULT_DUAL_GPU}"
MAX_NEW_TOKENS="16"
PROMPT="Explain tensor parallelism in one sentence."
DTYPE="bfloat16"
TRUST_REMOTE_CODE=0

usage() {
  cat <<'EOF'
Usage:
  bash consumer_gpu_benchmark/scripts/run_hf_smoke.sh --model-id <model-id> [options]

Required:
  --model-id <id>              Model id from configs/models.json, e.g. qwen3.5-9b

Options:
  --devices <ids>              CUDA_VISIBLE_DEVICES value (default: 0,1)
  --max-new-tokens <n>         Number of generated tokens (default: 16)
  --prompt <text>              Prompt text for the smoke run
  --dtype <dtype>              torch dtype: auto|bfloat16|float16|float32 (default: bfloat16)
  --trust-remote-code          Pass trust_remote_code=True to tokenizer/model loaders
  -h, --help                   Show this help message
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
      --devices)
        shift
        [[ $# -gt 0 ]] || die "--devices requires a value"
        DEVICES="$1"
        ;;
      --max-new-tokens)
        shift
        [[ $# -gt 0 ]] || die "--max-new-tokens requires a value"
        MAX_NEW_TOKENS="$1"
        ;;
      --prompt)
        shift
        [[ $# -gt 0 ]] || die "--prompt requires a value"
        PROMPT="$1"
        ;;
      --dtype)
        shift
        [[ $# -gt 0 ]] || die "--dtype requires a value"
        DTYPE="$1"
        ;;
      --trust-remote-code)
        TRUST_REMOTE_CODE=1
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

main() {
  parse_args "$@"
  [[ -n "${MODEL_ID}" ]] || die "--model-id is required"

  local pybin model_path hf_name
  pybin="$(framework_python sglang)"
  model_path="$(model_field "${MODEL_ID}" "local_path")"
  hf_name="$(model_field "${MODEL_ID}" "hf_name")"

  require_dir "${model_path}"
  [[ -x "${pybin}" ]] || die "python not found: ${pybin}"

  log "Starting HF smoke for model_id=${MODEL_ID}"
  log "Resolved local model path: ${model_path}"
  log "Resolved HF model name: ${hf_name}"
  log "CUDA_VISIBLE_DEVICES=${DEVICES}"

  CUDA_VISIBLE_DEVICES="${DEVICES}" "${pybin}" - <<PY
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = ${model_path@Q}
prompt = ${PROMPT@Q}
max_new_tokens = int(${MAX_NEW_TOKENS@Q})
dtype_name = ${DTYPE@Q}
trust_remote_code = bool(${TRUST_REMOTE_CODE})

dtype_map = {
    "auto": "auto",
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
if dtype_name not in dtype_map:
    raise SystemExit(f"unsupported dtype: {dtype_name}")

torch_dtype = dtype_map[dtype_name]

print(f"HF_MODEL_DIR={model_dir}")
print(f"HF_PROMPT={prompt}")
print(f"HF_MAX_NEW_TOKENS={max_new_tokens}")
print(f"HF_DTYPE={dtype_name}")
print("HF_LOADING_TOKENIZER=1")
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=trust_remote_code,
)

print("HF_LOADING_MODEL=1")
load_started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=trust_remote_code,
    dtype=torch_dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
load_elapsed = time.perf_counter() - load_started

first_param = next(model.parameters())
inputs = tokenizer(prompt, return_tensors="pt").to(first_param.device)

infer_started = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
infer_elapsed = time.perf_counter() - infer_started

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"HF_LOAD_S={load_elapsed:.3f}")
print(f"HF_INFER_S={infer_elapsed:.3f}")
print(f"HF_INPUT_DEVICE={first_param.device}")
print("HF_DEVICE_MAP_JSON=" + json.dumps(getattr(model, "hf_device_map", {}), sort_keys=True))
print("HF_OUTPUT_BEGIN")
print(decoded)
print("HF_OUTPUT_END")
PY
}

main "$@"
