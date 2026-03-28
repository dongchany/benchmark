#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=consumer_gpu_benchmark/scripts/toolkit_env.sh
source "${SCRIPT_DIR}/toolkit_env.sh"

export BENCHMARK_CUDA_HOME_OVERRIDE="${TOOLKIT_SYSTEM_CUDA_12_8}"
benchmark_toolkit_apply_base_env

printf 'Activated system CUDA toolkit at %s\n' "${CUDA_HOME}"
printf 'nvcc=%s\n' "${CUDACXX:-}"
