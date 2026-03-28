#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=consumer_gpu_benchmark/scripts/toolkit_env.sh
source "${SCRIPT_DIR}/toolkit_env.sh"

MODE="${1:-auto}"
TOOLKIT_ROOT=""

log() {
  printf '[smoke_check_cuda12_8] %s\n' "$*"
}

die() {
  printf '[smoke_check_cuda12_8][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash consumer_gpu_benchmark/scripts/smoke_check_cuda12_8.sh [auto|system|rootless]

Modes:
  auto      Prefer /usr/local/cuda-12.8, then fall back to benchmark rootless 12.8
  system    Require /usr/local/cuda-12.8
  rootless  Require the benchmark rootless toolkit
EOF
}

resolve_toolkit_root() {
  case "${MODE}" in
    auto)
      if [[ -x "${TOOLKIT_SYSTEM_CUDA_12_8}/bin/nvcc" ]]; then
        printf '%s\n' "${TOOLKIT_SYSTEM_CUDA_12_8}"
        return 0
      fi
      if [[ -x "${TOOLKIT_ROOTLESS_CUDA_12_8}/bin/nvcc" ]]; then
        printf '%s\n' "${TOOLKIT_ROOTLESS_CUDA_12_8}"
        return 0
      fi
      ;;
    system)
      if [[ -x "${TOOLKIT_SYSTEM_CUDA_12_8}/bin/nvcc" ]]; then
        printf '%s\n' "${TOOLKIT_SYSTEM_CUDA_12_8}"
        return 0
      fi
      die "system toolkit not found: ${TOOLKIT_SYSTEM_CUDA_12_8}"
      ;;
    rootless)
      if [[ -x "${TOOLKIT_ROOTLESS_CUDA_12_8}/bin/nvcc" ]]; then
        printf '%s\n' "${TOOLKIT_ROOTLESS_CUDA_12_8}"
        return 0
      fi
      die "rootless toolkit not found: ${TOOLKIT_ROOTLESS_CUDA_12_8}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown mode: ${MODE}"
      ;;
  esac

  die "no usable CUDA 12.8 toolkit found"
}

compile_smoke() {
  local tmpdir source_file output_file
  tmpdir="$(mktemp -d)"
  source_file="${tmpdir}/cuda12_8_smoke.cu"
  output_file="${tmpdir}/cuda12_8_smoke"

  cat >"${source_file}" <<'EOF'
#include <cuda_fp8.h>
#include <cuda_runtime.h>

__global__ void noop_kernel() {}

int main() {
  noop_kernel<<<1, 1>>>();
  return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
}
EOF

  log "Compiling smoke test with ${TOOLKIT_ROOT}/bin/nvcc"
  "${TOOLKIT_ROOT}/bin/nvcc" \
    -I"${TOOLKIT_ROOT}/targets/x86_64-linux/include" \
    -L"${TOOLKIT_ROOT}/targets/x86_64-linux/lib" \
    -Xlinker "-rpath=${TOOLKIT_ROOT}/targets/x86_64-linux/lib" \
    "${source_file}" \
    -o "${output_file}"

  log "Running compiled smoke binary"
  "${output_file}"
  rm -rf -- "${tmpdir}"
}

check_ambient_pollution() {
  local polluted=0

  if [[ "${CUDA_HOME:-}" == "/usr/local/cuda-13.0" ]]; then
    log "ambient CUDA_HOME is still polluted: ${CUDA_HOME}"
    polluted=1
  fi
  if [[ "${CUDA_PATH:-}" == "/usr/local/cuda-13.0" ]]; then
    log "ambient CUDA_PATH is still polluted: ${CUDA_PATH}"
    polluted=1
  fi
  if [[ "${CUDACXX:-}" == "/usr/local/cuda-13.0/bin/nvcc" ]]; then
    log "ambient CUDACXX is still polluted: ${CUDACXX}"
    polluted=1
  fi
  case ":${PATH:-}:" in
    *":/usr/local/cuda-13.0/bin:"*)
      log "ambient PATH still contains /usr/local/cuda-13.0/bin"
      polluted=1
      ;;
  esac
  case ":${LD_LIBRARY_PATH:-}:" in
    *":/usr/local/cuda-13.0/lib64:"*)
      log "ambient LD_LIBRARY_PATH still contains /usr/local/cuda-13.0/lib64"
      polluted=1
      ;;
  esac

  if [[ "${polluted}" -eq 0 ]]; then
    log "ambient shell does not expose the broken /usr/local/cuda-13.0 toolkit root"
  else
    log "smoke-check can still pass, but your login shell needs cleanup"
  fi
}

main() {
  TOOLKIT_ROOT="$(resolve_toolkit_root)"

  log "toolkit_root=${TOOLKIT_ROOT}"
  log "nvcc_version=$("${TOOLKIT_ROOT}/bin/nvcc" --version | tail -n 1)"
  [[ -f "${TOOLKIT_ROOT}/targets/x86_64-linux/include/cuda_fp8.h" ]] || die "missing cuda_fp8.h"
  [[ -f "${TOOLKIT_ROOT}/targets/x86_64-linux/lib/libcudart.so" ]] || die "missing libcudart.so"

  check_ambient_pollution
  compile_smoke

  log "smoke check passed"
}

main "$@"
