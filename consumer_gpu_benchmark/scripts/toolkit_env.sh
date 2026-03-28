#!/usr/bin/env bash
set -euo pipefail

TOOLKIT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_BENCH_ROOT="$(cd -- "${TOOLKIT_SCRIPT_DIR}/.." && pwd)"
TOOLKIT_CACHE_ROOT="${TOOLKIT_BENCH_ROOT}/artifacts/toolkit_cache"
TOOLKIT_SYSTEM_CUDA_12_8="${TOOLKIT_SYSTEM_CUDA_12_8:-/usr/local/cuda-12.8}"
TOOLKIT_ROOTLESS_CUDA_12_8="${TOOLKIT_BENCH_ROOT}/artifacts/cuda-12.8-rootless/usr/local/cuda-12.8"
TOOLKIT_DEFAULT_HF_HOME="/home/dong/xilinx/huggingface"

benchmark_toolkit_prepend_path() {
  local var_name="$1"
  local path_item="$2"
  local current_value="${!var_name:-}"

  [[ -n "${path_item}" ]] || return 0
  [[ -d "${path_item}" ]] || return 0

  case ":${current_value}:" in
    *":${path_item}:"*) ;;
    *)
      export "${var_name}=${path_item}${current_value:+:${current_value}}"
      ;;
  esac
}

benchmark_toolkit_strip_path() {
  local var_name="$1"
  local path_item="$2"
  local current_value="${!var_name:-}"
  local rebuilt=""
  local entry=""

  [[ -n "${current_value}" ]] || return 0

  IFS=':' read -r -a _toolkit_path_entries <<< "${current_value}"
  for entry in "${_toolkit_path_entries[@]}"; do
    [[ -n "${entry}" ]] || continue
    [[ "${entry}" == "${path_item}" ]] && continue
    rebuilt="${rebuilt:+${rebuilt}:}${entry}"
  done

  export "${var_name}=${rebuilt}"
}

benchmark_toolkit_append_flag() {
  local var_name="$1"
  local flag="$2"
  local current_value="${!var_name:-}"

  [[ -n "${flag}" ]] || return 0

  case " ${current_value} " in
    *" ${flag} "*) ;;
    *)
      export "${var_name}=${current_value:+${current_value} }${flag}"
      ;;
  esac
}

benchmark_toolkit_unset_if_equals() {
  local var_name="$1"
  local expected_value="$2"
  local current_value="${!var_name:-}"

  if [[ -n "${current_value}" && "${current_value}" == "${expected_value}" ]]; then
    unset "${var_name}"
  fi
}

benchmark_toolkit_detect_compiler_cuda_home() {
  local nvcc_path=""
  local resolved_nvcc=""
  local candidate=""

  if [[ -n "${BENCHMARK_CUDA_HOME_OVERRIDE:-}" && -x "${BENCHMARK_CUDA_HOME_OVERRIDE}/bin/nvcc" ]]; then
    printf '%s\n' "${BENCHMARK_CUDA_HOME_OVERRIDE}"
    return 0
  fi

  if [[ -x "${TOOLKIT_SYSTEM_CUDA_12_8}/bin/nvcc" ]]; then
    printf '%s\n' "${TOOLKIT_SYSTEM_CUDA_12_8}"
    return 0
  fi

  if [[ -x "${TOOLKIT_ROOTLESS_CUDA_12_8}/bin/nvcc" ]]; then
    printf '%s\n' "${TOOLKIT_ROOTLESS_CUDA_12_8}"
    return 0
  fi

  nvcc_path="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "${nvcc_path}" ]]; then
    resolved_nvcc="$(readlink -f "${nvcc_path}" 2>/dev/null || printf '%s' "${nvcc_path}")"
    candidate="$(cd -- "$(dirname -- "${resolved_nvcc}")/.." && pwd)"
    printf '%s\n' "${candidate}"
    return 0
  fi

  return 1
}

benchmark_toolkit_python_cuda_include() {
  local pybin="$1"
  "${pybin}" - <<'PY'
import importlib.util
import pathlib

spec = importlib.util.find_spec("nvidia.cuda_runtime")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(1)

root = pathlib.Path(list(spec.submodule_search_locations)[0])
include_dir = root / "include"
if not include_dir.is_dir():
    raise SystemExit(1)

print(include_dir)
PY
}

benchmark_toolkit_python_cuda_lib() {
  local pybin="$1"
  "${pybin}" - <<'PY'
import importlib.util
import pathlib

spec = importlib.util.find_spec("nvidia.cuda_runtime")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(1)

root = pathlib.Path(list(spec.submodule_search_locations)[0])
lib_dir = root / "lib"
if not lib_dir.is_dir():
    raise SystemExit(1)

print(lib_dir)
PY
}

benchmark_toolkit_apply_base_env() {
  local compiler_cuda_home=""

  mkdir -p \
    "${TOOLKIT_CACHE_ROOT}/xdg" \
    "${TOOLKIT_CACHE_ROOT}/pip" \
    "${TOOLKIT_CACHE_ROOT}/triton" \
    "${TOOLKIT_CACHE_ROOT}/torchinductor" \
    "${TOOLKIT_CACHE_ROOT}/flashinfer"

  export HF_HOME="${HF_HOME:-${TOOLKIT_DEFAULT_HF_HOME}}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TOOLKIT_CACHE_ROOT}/xdg}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TOOLKIT_CACHE_ROOT}/pip}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TOOLKIT_CACHE_ROOT}/triton}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TOOLKIT_CACHE_ROOT}/torchinductor}"
  export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${TOOLKIT_CACHE_ROOT}/flashinfer}"

  benchmark_toolkit_unset_if_equals CUDA_HOME "/usr/local/cuda-13.0"
  benchmark_toolkit_unset_if_equals CUDA_PATH "/usr/local/cuda-13.0"
  benchmark_toolkit_unset_if_equals CUDACXX "/usr/local/cuda-13.0/bin/nvcc"

  compiler_cuda_home="$(benchmark_toolkit_detect_compiler_cuda_home || true)"
  if [[ -n "${compiler_cuda_home}" && -x "${compiler_cuda_home}/bin/nvcc" ]]; then
    export CUDA_HOME="${compiler_cuda_home}"
    export CUDA_PATH="${compiler_cuda_home}"
    export CUDACXX="${compiler_cuda_home}/bin/nvcc"
  fi

  if [[ ! -x /usr/local/cuda-13.0/bin/nvcc ]]; then
    benchmark_toolkit_strip_path PATH "/usr/local/cuda-13.0/bin"
  fi
  benchmark_toolkit_strip_path CPATH "/usr/local/cuda-13.0/include"
  benchmark_toolkit_strip_path CPLUS_INCLUDE_PATH "/usr/local/cuda-13.0/include"
  if [[ ! -d /usr/local/cuda-13.0/lib64 ]]; then
    benchmark_toolkit_strip_path LD_LIBRARY_PATH "/usr/local/cuda-13.0/lib64"
  fi

  benchmark_toolkit_prepend_path PATH "${CUDA_HOME:-}/bin"
  benchmark_toolkit_prepend_path LD_LIBRARY_PATH "${CUDA_HOME:-}/lib64"
  benchmark_toolkit_prepend_path LD_LIBRARY_PATH "${CUDA_HOME:-}/targets/x86_64-linux/lib"
  benchmark_toolkit_prepend_path LD_LIBRARY_PATH "/usr/local/cuda-13.0/targets/x86_64-linux/lib"
}

benchmark_toolkit_apply_framework_env() {
  local framework="$1"
  local pybin="$2"
  local cuda_home="${CUDA_HOME:-}"
  local python_cuda_include=""
  local python_cuda_lib=""

  benchmark_toolkit_apply_base_env

  cuda_home="${CUDA_HOME:-}"
  python_cuda_include="$(benchmark_toolkit_python_cuda_include "${pybin}" 2>/dev/null || true)"
  python_cuda_lib="$(benchmark_toolkit_python_cuda_lib "${pybin}" 2>/dev/null || true)"

  if [[ -n "${cuda_home}" ]]; then
    benchmark_toolkit_prepend_path CPATH "${cuda_home}/include"
    benchmark_toolkit_prepend_path CPATH "${cuda_home}/targets/x86_64-linux/include"
    benchmark_toolkit_prepend_path CPLUS_INCLUDE_PATH "${cuda_home}/include"
    benchmark_toolkit_prepend_path CPLUS_INCLUDE_PATH "${cuda_home}/targets/x86_64-linux/include"
  fi

  if [[ -n "${python_cuda_include}" ]]; then
    benchmark_toolkit_prepend_path CPATH "${python_cuda_include}"
    benchmark_toolkit_prepend_path CPLUS_INCLUDE_PATH "${python_cuda_include}"
  fi

  if [[ -n "${python_cuda_lib}" ]]; then
    benchmark_toolkit_prepend_path LD_LIBRARY_PATH "${python_cuda_lib}"
  fi

  case "${framework}" in
    vllm)
      export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
      ;;
    sglang)
      if [[ -n "${cuda_home}" && -x "${cuda_home}/bin/nvcc" ]]; then
        export FLASHINFER_NVCC="${FLASHINFER_NVCC:-${cuda_home}/bin/nvcc}"
      elif [[ -x /usr/bin/nvcc ]]; then
        export FLASHINFER_NVCC="${FLASHINFER_NVCC:-/usr/bin/nvcc}"
      fi
      if [[ -n "${python_cuda_include}" ]]; then
        benchmark_toolkit_append_flag FLASHINFER_EXTRA_CUDAFLAGS "-I${python_cuda_include}"
        benchmark_toolkit_append_flag FLASHINFER_EXTRA_CFLAGS "-I${python_cuda_include}"
      fi
      ;;
  esac
}
