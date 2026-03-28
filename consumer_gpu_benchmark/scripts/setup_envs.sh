#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${BENCH_ROOT}/.." && pwd)"
HOME_DIR="${HOME:-$(cd ~ && pwd)}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

VLLM_REPO="${VLLM_REPO:-${HOME_DIR}/workspace/vllm}"
SGLANG_REPO="${SGLANG_REPO:-${HOME_DIR}/workspace/sglang}"
SGLANG_PYPROJECT_DIR="${SGLANG_PYPROJECT_DIR:-${SGLANG_REPO}/python}"
TOOLKIT_ENV_SCRIPT="${SCRIPT_DIR}/toolkit_env.sh"

VLLM_ENV_DIR="${BENCH_ROOT}/envs/vllm"
SGLANG_ENV_DIR="${BENCH_ROOT}/envs/sglang"

PYTHON_BIN="python3.10"
RECREATE=0
INSTALL_VLLM=1
INSTALL_SGLANG=1
VLLM_INSTALL_MODE="${VLLM_INSTALL_MODE:-auto}"
VLLM_PYPI_SPEC="${VLLM_PYPI_SPEC:-vllm==0.18.0}"

log() {
  printf '[setup_envs] %s\n' "$*"
}

run() {
  log "RUN: $*"
  "$@"
}

usage() {
  cat <<'EOF'
Usage:
  bash consumer_gpu_benchmark/scripts/setup_envs.sh [options]

Options:
  --python <python-bin>   Python interpreter used by uv venv (default: python3.10)
  --recreate              Remove existing benchmark virtual environments before reinstalling
  --only vllm             Install only the vLLM environment
  --only sglang           Install only the SGLang environment
  -h, --help              Show this help message

What this script does:
  1. Creates isolated uv virtual environments under consumer_gpu_benchmark/envs/
  2. Installs benchmark helper packages shared by both frameworks
  3. Installs vLLM from ~/workspace/vllm in editable mode
  4. Installs SGLang from ~/workspace/sglang/python in editable mode

Notes:
  - This script intentionally keeps vLLM and SGLang in separate environments because
    the two repositories depend on different Torch / kernel stacks.
  - Editable installs are used so benchmarking always targets the current local source tree.
  - vLLM install mode is controlled by env var VLLM_INSTALL_MODE=auto|source|precompiled|pypi
    (default: auto). auto tries a source build first, then falls back to precompiled
    native extensions when the local CUDA toolchain cannot satisfy vLLM's build.
  - To install a stable PyPI release instead of the local checkout, set
    VLLM_INSTALL_MODE=pypi and optionally override VLLM_PYPI_SPEC (default: vllm==0.18.0).
  - To force a specific CUDA toolkit root for vLLM source builds, set
    VLLM_CUDA_HOME_OVERRIDE=/path/to/cuda.
  - To avoid network fetches in precompiled mode, set
    VLLM_PRECOMPILED_WHEEL_LOCATION=/absolute/path/to/vllm.whl.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --python)
        shift
        if [[ $# -eq 0 ]]; then
          log "ERROR: --python requires a value"
          exit 1
        fi
        PYTHON_BIN="$1"
        ;;
      --recreate)
        RECREATE=1
        ;;
      --only)
        shift
        if [[ $# -eq 0 ]]; then
          log "ERROR: --only requires one of: vllm, sglang"
          exit 1
        fi
        case "$1" in
          vllm)
            INSTALL_VLLM=1
            INSTALL_SGLANG=0
            ;;
          sglang)
            INSTALL_VLLM=0
            INSTALL_SGLANG=1
            ;;
          *)
            log "ERROR: unsupported value for --only: $1"
            exit 1
            ;;
        esac
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        log "ERROR: unknown argument: $1"
        usage
        exit 1
        ;;
    esac
    shift
  done
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: required command not found: $1"
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    log "ERROR: required directory not found: $1"
    exit 1
  fi
}

recreate_env_if_requested() {
  local env_dir="$1"
  if [[ ${RECREATE} -eq 1 && -d "${env_dir}" ]]; then
    log "Removing existing environment: ${env_dir}"
    rm -rf -- "${env_dir}"
  fi
}

create_uv_env() {
  local env_dir="$1"
  if [[ ! -x "${env_dir}/bin/python" ]]; then
    run "${UV_BIN}" venv --python "${PYTHON_BIN}" "${env_dir}"
  else
    log "Reusing existing uv environment: ${env_dir}"
  fi
}

bootstrap_common_tools() {
  local pybin="$1"
  run "${UV_BIN}" pip install --python "${pybin}" -U pip setuptools wheel packaging
  run "${UV_BIN}" pip install --python "${pybin}" psutil requests aiohttp pandas tabulate
}

resolve_vllm_cuda_home() {
  local nvcc_path=""
  local resolved_nvcc=""
  local candidate=""

  if [[ -n "${VLLM_CUDA_HOME_OVERRIDE:-}" ]]; then
    printf '%s\n' "${VLLM_CUDA_HOME_OVERRIDE}"
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

install_vllm_source() {
  local pybin="$1"
  local cuda_home=""
  local cudacxx=""

  cuda_home="$(resolve_vllm_cuda_home || true)"
  if [[ -n "${cuda_home}" ]]; then
    cudacxx="${cuda_home}/bin/nvcc"
    log "Installing vLLM from source with CUDA_HOME=${cuda_home}"
    run env CUDA_HOME="${cuda_home}" CUDACXX="${cudacxx}" \
      "${UV_BIN}" pip install --python "${pybin}" -e "${VLLM_REPO}"
  else
    log "Installing vLLM from source without CUDA_HOME override"
    run "${UV_BIN}" pip install --python "${pybin}" -e "${VLLM_REPO}"
  fi
}

install_vllm_precompiled() {
  local pybin="$1"
  local -a env_args

  env_args=("VLLM_USE_PRECOMPILED=1")
  if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
    env_args+=("VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION}")
    log "Installing vLLM with precompiled extensions from ${VLLM_PRECOMPILED_WHEEL_LOCATION}"
  else
    log "Installing vLLM with precompiled extensions fetched by setup.py"
  fi

  run env "${env_args[@]}" \
    "${UV_BIN}" pip install --python "${pybin}" -e "${VLLM_REPO}"
}

install_vllm_pypi() {
  local pybin="$1"
  log "Installing vLLM from PyPI spec ${VLLM_PYPI_SPEC}"
  run "${UV_BIN}" pip install --python "${pybin}" "${VLLM_PYPI_SPEC}"
}

install_vllm() {
  local pybin="${VLLM_ENV_DIR}/bin/python"
  log "Installing vLLM into ${VLLM_ENV_DIR} (mode=${VLLM_INSTALL_MODE})"
  bootstrap_common_tools "${pybin}"

  case "${VLLM_INSTALL_MODE}" in
    source)
      install_vllm_source "${pybin}"
      ;;
    precompiled)
      install_vllm_precompiled "${pybin}"
      ;;
    pypi)
      install_vllm_pypi "${pybin}"
      ;;
    auto)
      if install_vllm_source "${pybin}"; then
        log "vLLM source install succeeded"
      else
        log "vLLM source install failed, retrying with precompiled native extensions"
        install_vllm_precompiled "${pybin}"
      fi
      ;;
    *)
      log "ERROR: unsupported VLLM_INSTALL_MODE=${VLLM_INSTALL_MODE}; expected auto, source, precompiled, or pypi"
      exit 1
      ;;
  esac

  run "${pybin}" -c "import vllm; print('vllm_import_ok', vllm.__file__)"
}

install_sglang() {
  local pybin="${SGLANG_ENV_DIR}/bin/python"
  log "Installing SGLang into ${SGLANG_ENV_DIR}"
  bootstrap_common_tools "${pybin}"
  run "${UV_BIN}" pip install --python "${pybin}" -e "${SGLANG_PYPROJECT_DIR}"
  run "${pybin}" -c "import sglang; print('sglang_import_ok', sglang.__file__)"
}

write_env_manifest() {
  local manifest_path="${BENCH_ROOT}/envs/env_manifest.txt"
  log "Writing environment manifest to ${manifest_path}"
  {
    printf 'timestamp=%s\n' "$(date -Iseconds)"
    printf 'python_bin=%s\n' "${PYTHON_BIN}"
    printf 'workspace_root=%s\n' "${WORKSPACE_ROOT}"
    printf 'vllm_repo=%s\n' "${VLLM_REPO}"
    printf 'sglang_repo=%s\n' "${SGLANG_REPO}"
    printf 'install_vllm=%s\n' "${INSTALL_VLLM}"
    printf 'install_sglang=%s\n' "${INSTALL_SGLANG}"
    printf 'vllm_install_mode=%s\n' "${VLLM_INSTALL_MODE}"
    if [[ -n "${VLLM_CUDA_HOME_OVERRIDE:-}" ]]; then
      printf 'vllm_cuda_home_override=%s\n' "${VLLM_CUDA_HOME_OVERRIDE}"
    fi
    if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
      printf 'vllm_precompiled_wheel_location=%s\n' "${VLLM_PRECOMPILED_WHEEL_LOCATION}"
    fi
    printf 'vllm_pypi_spec=%s\n' "${VLLM_PYPI_SPEC}"
    if [[ -x "${VLLM_ENV_DIR}/bin/python" ]]; then
      printf 'vllm_python=%s\n' "${VLLM_ENV_DIR}/bin/python"
      "${VLLM_ENV_DIR}/bin/python" --version
    fi
    if [[ -x "${SGLANG_ENV_DIR}/bin/python" ]]; then
      printf 'sglang_python=%s\n' "${SGLANG_ENV_DIR}/bin/python"
      "${SGLANG_ENV_DIR}/bin/python" --version
    fi
  } >"${manifest_path}"
}

main() {
  parse_args "$@"

  if [[ -f "${TOOLKIT_ENV_SCRIPT}" ]]; then
    # shellcheck source=/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/scripts/toolkit_env.sh
    source "${TOOLKIT_ENV_SCRIPT}"
    benchmark_toolkit_apply_base_env
  fi

  require_cmd git
  if [[ -z "${UV_BIN}" ]]; then
    log "ERROR: required command not found: uv"
    exit 1
  fi
  require_dir "${BENCH_ROOT}"
  require_dir "${VLLM_REPO}"
  require_dir "${SGLANG_REPO}"
  require_dir "${SGLANG_PYPROJECT_DIR}"

  mkdir -p "${BENCH_ROOT}/envs"

  if [[ ${INSTALL_VLLM} -eq 1 ]]; then
    recreate_env_if_requested "${VLLM_ENV_DIR}"
    create_uv_env "${VLLM_ENV_DIR}"
    install_vllm
  fi

  if [[ ${INSTALL_SGLANG} -eq 1 ]]; then
    recreate_env_if_requested "${SGLANG_ENV_DIR}"
    create_uv_env "${SGLANG_ENV_DIR}"
    install_sglang
  fi

  write_env_manifest

  log "Environment setup completed."
}

main "$@"
