#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOURCE_ROOT="${BENCH_ROOT}/artifacts/cuda-12.8-rootless/usr/local/cuda-12.8"
SOURCE_PKGCONFIG_ROOT="${BENCH_ROOT}/artifacts/cuda-12.8-rootless/usr/lib/pkgconfig"
TARGET_ROOT="${TARGET_ROOT:-/usr/local/cuda-12.8}"
TARGET_PKGCONFIG_ROOT="${TARGET_PKGCONFIG_ROOT:-/usr/local/lib/pkgconfig}"

log() {
  printf '[install_cuda12_8_system] %s\n' "$*"
}

die() {
  printf '[install_cuda12_8_system][error] %s\n' "$*" >&2
  exit 1
}

require_dir() {
  [[ -d "$1" ]] || die "required directory not found: $1"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

main() {
  require_dir "${SOURCE_ROOT}"
  require_dir "${SOURCE_PKGCONFIG_ROOT}"
  require_cmd rsync

  if [[ "${EUID}" -ne 0 ]]; then
    die "run this script with sudo so it can write ${TARGET_ROOT}"
  fi

  log "Installing CUDA 12.8 payload into ${TARGET_ROOT}"
  mkdir -p "${TARGET_ROOT}"
  rsync -a --delete "${SOURCE_ROOT}/" "${TARGET_ROOT}/"

  log "Installing pkg-config metadata into ${TARGET_PKGCONFIG_ROOT}"
  mkdir -p "${TARGET_PKGCONFIG_ROOT}"
  rsync -a \
    "${SOURCE_PKGCONFIG_ROOT}/cuda-12.8.pc" \
    "${SOURCE_PKGCONFIG_ROOT}/cudart-12.8.pc" \
    "${TARGET_PKGCONFIG_ROOT}/"

  log "Install finished"
  log "nvcc: ${TARGET_ROOT}/bin/nvcc"
  log "headers: ${TARGET_ROOT}/targets/x86_64-linux/include"
  log "libs: ${TARGET_ROOT}/targets/x86_64-linux/lib"
}

main "$@"
