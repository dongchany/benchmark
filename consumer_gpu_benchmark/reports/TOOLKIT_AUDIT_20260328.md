# Toolkit Audit: 2026-03-28

## Executive Summary

This machine currently exposes **three different CUDA/toolkit layers**:

1. **Driver/runtime capability**
   - `nvidia-smi` reports driver `590.48.01`
   - runtime capability shown as `CUDA Version: 13.1`

2. **System compiler toolkit**
   - `nvcc` comes from `/usr/bin/nvcc`
   - version is `11.5.119`
   - installed by Ubuntu package `nvidia-cuda-toolkit 11.5.1`

3. **Python environment CUDA user-space packages**
   - benchmark venvs ship `nvidia-cuda-runtime-cu12 12.8.90`
   - they also include modern headers such as `cuda_fp8.h`
   - they provide runtime libraries under each venv's `site-packages/nvidia/cuda_runtime/lib`

These three layers are **not aligned**, which is why framework behavior diverges.

## What Is Broken Today

### 1. Ambient `CUDA_HOME` is misleading

The shell environment currently exports:

- `CUDA_HOME=/usr/local/cuda-13.0`
- `LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:`
- `PATH=/usr/local/cuda-13.0/bin:...`

But `/usr/local/cuda-13.0` is **not** a full CUDA toolkit.

Observed structure:

- `/usr/local/cuda-13.0/targets/x86_64-linux/include`
- `/usr/local/cuda-13.0/targets/x86_64-linux/lib`

Missing:

- no `/usr/local/cuda-13.0/bin/nvcc`
- no `/usr/local/cuda-13.0/include`
- no `/usr/local/cuda-13.0/lib64`
- no `cuda.h`
- no `cuda_runtime.h`
- no `cuda_fp8.h`
- no `libcudart.so`

This directory is effectively a **cuDNN/NCCL-style add-on payload**, not a compiler toolkit root.

### 2. System `nvcc` is older than the headers expected by current JIT stacks

Installed system compiler stack:

- `nvcc 11.5.119`
- `libcudart.so.11.5.117` under `/usr/lib/x86_64-linux-gnu`

But modern frameworks in the benchmark venvs use:

- Torch CUDA 12.8 user-space packages
- FlashInfer 0.6.6
- headers such as `cuda_fp8.h`

This mismatch is why SGLang/FlashInfer runtime JIT fails: the compiler path resolves to old CUDA 11.5 tooling, while the code being compiled expects newer CUDA headers.

### 3. `~/.cache` is not reliable for benchmark toolchains

The benchmark environments reported permission/cache issues under `~/.cache`, including:

- pip cache disabled
- FlashInfer JIT log/cache writes failing

For reproducible local benchmarking, cache directories should be moved to a known writable path.

## What Works Reliably

### vLLM

vLLM can run successfully on this machine when:

- benchmark commands run outside the sandbox
- `CUDA_HOME` is treated as `/usr`
- runtime libraries come from the Python environment
- JIT/cache directories are writable

### SGLang

SGLang still fails on this machine during FlashInfer JIT for the current mainline setup.

The relevant failure is:

- `fatal error: cuda_fp8.h: No such file or directory`

Even though the header exists inside the Python environment, the current FlashInfer JIT command line is still not consuming a coherent full modern toolkit setup.

## Installed Packages Snapshot

### System packages

- `nvidia-cuda-toolkit 11.5.1`
- `nvidia-cuda-dev 11.5.1`
- `cudnn9-cuda-13 9.16.0.29-1`
- `libnccl-dev 2.28.9-1+cuda13.0`

This confirms the machine is currently a **hybrid install**:

- compiler/toolkit from Ubuntu CUDA 11.5 packages
- newer cuDNN/NCCL user-space components from CUDA 13-era packages

### Benchmark venv packages

Both benchmark environments carry their own modern NVIDIA Python wheels:

- `nvidia-cuda-runtime-cu12 12.8.90`
- `nvidia-cuda-nvrtc-cu12 12.8.93`
- `nvidia-cublas-cu12 12.8.4.1`
- `nvidia-cudnn-cu12 9.10.2.21`

Header found in venvs:

- `.../site-packages/nvidia/cuda_runtime/include/cuda_fp8.h`

## Recommended Policy For Future Tests

### Keep these rules

1. **Do not use `/usr/local/cuda-13.0` as `CUDA_HOME`.**
2. **Use `/usr` as the compiler CUDA root** when the machine is relying on `/usr/bin/nvcc`.
3. **Use framework venv CUDA headers/libs** for Python-based runtime stacks.
4. **Use benchmark-local cache directories** instead of `~/.cache`.
5. **Treat vLLM and SGLang as separate stacks** with separate venvs.

### Practical implication

Current safe default:

- `CUDA_HOME=/usr`
- `CUDACXX=/usr/bin/nvcc`
- `LD_LIBRARY_PATH` should include:
  - `/usr/local/cuda-13.0/targets/x86_64-linux/lib`
  - the active venv's `site-packages/nvidia/cuda_runtime/lib`
- `CPATH` / `CPLUS_INCLUDE_PATH` should include the active venv's:
  - `site-packages/nvidia/cuda_runtime/include`

### Remaining machine-level gap

If you want SGLang/FlashInfer JIT to be first-class on this host, the clean fix is to install a **coherent full CUDA 12.8+ or 13.x compiler toolkit**, not only runtime/devel add-ons.

Without that, vLLM is currently the more reliable framework for repeatable local tests.

## Rootless CUDA 12.8 Upgrade Applied

Because passworded `sudo` was not available for a true system-wide install in this session,
the practical upgrade was applied as a **rootless local toolkit** under:

- `/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/artifacts/cuda-12.8-rootless/usr/local/cuda-12.8`

Validated components:

- `bin/nvcc` -> `release 12.8, V12.8.93`
- `include/cuda_fp8.h` present
- `targets/x86_64-linux/lib/libcudart.so.12.8.90` present

The benchmark harness now prefers this toolkit automatically through:

- `consumer_gpu_benchmark/scripts/toolkit_env.sh`

Manual activation helper:

- `consumer_gpu_benchmark/scripts/activate_cuda12_8_rootless.sh`

## Verification Result After Upgrade

The upgraded rootless toolkit was verified in two ways:

1. A direct `nvcc` smoke compile with:
   - `#include <cuda_fp8.h>`
   - `#include <cuda_runtime.h>`
   - build succeeded

2. SGLang startup retry on `Qwen/Qwen3.5-0.8B`
   - previously failed during FlashInfer JIT
   - after switching to the rootless CUDA 12.8 toolkit, SGLang passed:
     - weight loading
     - KV cache allocation
     - CUDA graph capture
     - `/health`
     - `/v1/models`

This confirms the toolkit mismatch was a real machine-level blocker and that the new local CUDA 12.8 toolkit is a valid working baseline for future tests.
