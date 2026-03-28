# vLLM vs SGLang Formal Comparison: 2026-03-28

## Executive Summary

This report consolidates the benchmark results collected on `2026-03-28` for the
local machine with dual RTX 3080 Ti 12GB GPUs.

The final conclusions are:

1. The machine-level CUDA/toolkit situation is now usable and defaults to
   CUDA 12.8 in new shells.
2. For single-GPU small and medium-small online serving, SGLang currently has
   the better latency profile on this host.
3. For dual-GPU `Qwen3.5-4B`, SGLang is both faster and easier to run than the
   original local vLLM dev environment.
4. `Qwen3.5-9B` is confirmed runnable on this host via
   `Transformers + Accelerate + device_map="auto" + bf16`, but the benchmark
   serving comparison for 9B is still incomplete.
5. The original benchmark vLLM environment was not a stable baseline for dual-GPU
   `Qwen3.5-4B`; a clean stable `vllm 0.18.0` environment was required.

## Machine And Toolkit State

### Hardware

- GPUs: `2 x NVIDIA GeForce RTX 3080 Ti`
- VRAM: `12 GB` per GPU
- Single-GPU tests prefer `GPU1`
- Dual-GPU tests use `GPU0,1`

### CUDA/toolkit status after cleanup

The following cleanup and validation work was completed:

- CUDA 12.8 installed to `/usr/local/cuda-12.8`
- shell pollution from `/usr/local/cuda-13.0` removed
- `/etc/profile.d/cuda13.sh` corrected to use `/usr/local/cuda-12.8`
- `/home/dong/.bashrc` updated so interactive non-login shells also default to
  CUDA 12.8

Validated results:

- `CUDA_HOME=/usr/local/cuda-12.8`
- `nvcc=/usr/local/cuda-12.8/bin/nvcc`
- `nvcc --version = 12.8`

Related materials:

- [CUDA12_8_SYSTEM_WIDE_MIN_CLEANUP.md](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/reports/CUDA12_8_SYSTEM_WIDE_MIN_CLEANUP.md)
- [TOOLKIT_AUDIT_20260328.md](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/reports/TOOLKIT_AUDIT_20260328.md)

## Benchmark Scope

### Completed comparisons

- `Qwen3.5-0.8B` online: `vLLM vs SGLang`
- `Qwen3.5-0.8B` offline: `vLLM completed`, `SGLang offline harness stalled`
- `Qwen3-1.7B` online: `vLLM vs SGLang`
- `Qwen3.5-4B` dual-GPU online:
  - `SGLang` completed on the original benchmark environment
  - `vLLM` completed only after switching to a clean stable `vllm 0.18.0` env
- `Qwen3.5-9B` dual-GPU HF baseline smoke: completed

### Not completed

- `Qwen3.5-9B` formal serving comparison for `vLLM vs SGLang`
- stable `SGLang offline` result on the current harness

## Important Corrections To Earlier Assumptions

The following earlier assumptions are now known to be incomplete or incorrect:

1. "`Qwen3.5-4B` cannot run on this machine."
   Corrected conclusion:
   - single-GPU serving is not a good baseline on this 12GB card
   - dual-GPU `SGLang` works
   - dual-GPU `vLLM` also works when using a clean stable environment

2. "`Qwen3.5-9B` cannot run on this machine."
   Corrected conclusion:
   - HF baseline proves that `Qwen3.5-9B` is runnable on dual 3080 Ti
   - the unresolved issue is serving-engine comparability, not model viability

3. "vLLM dual-GPU failure proves the framework is unusable here."
   Corrected conclusion:
   - the local benchmark vLLM dev build was the unstable component
   - clean `vllm 0.18.0` successfully runs dual-GPU `Qwen3.5-4B`

## Results

### 1. Qwen3.5-0.8B Online

Scenario:

- `online_short_latency`
- `input_len=256`
- `output_len=32`
- `num_prompts=200`
- `request_rate=2`
- `max_concurrency=4`

vLLM mean over 2 runs:

- `request throughput`: `1.9974 req/s`
- `output throughput`: `63.9158 tok/s`
- `mean TTFT`: `46.47 ms`
- `p99 TTFT`: `258.50 ms`
- `mean ITL`: `3.06 ms`
- `p99 ITL`: `3.57 ms`

SGLang mean over 2 runs:

- `request throughput`: `1.9973 req/s`
- `output throughput`: `63.9138 tok/s`
- `mean TTFT`: `29.10 ms`
- `p99 TTFT`: `47.26 ms`
- `mean ITL`: `3.74 ms`
- `p99 ITL`: `4.58 ms`

Interpretation:

- Throughput is effectively tied.
- SGLang is materially better on first-token latency and TTFT tail.
- vLLM is better on token-to-token latency.

Files:

- [vllm_qwen3.5-0.8b_online_short_latency_20260328_160927_r1_rr2_mc4.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/vllm_qwen3.5-0.8b_online_short_latency_20260328_160927_r1_rr2_mc4.jsonl)
- [vllm_qwen3.5-0.8b_online_short_latency_20260328_161118_r2_rr2_mc4.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/vllm_qwen3.5-0.8b_online_short_latency_20260328_161118_r2_rr2_mc4.jsonl)
- [sglang_qwen3.5-0.8b_online_short_latency_20260328_161910_r1_rr2_mc4.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/sglang_qwen3.5-0.8b_online_short_latency_20260328_161910_r1_rr2_mc4.jsonl)
- [sglang_qwen3.5-0.8b_online_short_latency_20260328_162112_r2_rr2_mc4.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/sglang_qwen3.5-0.8b_online_short_latency_20260328_162112_r2_rr2_mc4.jsonl)

### 2. Qwen3.5-0.8B Offline

vLLM mean over 2 runs:

- `request throughput`: `22.83 req/s`
- `total throughput`: `26298.18 tok/s`

SGLang:

- current offline harness did not produce a usable result file
- the run stalled and was terminated

Interpretation:

- vLLM offline benchmarking is currently the more reliable path in this repo
- SGLang offline results should not be treated as complete

Files:

- [vllm_qwen3.5-0.8b_offline_medium_20260328_162339_r1_tp1.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/vllm_qwen3.5-0.8b_offline_medium_20260328_162339_r1_tp1.jsonl)
- [vllm_qwen3.5-0.8b_offline_medium_20260328_162431_r2_tp1.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/vllm_qwen3.5-0.8b_offline_medium_20260328_162431_r2_tp1.jsonl)

### 3. Qwen3-1.7B Online

Scenario:

- `online_short_latency`
- `input_len=256`
- `output_len=32`
- `num_prompts=200`
- `request_rate=2`
- `max_concurrency=4`

vLLM:

- `request throughput`: `2.00 req/s`
- `output throughput`: `63.86 tok/s`
- `mean TTFT`: `34.39 ms`
- `p99 TTFT`: `83.30 ms`
- `mean ITL`: `6.06 ms`
- `p99 ITL`: `21.04 ms`

SGLang:

- `request throughput`: `1.9963 req/s`
- `output throughput`: `63.88 tok/s`
- `mean TTFT`: `24.58 ms`
- `p99 TTFT`: `69.94 ms`
- `mean ITL`: `5.67 ms`
- `p99 ITL`: `16.79 ms`

Interpretation:

- On this host and this model size, SGLang wins both TTFT and ITL.
- This makes `Qwen3-1.7B` a useful stable supporting data point beyond `0.8B`.

Files:

- [vllm_qwen3-1.7b_online_short_latency_20260328_165910_r1_rr2_mc4.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/vllm_qwen3-1.7b_online_short_latency_20260328_165910_r1_rr2_mc4.jsonl)
- [sglang_qwen3-1.7b_online_short_latency_20260328_170237_r1_rr2_mc4.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/sglang_qwen3-1.7b_online_short_latency_20260328_170237_r1_rr2_mc4.jsonl)

### 4. Qwen3.5-4B Dual-GPU Online

Scenario:

- `online_short_latency`
- `input_len=256`
- `output_len=32`
- `num_prompts=200`
- `request_rate=2`
- `max_concurrency=4`
- `devices=0,1`
- `tp_size=2`

#### SGLang

Formal run:

- `request throughput`: `1.9382 req/s`
- `output throughput`: `62.0215 tok/s`
- `mean TTFT`: `1032.06 ms`
- `p99 TTFT`: `1485.04 ms`
- `mean ITL`: `12.58 ms`
- `p99 ITL`: `15.59 ms`

Files:

- [sglang_qwen3.5-4b_online_short_smoke_20260328_172521_r0_rr1_mc1.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/sglang_qwen3.5-4b_online_short_smoke_20260328_172521_r0_rr1_mc1.jsonl)
- [sglang_qwen3.5-4b_online_short_latency_20260328_172600_r1_rr2_mc4.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/sglang_qwen3.5-4b_online_short_latency_20260328_172600_r1_rr2_mc4.jsonl)

#### vLLM clean stable environment

The original local benchmark vLLM dev environment failed during dual-GPU
initialization. A separate clean stable environment was created:

- env path: `/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/envs/vllm_clean`
- package: `vllm 0.18.0`

Mean over 2 formal runs:

- `request throughput`: `1.0756 req/s`
- `output throughput`: `34.4200 tok/s`
- `mean TTFT`: `2795.53 ms`
- `p99 TTFT`: `3137.94 ms`
- `mean ITL`: `27.57 ms`
- `p99 ITL`: `32.51 ms`

Files:

- [vllm_clean_qwen3.5-4b_online_short_smoke_20260328_180152_r0_rr1_mc1.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/vllm_clean_qwen3.5-4b_online_short_smoke_20260328_180152_r0_rr1_mc1.jsonl)
- [vllm_clean_qwen3.5-4b_online_short_latency_20260328_215211_r1_rr2_mc4.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/vllm_clean_qwen3.5-4b_online_short_latency_20260328_215211_r1_rr2_mc4.jsonl)
- [vllm_clean_qwen3.5-4b_online_short_latency_20260328_215540_r2_rr2_mc4.jsonl](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/results/raw/vllm_clean_qwen3.5-4b_online_short_latency_20260328_215540_r2_rr2_mc4.jsonl)

Interpretation:

- Dual-GPU `Qwen3.5-4B` is now confirmed runnable for both frameworks, but not
  under the same vLLM environment.
- On the current host and current runnable software stack, SGLang is
  substantially faster than the clean stable vLLM environment:
  - better TTFT by roughly `1.76 s`
  - better ITL by roughly `15 ms`
  - higher output throughput by roughly `27.6 tok/s`

### 5. Qwen3.5-9B HF Baseline

HF baseline used:

- `Transformers + Accelerate`
- `device_map="auto"`
- `bf16`

Validated result:

- model loaded successfully on `GPU0,1`
- `HF_LOAD_S=5.189`
- `HF_INFER_S=1.020`
- generation succeeded

Practical interpretation:

- `Qwen3.5-9B` is not too large for this machine in principle
- unresolved problems are serving-engine path issues, not raw model viability

Related script:

- [run_hf_smoke.sh](/home/dong/workspace/tmp/benchmark/consumer_gpu_benchmark/scripts/run_hf_smoke.sh)

## Framework Assessment

### SGLang

Current strengths on this host:

- better online latency on `0.8B`, `1.7B`, and `4B`
- successful dual-GPU `Qwen3.5-4B` serving on the original benchmark stack

Current weakness:

- offline harness is still not reliable in this repo/environment

### vLLM

Current strengths on this host:

- reliable single-GPU online and offline on small models
- clean stable `0.18.0` environment can run dual-GPU `Qwen3.5-4B`

Current weaknesses:

- original local dev environment is not a stable dual-GPU baseline
- even after switching to stable `vllm 0.18.0`, dual-GPU `4B` serving is much
  slower than SGLang under the tested parameters

## Final Recommendations

### If the goal is immediate formal serving comparison on this host

Use these completed conclusions:

1. `Qwen3.5-0.8B`: SGLang wins TTFT, vLLM wins ITL, throughput tied.
2. `Qwen3-1.7B`: SGLang wins TTFT and ITL.
3. `Qwen3.5-4B dual-GPU`: SGLang wins clearly on latency and throughput.

### If the goal is to continue improving benchmark completeness

Next high-value tasks are:

1. bring `Qwen3.5-9B` serving comparison to completion
2. stabilize `SGLang offline` harness
3. decide whether the original local vLLM dev environment should remain the
   benchmark default, or whether stable-wheel mode should become the default for
   dual-GPU tests

## Bottom Line

On this machine and with the currently validated runnable stacks, the most
defensible summary is:

- `SGLang` is the stronger online serving result on the completed comparisons
- `vLLM` remains usable, but its dual-GPU story depends heavily on using a
  clean stable environment instead of the current local dev build
- `HF` is valuable as a viability baseline and proves that `Qwen3.5-9B` is
  runnable even though the full serving comparison is not yet complete
