# CUDA 12.8 System-Wide Minimal Cleanup

This is the recommended machine-level cleanup path when the benchmark harness is
already working with the rootless CUDA 12.8 toolkit and the remaining problem is
global shell pollution.

## Policy

Keep these rules:

1. Install CUDA 12.8 into `/usr/local/cuda-12.8`.
2. Do not overwrite `/usr/bin/nvcc`.
3. Do not repoint `/usr/local/cuda` yet.
4. Remove broken `/usr/local/cuda-13.0` exports from shell startup files.
5. Run a smoke test before deciding whether to make CUDA 12.8 the machine default.

## Files Added

- `consumer_gpu_benchmark/scripts/install_cuda12_8_system.sh`
- `consumer_gpu_benchmark/scripts/activate_cuda12_8_system.sh`
- `consumer_gpu_benchmark/scripts/smoke_check_cuda12_8.sh`

## Suggested Execution Order

1. Install the toolkit payload:

```bash
sudo bash consumer_gpu_benchmark/scripts/install_cuda12_8_system.sh
```

2. Remove any bad exports from shell init files such as `~/.bashrc`,
   `~/.profile`, `/etc/profile`, or `/etc/profile.d/*.sh`:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

3. If you want an explicit opt-in activation instead of changing machine
   defaults, source a small profile snippet like this:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CPATH=/usr/local/cuda-12.8/targets/x86_64-linux/include${CPATH:+:$CPATH}
export CPLUS_INCLUDE_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
```

4. Run the smoke test:

```bash
bash consumer_gpu_benchmark/scripts/smoke_check_cuda12_8.sh system
```

5. If that passes, benchmark scripts will automatically prefer
   `/usr/local/cuda-12.8` over the repo-local rootless copy.

## Notes

- The benchmark harness still works without this cleanup because
  `consumer_gpu_benchmark/scripts/toolkit_env.sh` can fall back to the rootless
  toolkit.
- This cleanup is intentionally reversible and does not modify apt-managed CUDA
  packages.
