# vLLM vs SGLang：消费级双 3080 Ti 本机降级对比报告

## 1. 本次降级说明

原计划是对本机上的 `/home/dong/workspace/vllm` 与 `/home/dong/workspace/sglang` 做较完整的 serving / offline / micro / 长上下文 / 双卡对比。

但在实际执行中，遇到了两个会显著拉长周期的环境性问题：

1. **vLLM 源码可编辑安装被错误的 `CUDA_HOME=/usr/local/cuda-13.0` 阻塞**，后续改为使用预编译 wheel 成功落地。
2. **SGLang 在启动阶段会触发 flashinfer JIT 编译，仍然硬编码调用 `/usr/local/cuda-13.0/bin/nvcc`，而该路径不存在可执行 `nvcc`**，导致服务在 CUDA graph capture 阶段失败。

因此本报告改为：

- 以 **vLLM 已跑通的真实本机结果** 为主；
- 以 **SGLang 在同机同模型上的真实启动日志与失败点** 为辅；
- 给出一个可用于当前机器决策的**粗粒度结论**。

## 2. 测试环境

- 机器：2 x RTX 3080 Ti，12GB VRAM / 卡
- 单卡测试偏向 GPU1，避免 GPU0 桌面负载干扰
- 模型：`Qwen/Qwen3-0.6B`
- 本地模型路径：`/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`

## 3. 已完成的真实执行

### 3.1 vLLM

已完成：

- 预编译扩展方式安装 vLLM benchmark 环境
- 单卡服务启动成功
- online smoke benchmark 成功
- offline smoke benchmark 成功

关键元数据文件：

- `consumer_gpu_benchmark/results/raw/vllm_qwen3-0.6b_serve_20260328_093209.meta.json`
- `consumer_gpu_benchmark/results/raw/vllm_qwen3-0.6b_online_short_smoke_20260328_093304_r1_rr1_mc1.meta.json`
- `consumer_gpu_benchmark/results/raw/vllm_qwen3-0.6b_offline_short_smoke_20260328_093544_r1_tp1.meta.json`

### 3.2 SGLang

已完成：

- 单独隔离环境安装成功
- 服务启动进入到模型加载、KV cache 分配、CUDA graph capture 阶段
- **在 flashinfer JIT 编译阶段失败**，未能完成 `/health` 与 `/v1/models` readiness

关键失败日志：

- `consumer_gpu_benchmark/logs/sglang/sglang_qwen3-0.6b_serve_20260328_094833.log`

## 4. vLLM 实测结果

### 4.1 online smoke

场景：

- input_len = 64
- output_len = 16
- num_prompts = 16
- request_rate = 1 RPS
- max_concurrency = 1

实测结果：

- Successful requests: **16**
- Failed requests: **0**
- Benchmark duration: **16.06 s**
- Request throughput: **1.00 req/s**
- Output token throughput: **15.94 tok/s**
- Total token throughput: **79.72 tok/s**
- Mean TTFT: **44.09 ms**
- Median TTFT: **30.25 ms**
- P99 TTFT: **244.12 ms**
- Mean TPOT: **3.72 ms**
- Median TPOT: **2.92 ms**
- Mean ITL: **3.72 ms**
- Median ITL: **2.96 ms**

这说明在这台机器上，vLLM 至少已经具备：

- 正常拉起服务
- 正常对外提供 OpenAI 兼容接口
- 基本可接受的短请求首 token 延迟
- 稳定的短请求 smoke 吞吐

### 4.2 offline smoke

场景：

- input_len = 64
- output_len = 16
- num_prompts = 32
- 单卡 GPU1

实测结果：

- Throughput: **167.08 requests/s**
- Total token throughput: **13366.41 tok/s**
- Output token throughput: **2673.28 tok/s**
- Total prompt tokens: **2048**
- Total output tokens: **512**

补充观察：

- engine 初始化、profile、KV cache 建立、warmup 总计约 **13.56 s**
- 日志显示可用 KV cache 约 **81,264 tokens**
- 这个模型尺寸很小，因此此结果更接近“框架基础开销 + 小模型执行效率”的参考值，而不是大模型极限值

## 5. SGLang 实测状态

SGLang 并不是“完全不能运行”，而是**已经跑到相当后面才失败**，这点很重要。

从日志看，SGLang 已完成：

- 参数解析
- 分布式初始化
- 模型权重加载
- KV cache 分配
- 进入 CUDA graph capture

但在 flashinfer 动态 JIT 编译时失败，关键报错是：

- 调用了 **`/usr/local/cuda-13.0/bin/nvcc`**
- 实际错误：**`/bin/sh: 1: /usr/local/cuda-13.0/bin/nvcc: not found`**

因此，当前机器上的 SGLang 问题更准确地说是：

> **不是模型不适配，也不是显存不够，而是运行期内核 JIT 编译路径绑定到了一个错误的 CUDA toolkit 路径。**

这与 vLLM 前面遇到的问题本质相似：

- 都受到了这台机器上错误 `CUDA_HOME` / CUDA toolkit 路径布局的影响；
- 但 vLLM 可以通过**预编译 wheel** 绕过；
- SGLang 当前这条执行链路仍然会在运行时触发 **flashinfer JIT**，所以没有像 vLLM 那样轻易绕开。

## 6. 当前这台机器上的粗粒度结论

### 6.1 如果你的目标是“先跑起来、先稳定服务”

**vLLM 当前更占优。**

原因不是抽象性能宣传，而是本机实测上：

- vLLM 已成功完成安装、启动、online smoke、offline smoke；
- SGLang 目前卡在运行期 JIT 编译路径错误，尚未形成可用服务。

### 6.2 如果你的目标是“理论上的调度/Kernel 潜力”

**SGLang 仍然值得继续投入，但前提是先修好本机 CUDA/JIT 路径。**

从它已经跑到：

- 权重加载完成
- KV cache 建立完成
- 开始做 CUDA graph capture

可以看出框架主体流程没有根本性障碍。问题集中在：

- flashinfer 的 JIT 编译工具链选择
- 对本机 CUDA toolkit 路径的假设过强

这意味着一旦把 JIT 的 `nvcc` 路径问题修正，SGLang 很可能就能继续跑完。

### 6.3 面向消费级 12GB 显卡的实际体验判断

基于这轮降级 benchmark，我会给出一个**偏工程落地**的判断：

#### vLLM 的现实优势

1. **可用性更强**：至少在这台机器上，已经形成可跑通链路。
2. **对“本机环境不整洁”更有容错**：通过预编译 wheel 规避了源码编译依赖。
3. **OpenAI 接口与 benchmark 工具链更容易直接用起来**。

#### vLLM 的现实不足

1. 在这台 12GB 卡上，offline benchmark 如果服务已占用显存，会直接因空闲显存不足失败。
2. 源码安装对 CUDA 工具链也敏感，只是这次通过预编译方案绕过去了。

#### SGLang 的现实优势

1. 从启动过程看，内核/图捕获路径做得更激进，理论上对高性能路径投入更深。
2. 一旦环境正确，通常更值得在吞吐/延迟极致优化场景中继续挖。

#### SGLang 的现实不足

1. **当前本机环境下可用性明显差于 vLLM**。
2. 运行时 JIT 对 CUDA toolkit 路径依赖更强，消费级本机环境稍微“脏”一点就容易卡住。
3. 在 benchmark 自动化阶段，失败恢复和环境要求更苛刻。

## 7. 建议的降级决策

如果你的目标只是“先大概看下谁更适合这台机器”，我建议直接按下面结论用：

### 建议 A：当前就要用

优先用 **vLLM**。

理由：

- 已经本机跑通；
- online / offline smoke 都有真实结果；
- 风险更低，投入产出比更高。

### 建议 B：后续再花半天继续折腾

可以继续补修 **SGLang**，重点不是模型也不是 benchmark 脚本，而是：

- 统一 `CUDA_HOME`
- 让 flashinfer JIT 找到真正可执行的 `nvcc`
- 必要时关闭部分 CUDA graph / JIT 路径做一次保守启动验证

## 8. 最终一句话结论

**在你这台消费级双 3080 Ti 机器上，当前阶段的粗对比结果是：vLLM 已经证明“能稳定跑起来并给出可用吞吐/延迟结果”，而 SGLang 暂时被运行时 CUDA JIT 路径问题卡住，因此如果现在就要落地使用，优先选 vLLM。**
