# vLLM vs SGLang 消费级显卡 Benchmark 计划

## 1. 目标

本计划用于在当前机器上，对以下两套推理框架做尽量公平、可复现、面向消费级显卡场景的对比：

- vLLM 源码仓库：`~/workspace/vllm`
- SGLang 源码仓库：`~/workspace/sglang`
- 模型来源：`~/xilinx/huggingface/hub`

目标不是复现官方最佳成绩，而是回答更贴近本机实际使用的问题：

1. 在 2 x RTX 3080 Ti 12GB 上，谁更容易跑起来。
2. 在单卡消费级场景下，谁的延迟、吞吐、稳定性更好。
3. 在双卡场景下，谁的扩展性更好。
4. 在长上下文、并发和前缀复用场景下，谁更占优。
5. 哪些优势来自默认策略，哪些优势来自额外调参。

---

## 2. 已确认环境

### 硬件
- GPU: 2 x NVIDIA GeForce RTX 3080 Ti
- 单卡显存: 12GB
- Driver: 590.48.01
- CUDA: 13.1

### 仓库版本
- vLLM: branch `main`, commit `148a5c122`
- SGLang: branch `main`, commit `a27651d5e`

### 工具链
- `uv 0.9.11`
- `gcc`, `g++`, `cmake`, `ninja`, `nvcc` 可用

### 候选模型快照
本地 `~/xilinx/huggingface/hub` 中可见候选包括：
- Qwen3.5-0.8B
- Qwen3-1.7B
- Qwen3.5-4B
- Qwen3-4B-Instruct-2507
- Qwen3.5-9B
- Qwen3.5-35B-A3B
- 以及更大的 14B / 35B-FP8 等快照

本次优先选择“**最新且典型**”的模型，而不是单纯按最小参数量优先：
- 小模型代表：`Qwen3.5-0.8B`
- 单卡实用代表：`Qwen3-4B-Instruct-2507`
- 双卡扩展代表：`Qwen3.5-9B`
- 可选可行性附录：`Qwen3.5-35B-A3B`

---

## 3. 设计原则

### 3.1 公平性原则
为了避免把“框架差异”与“模型规模差异 / 量化差异 / 多机技巧差异”混在一起，本次对比采用：

- 同一模型、同一请求分布、同一输入输出长度。
- 单卡优先，双卡作为扩展性补充项。
- 优先使用两边都原生支持、且不需要额外特殊量化路径的模型。
- 先做“默认可用配置”的比较，再做“轻量调优配置”的比较。

### 3.2 面向消费级显卡原则
这台机器的核心限制不是算力，而是：

- 12GB 显存较紧张。
- GPU0 被桌面/Xorg/VSCode 占用，不适合拿来做绝对干净的显存测试。
- 因此优先将 benchmark 绑定到 GPU1；若需要双卡，则记录 GPU0 存在桌面噪声。

### 3.3 可解释性原则
除了结果数字，本次还会保留：

- 环境信息
- 启动命令
- benchmark 原始 JSON/JSONL
- 服务日志
- GPU 采样日志
- 最终 Markdown 报告

这样后续可以追溯“为什么这个框架更快/更慢”。

---

## 4. 模型选择策略

### 4.1 主测试模型
考虑到 12GB 显存和公平性，本次分三档：

#### A. 烟雾测试模型
- `Qwen3.5-0.8B`
- 作用：快速验证环境、命令、服务与 benchmark harness 是否工作。
- 原因：这是本地已有的较新小模型，启动快、失败成本低，同时比旧的 0.6B 更接近当前常用版本。

#### B. 单卡主对比模型
- `Qwen3.5-0.8B`
- `Qwen3-4B-Instruct-2507`
- 作用：分别覆盖“低延迟小模型”和“单卡典型实用模型”两类消费级部署场景。
- 原因：
  - 0.8B 适合看框架本身调度、首 token 延迟和轻负载吞吐。
  - 4B instruct 更贴近日常问答、代码助手和通用本地助手负载。

#### C. 双卡扩展模型
- `Qwen3.5-9B`
- 作用：观察 TP=2 时双框架的可扩展性与额外开销。
- 原因：这是本地已有的较新大一点 dense 模型，比旧的 8B 更符合“最新且典型”的选择标准。

#### D. 可行性附录模型
- `Qwen3.5-35B-A3B`
- 作用：记录在双 3080 Ti 上的启动可行性、失败模式和工程适配成本。
- 原因：即使无法稳定跑完整 benchmark，启动成功或失败本身也有参考价值。

### 4.2 暂不纳入主对比的模型
- `Qwen3-0.6B`
- `Qwen3-1.7B`
- `Qwen3.5-4B`
- `Qwen3-8B`
- `Qwen3-14B`
- `Qwen3.5-35B-A3B-FP8`

原因：
- 很可能需要额外量化、分片、特殊 kernel 或更激进的显存策略。
- 这些设置更容易把“工程适配能力”与“框架核心效率”混在一起。
- 若主测试完成且环境稳定，再考虑把 9B/14B 作为附录实验。

---

## 5. 对齐的运行方式

### 5.1 vLLM
已确认：
- 在线 benchmark CLI 为 `vllm bench serve`
- 离线吞吐 benchmark CLI 为 `vllm bench throughput`
- 单批次延迟 benchmark CLI 为 `vllm bench latency`
- 服务启动入口为 `vllm serve`

### 5.2 SGLang
已确认：
- 服务启动入口为 `sglang serve`
- 在线 benchmark CLI 为 `python -m sglang.bench_serving`
- 离线吞吐 benchmark CLI 为 `python -m sglang.bench_offline_throughput`
- 单批次服务 benchmark CLI 为 `python -m sglang.bench_one_batch_server`

### 5.3 统一比较口径
| 维度 | vLLM | SGLang | 统一口径 |
| --- | --- | --- | --- |
| 在线服务 | `vllm serve` + `vllm bench serve` | `sglang serve` + `sglang.bench_serving` | 同模型、同随机请求分布、同 RPS、同并发上限 |
| 离线吞吐 | `vllm bench throughput` | `sglang.bench_offline_throughput` | 同随机数据集、同输入输出长度、同 prompt 数 |
| 单批次微基准 | `vllm bench latency` + 补充服务端单批次测试 | `sglang.bench_one_batch_server` | 重点观察 prefill / decode / batch-size 敏感性 |
| 健康检查 | `/health` 与 `/v1/models` | `/health` / `/health_generate` / `/v1/models` | 统一使用 `/v1/models` 验证模型就绪 |

---

## 6. Benchmark 维度

### 6.1 维度 A：启动与可用性
记录：
- 创建独立 `uv` 环境耗时
- 安装依赖耗时
- 首次编译/冷启动耗时
- 服务启动到 `/v1/models` 可用耗时
- 常见错误与修复成本

意义：
这是消费级显卡用户最在意的第一层体验，往往比极限吞吐更重要。

### 6.2 维度 B：在线服务性能
主要指标：
- TTFT（首 token 延迟）
- ITL（inter-token latency）
- TPOT / token latency
- E2E latency
- request throughput
- output token throughput
- failed requests

测试场景：
1. **低并发低延迟**：RPS = 1, 2
2. **中等负载**：RPS = 4, 8
3. **高压负载**：RPS = 16（若模型和服务稳定）
4. **并发上限控制**：`max_concurrency = 1, 4, 8, 16`

数据集：
- 优先使用随机数据集，确保无需额外下载外部数据。
- 统一输入输出长度，避免 ShareGPT 数据分布偏差。

推荐参数组：
- short: input 256 / output 32
- medium: input 1024 / output 128
- long: input 4096 / output 256

### 6.3 维度 C：离线吞吐
主要指标：
- 总耗时
- request throughput
- input token throughput
- output token throughput
- total token throughput

测试场景：
- random dataset
- 固定 prompt 数，例如 200 / 500 / 1000
- 长度组合：
  - 256 -> 32
  - 1024 -> 128
  - 4096 -> 256

意义：
更接近批处理 / 批量离线生成 / rerank 前处理的场景。

### 6.4 维度 D：单批次微基准
主要指标：
- 单 batch prefill latency
- 单 batch decode latency
- median decode throughput
- batch-size scaling

测试场景：
- batch = 1, 2, 4, 8, 16
- input = 256, 1024, 4096
- output = 1, 16, 64

意义：
这个维度最容易定位：
- 哪个框架 prefill 更强
- 哪个框架 decode 更强
- 哪个框架在小 batch 与大 batch 间退化更明显

### 6.5 维度 E：前缀复用 / cache 友好场景
主要指标：
- 吞吐提升幅度
- TTFT 下降幅度
- cache hit 相关指标（若框架暴露）

测试场景：
- generated-shared-prefix
- group 数固定，问题后缀变化

意义：
这类场景非常贴近日常多轮助手、系统 prompt 固定、模板化任务。

### 6.6 维度 F：长上下文与显存压力
主要指标：
- 是否能稳定启动
- 4096 / 8192 上下文是否可运行
- TTFT 与吞吐退化曲线
- OOM / CUDA error / fallback 情况

测试场景：
- `Qwen3.5-0.8B` 优先做 4096、8192
- `Qwen3-4B-Instruct-2507` 至少做 4096

意义：
消费级卡的核心痛点是“能不能稳定撑住长上下文”。

### 6.7 维度 G：双卡扩展性
主要指标：
- TP=2 是否稳定
- 启动时间变化
- TTFT 与吞吐相对单卡提升比
- 额外调度/通信开销

测试模型：
- `Qwen3.5-9B`

意义：
双卡消费级主机是很常见的“穷人工作站”形态，这部分结果很有参考价值。

---

## 7. 环境与执行策略

### 7.1 Python 隔离策略
使用 `uv` 分别为两个仓库创建独立环境，避免：
- `torch` 版本冲突
- `flashinfer` / `flash-attn` / `sglang-kernel` / `vllm` 构建依赖相互污染
- benchmark 结果受到共享环境残留影响

计划目录：
- `consumer_gpu_benchmark/envs/vllm/`
- `consumer_gpu_benchmark/envs/sglang/`

### 7.2 GPU 绑定策略
- 单卡测试优先使用 `CUDA_VISIBLE_DEVICES=1`
- 双卡测试使用 `CUDA_VISIBLE_DEVICES=0,1`
- 每轮测试前后采集 `nvidia-smi` 信息

理由：
GPU0 当前承担桌面图形负载，单卡 benchmark 若使用 GPU0，会把桌面噪声混入结果。

### 7.3 冷启动 / 热启动分离
每个关键场景做两类记录：
- cold run：首次启动或首次服务
- warm run：经过一次热身后的稳定状态

理由：
很多框架在消费级卡上的体验差异主要体现在：
- 首次编译
- CUDA graph capture
- Triton autotune
- JIT kernel warmup

---

## 8. 输出目录设计

计划在当前工作区创建：

```text
consumer_gpu_benchmark/
  PLAN.md
  scripts/
    setup_envs.sh
    run_vllm.sh
    run_sglang.sh
    bench_online.sh
    bench_offline.sh
    bench_micro.sh
    collect_gpu_stats.sh
  configs/
    models.json
    scenarios.json
  logs/
    vllm/
    sglang/
  results/
    raw/
    normalized/
  reports/
    final_report.md
```

这样设计的原因：
- `scripts/` 放执行逻辑，便于复跑。
- `configs/` 放模型与场景矩阵，避免硬编码散落到脚本里。
- `logs/` 保存服务输出，方便追错。
- `results/raw/` 保留原始 JSON/JSONL。
- `results/normalized/` 放统一格式的汇总数据，便于最终报告。
- `reports/` 放面向人读的分析结论。

---

## 9. 第一阶段要跑的正式矩阵

### Phase 0：环境烟雾测试
模型：
- Qwen3.5-0.8B

内容：
- vLLM 环境创建、安装、启动服务
- SGLang 环境创建、安装、启动服务
- `/health` 与 `/v1/models` 可用性
- 单次随机请求成功返回

### Phase 1：单卡主对比（核心）
模型：
- Qwen3.5-0.8B
- Qwen3-4B-Instruct-2507

内容：
- 在线服务 benchmark
- 离线吞吐 benchmark
- 单批次微基准
- 长上下文压力测试
- shared-prefix 测试

### Phase 2：双卡扩展性
模型：
- Qwen3.5-9B

内容：
- TP=2 启动可用性
- 在线服务 benchmark
- 离线吞吐 benchmark
- 与单卡结果做扩展比分析

---

## 10. 结果分析口径

最终报告会按以下结构解释：

1. **易用性**：哪个更容易装、哪个更容易第一次跑通。
2. **单卡低延迟**：谁的 TTFT/ITL 更好。
3. **单卡吞吐**：谁在 0.8B / 4B 上更强。
4. **长上下文**：谁更稳，谁更容易 OOM。
5. **shared-prefix / cache 场景**：谁对模板化业务更友好。
6. **双卡扩展**：谁的 TP=2 更值得在消费级卡上使用。
7. **结论建议**：
   - 如果你更在意“开箱即用”
   - 如果你更在意“单卡低延迟”
   - 如果你更在意“多请求吞吐”
   - 如果你更在意“长上下文与 cache”

---

## 11. 风险与预期

### 主要风险
- vLLM 与 SGLang 的依赖栈都较重，首次构建可能耗时较长。
- 3080 Ti 12GB 对 4B 以上模型的上下文长度较敏感。
- GPU0 桌面占用会影响双卡测试的绝对公平性。
- 某些模型可能需要显式 `--trust-remote-code` 或 chat template 参数。

### 预期策略
- 先确保 `0.8B -> 4B -> 9B` 逐级稳定推进；若有余力，再补充 `35B-A3B` 可行性附录。
- 一旦某个模型在任一框架频繁 OOM，不继续扩大问题规模，而是记录为“消费级卡限制下的可用性差异”。
- 保留所有失败日志，失败本身也是结论的一部分。

---

## 12. 下一步实施项

下一步将落地：

1. 创建 `consumer_gpu_benchmark/` 下的目录结构。
2. 编写环境创建与 benchmark 执行脚本。
3. 先跑 `Qwen3.5-0.8B` 的烟雾测试。
4. 再进入 0.8B / 4B 主 benchmark，并补充 9B 双卡扩展对比。
