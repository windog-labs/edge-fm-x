# EdgeFM Design

## 1. Design Goals

EdgeFM is now intentionally biased toward a simpler model-specific runtime:

1. `engine.json` explicitly declares `model_name`. We do not infer model family from model files.
2. We do not introduce a generic IR for the main execution path.
3. We do not do runtime operator tuning in CUDA anymore.
4. Operator selection is driven by a per-hardware operator implementation table.
5. The codebase is organized around five layers:
   - `engine/`: execution entry and request lifecycle
   - `models/`: model-specific runtime
   - `layers/`: model layer semantics
   - `operators/`: concrete operator implementations, lightweight dispatch, and low-level kernels
   - `backends/`: backend-specific compilation / artifact helpers

Current supported model family:

- `qwen2_5`
- `qwen2_5_vl`

## 2. High-Level Architecture

```mermaid
graph TD
    A[EdgeFM(config_path)] --> B[EngineConfig]
    B --> C{backend_target}

    C -->|cuda| D[StandardEngine]
    C -->|horizon| E[HorizonEngine]

    D --> F[Model::create(model_name)]
    F --> G[Qwen2_5 Runtime]
    G --> H[layers/]
    H --> I[operators/\nLinear impl table + direct attention/norm/activation ops + kernels/]

    E --> J[backends/\nHorizon emitter + artifact cache]
    J --> K[compile spec v2 + generated module]
```

核心方向是：

- `models/` 负责模型拓扑和张量流。
- `layers/` 负责层级语义，比如 `LinearLayer`、`AttentionLayer`。
- `operators/` 负责“这层到底选哪个实现”，以及承接具体算子入口与底层 kernel。
- `backends/` 负责 Horizon 这种后端专属的 compile spec、artifact metadata 等逻辑。

## 3. Configuration Model

### 3.1 Required fields

主入口统一为：

```python
engine = edge_fm.EdgeFM("/path/to/engine.json")
```

`engine.json` 至少需要：

- `model_name`
- `prefill_model_path`
- `runtime.device`

### 3.2 Key runtime fields

```json
{
  "model_name": "Qwen2.5",
  "runtime": {
    "device": "cuda",
    "device_id": 0,
    "hw_profile": "cuda"
  },
  "operator_impl_table_path": "../config/operator_impl_table.json"
}
```

说明：

- `model_name`
  - 当前必须显式指定。
  - 支持轻量归一化，例如 `Qwen2.5 -> qwen2_5`。
- `runtime.hw_profile`
  - 表示当前硬件档位 / 目标 profile。
  - 若未显式提供，CUDA 会退化为 `cuda_smXX` 或 `cuda`，Horizon 会退化为 `horizon`。
- `operator_impl_table_path`
  - 可选。
  - 若提供，会在 builtin defaults 之上叠加外部表。
  - 相对路径按 `engine.json` 所在目录解析。

## 4. Model Dispatch

`Model::create` 只根据 `engine.json` 中的 `model_name` 做分派：

- `qwen2_5` -> `Qwen2_5`
- `qwen2_5_vl` -> `Qwen2_5`

这里明确不再做：

- 从 HF `config.json` 推断模型族
- 生成通用 `model_description`
- 构造通用 `execution_plan`

模型文件里的 `config.json` 仍然会被读取，但用途只剩下：

- 获取静态维度信息
- 获取模型 dtype
- 对 VL 模型读取 `text_config`
- 获取 M-RoPE 等模型内部参数

## 5. Text / VL Unified Runtime

`Qwen2_5` 是当前共享 runtime：

- 文本模型走标准 token path。
- VL 模型仍通过请求侧注入多模态信息。

VL 请求契约保持不变：

- `Request(..., embedding=...)`
- `Request(..., embed_token_id=...)`
- `Request(..., position_ids=...)`

因此：

- `qwen2_5` 和 `qwen2_5_vl` 共用一套 CUDA runtime
- 区别主要体现在请求输入和模型配置约束上

## 6. Layers vs Operators

这次重构的核心边界是把“层语义”和“具体实现”拆开。

### 6.1 `layers/`

`layers/` 表示模型图里的语义层，例如：

- `EmbedHeadLayer`
- `RMSNormLayer`
- `AttentionLayer`
- `LinearLayer`
- `FusedQKVLinearLayer`
- `FusedGateUpLinearLayer`
- `LMHeadLinearLayer`

这些类负责：

- 输入 / 输出 tensor 约定
- 权重组织方式
- 融合权重布局
- 层级 forward 语义

### 6.2 `operators/`

`operators/` 表示具体算子实现与选择策略，例如：

- `OperatorImplTable`
- `linear_impl.cu`
- `attention_op.cu`
- `norm_op.cu`
- `activation_op.cu`
- `operators/kernels/int4_groupwise_gemm/*`

这些模块负责：

- 按 `model_name + hw_profile + op_kind + layer_role + op_name + stage + shape_sig` 选择实现
- 管理具体后端实现入口
- 让后续新增 CUTLASS 或 agent kernel 时尽量不碰模型层代码

当前边界进一步明确为：

- `LinearLayer`
  - 保留 layer 语义和融合权重组织
  - 通过 `OperatorImplTable` 选择具体实现
- `AttentionLayer`
  - 保留 attention 语义、张量约束、KV cache 交互
  - `apply_mrope(...)` 仍保留在 layer
  - prefill / decode 计算直接调用 `operators/attention_op.*`
- `RMSNormLayer`
  - 保留权重选择和 fused/plain 语义
  - 具体 norm kernel 直接调用 `operators/norm_op.*`
- `ActivationLayer`
  - 保留 `hidden_act` 校验和张量约束
  - 具体 `silu_and_mul` 直接调用 `operators/activation_op.*`

### 6.3 Why `operators/` with `kernels/`

当前更适合叫 `operators/`，并在其下保留 `kernels/`，因为这里承载的不只是自写 CUDA kernel，还包括：

- cuBLASLt
- CUTLASS
- cutile-python generated kernels
- agent-generated kernel

它们都属于“算子实现”，但不一定都是 repo 内部手写 kernel。现在把真正底层的 CUDA 资产放进 `operators/kernels/`，其余算子入口继续保留在 `operators/` 根下，结构更直接，也避免在 `layers/` 下再叠一层实现目录。

## 7. Operator Implementation Table

### 7.1 Intent

之前的思路是 runtime tuning。现在改成：

- 用户或系统维护一张“按硬件选择最优实现”的表
- Engine 读取这张表
- Model runtime 按表把 layer 对应到 operator impl

换句话说，现在是“查表搭积木”，而不是“运行时 benchmark 再决定”。

### 7.2 Schema

```json
{
  "schema": "edgefm_operator_impl_table_v1",
  "records": [
    {
      "model_name": "qwen2_5",
      "hw_profile": "cuda",
      "op_kind": "linear",
      "layer_role": "",
      "op_name": "",
      "stage": "",
      "shape_sig": "",
      "impl_id": "cublasLt",
      "impl_params": {}
    }
  ]
}
```

字段含义：

- `model_name`: 模型族
- `hw_profile`: 硬件 profile，例如 `cuda`、`cuda_sm80`
- `op_kind`: 当前主要是 `linear`
- `layer_role`: 语义角色，例如 `fused_qkv`、`fused_gate_up`、`lm_head`
- `op_name`: 更具体的层名
- `stage`: `prefill` / `decode`
- `shape_sig`: 更细粒度形状签名
- `impl_id`: 选中的实现，例如 `cublasLt`、`cutlass`、`cutile`
- `impl_params`: 实现参数，例如：
  - `algo_index` for `cublasLt`
  - `artifact_path` / `kernel_name` / `launcher` for generated `cutile` kernels

### 7.3 Matching strategy

当前实现采用“越具体越优先”的匹配：

- `op_name` 精确匹配优先于 wildcard
- `layer_role` 精确匹配优先于 wildcard
- `shape_sig` 精确匹配优先于 wildcard
- `stage` 精确匹配优先于 wildcard
- `hw_profile` 精确匹配优先于泛化 profile
  - 例如 `cuda_sm80` 优先于 `cuda`

如果外部表没有命中：

- 先回退到 builtin defaults
- 再回退到第一个 `supports(...)` 的实现

## 8. Linear Multi-Implementation Path

`Linear` 是当前唯一正式接入 operator table 的算子族。

涉及的 layer：

- `LinearLayer`
- `FusedQKVLinearLayer`
- `FusedGateUpLinearLayer`
- `LMHeadLinearLayer`

执行流程：

1. `LinearLayer` 根据当前输入构造 `LinearOpContext`
2. 提供 `layer_prefix + layer_role + stage + shape_sig`
3. `operators/OperatorImplTable` 解析最优实现
4. `LinearImpl` 执行实际 forward

当前实现状态：

- `cublasLt`: 完整可用
- `cutlass`: 预留接入点
- `cutile`: 预留生成型 kernel 接入点
- `agent`: 预留接入点

对于 `cublasLt`：

- 仍然保留 descriptor cache 和 heuristic 逻辑
- 但不再做 runtime benchmark tuning
- 如果 operator table 给了 `impl_params.algo_index`，则直接使用
- 否则回退到 heuristic 首选算法

对于 `cutile`：

- 当前把它视为 Python DSL 生成的 kernel 来源，而不是直接链接的 vendor library
- `operator_impl_table` 可以记录对应 artifact metadata
- 当前 runtime 只保留 `impl_id` 和 `impl_params` 的接入点，还没有实现 launcher / artifact loader

## 9. `tune()` Semantics

### 9.1 CUDA

`StandardEngine::tune()` 现在不再做 benchmark。

它的职责退化为：

- 校验 `model_name`
- 校验 `hw_profile`
- 校验并加载 `operator_impl_table`

也就是说，`tune()` 还保留 API，但语义已经从“现场调优”变成“静态准备 / 校验”。

### 9.2 Horizon

`HorizonEngine::tune()` 仍然保留，因为它代表：

- 生成 compile spec
- 生成 Python module
- 写入 artifact metadata

这部分属于 backend-specific compile flow，不属于 CUDA runtime tuning。

## 10. Horizon Backend Path

`HorizonEngine::tune()` 直接基于：

- `model_name`
- `model_config`
- `graph_tuning`
- `operator_impl_table`
- `prefill / decode model path`

生成 compile spec v2。

### 10.1 Compile spec v2 core fields

- `schema`
- `backend`
- `model_name`
- `model_variant`
- `model_config`
- `graph_tuning`
- `generated_module`
- `artifact`

### 10.2 `graph_tuning`

当前最小字段：

- `attention_type`
- `kv_cache.dtype`
- `kv_cache.layout`
- `uses_mrope`
- `uses_embedding_injection`
- `linear_operator_table`
- `target_hw_constraints`

这里明确不再出现：

- `model_description`
- `execution_plan`
- `linear_impl_overrides` from runtime tuning cache

## 11. Source Tree Boundaries

### 11.1 `src/engine/`

负责：

- `EngineConfig`
- `StandardEngine`
- `HorizonEngine`
- `KVManager`
- `Scheduler`

### 11.2 `src/models/`

负责：

- 模型专用 runtime
- 当前主实现：`qwen2_5/`

### 11.3 `src/layers/`

负责：

- 模型层语义
- 融合层权重组织
- 与模型 forward 直接相关的 tensor contract

### 11.4 `src/operators/`

负责：

- operator impl table
- 具体 operator backend 实现
- layer 到 operator 的选择逻辑

### 11.5 `src/backends/`

负责：

- `BackendArtifactCache`
- `HorizonModuleEmitter`
- 其他后端专属编译 / 产物逻辑

### 11.6 `src/utils/`

负责：

- 日志
- 设备内存工具
- 权重装载
- CUDA graph
- 其他真正通用的小型工具

## 12. Current Boundaries

- 当前只正式支持 `Qwen2.5` 家族
- 当前只有 `Linear` 接入了多实现选择和 operator impl table
- `Attention / RMSNorm / Activation` 已经完成 `layers -> operators` 边界收口，但当前只有默认单实现
- 当前没有通用 IR
- 当前没有 CUDA runtime tuning benchmark
- 当前 Horizon 仍然只生成 compile spec / artifact metadata，不执行实际推理

## 13. Reading Guide

- `src/engine/engine.*`
  - config parsing, backend dispatch inputs
- `src/models/model.*`
  - model dispatch
- `src/models/qwen2_5/`
  - current model runtime
- `src/layers/linear.*`
  - linear layer semantics and fused weight organization
- `src/layers/attention.*`
  - attention semantics plus in-layer M-RoPE helper
- `src/layers/layernorm.*`
  - RMSNorm semantics and weight routing
- `src/layers/activation.*`
  - activation semantics and tensor contract
- `src/operators/operator_impl_table.*`
  - implementation table schema and matching
- `src/operators/linear_impl.cu`
  - concrete linear operator implementations
- `src/operators/attention_op.cu`
  - direct FlashInfer attention operator entry
- `src/operators/norm_op.cu`
  - direct FlashInfer RMSNorm operator entry
- `src/operators/activation_op.cu`
  - direct FlashInfer activation operator entry
- `src/operators/kernels/int4_groupwise_gemm/`
  - low-level CUDA kernels used by operator-side linear paths
- `src/backends/horizon_module_emitter.*`
  - Horizon compile-spec / module generation
- `src/backends/backend_artifact_cache.*`
  - backend artifact persistence
