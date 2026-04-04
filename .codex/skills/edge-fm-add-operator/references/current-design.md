# 当前设计

## 核心链路

`edge-fm` 当前算子设计是三层解耦：

1. `src/layers/*.h` / `src/layers/*.cu`
负责 layer public API、weight 绑定、tensor 校验、`OpContext` 构造、`OperatorQuery` 构造，以及按 stage 缓存已选实现。

2. `src/operators/*`
负责 operator interface、具体实现、registry，以及默认实现选择。常见模式是：
- `impl_id()` 返回稳定字符串
- `supports(ctx)` 判断当前上下文是否支持
- `Registry::default_impl(ctx)` 在没有表项时选一个可用实现

3. `src/operators/operator_impl_table.*`
负责加载内建记录和外部 JSON 表，并根据 `model_name`、`hw_profile`、`op_kind`、`layer_role`、`op_name`、`stage`、`shape_sig` 选择最优实现。

## 现有 op kind 与命名模式

### `activation`

- layer 文件：`src/layers/activation.h`、`src/layers/activation.cu`
- operator 文件：`src/operators/activation_op.h`、`src/operators/activation_op.cu`
- 当前 query：
  - `op_kind = "activation"`
  - `layer_role = "mlp_activation"`
  - `op_name = "silu_and_mul"`
  - `stage = "prefill"` 或 `"decode"`
- 当前实现：`flashinfer_silu_and_mul`

### `norm`

- layer 文件：`src/layers/layernorm.h`、`src/layers/layernorm.cu`
- operator 文件：`src/operators/norm_op.h`、`src/operators/norm_op.cu`
- `layer_role` 由 weight 语义推断：
  - `input_norm`
  - `post_attention_norm`
  - `final_norm`
- 当前 query：
  - `op_kind = "norm"`
  - `op_name = "rms_norm"`
  - `stage = "prefill"` 或 `"decode"`
- 当前实现：`flashinfer_norm`

### `attention`

- layer 文件：`src/layers/attention.h`、`src/layers/attention.cu`
- operator 文件：`src/operators/attention_op.h`、`src/operators/attention_op.cu`
- 当前 query：
  - `op_kind = "attention"`
  - `op_name = "attention"`
  - `stage = "prefill"` 或 `"decode"`
- 当前实现：`flashinfer_attention`
- 当前 layer 没有显式填 `layer_role` 或 `shape_sig`

### `linear`

- layer 文件：`src/layers/linear.h`、`src/layers/linear.cu`
- operator 实现集中在：`src/operators/linear_impl.cu`
- registry 头文件：`src/operators/linear_registry.h`
- `layer_role` 由 `layer_prefix` 推断：
  - `fused_qkv`
  - `attention_output`
  - `fused_gate_up`
  - `mlp_down`
  - `lm_head`
  - `linear`
- 当前 query：
  - `op_kind = "linear"`
  - `layer_role = infer_layer_role(layer_prefix)`
  - `op_name = layer_prefix`
  - `stage = "prefill"` 或 `"decode"`
  - `shape_sig = ctx.shape.to_string()`
- 当前实现示例：
  - `cublasLt`
  - `cutlass`
  - `cutile`
  - `agent`

## shape signature 约定

当前只有 linear 显式使用 `shape_sig`。格式定义在 `LinearLayer::LinearShapeSignature::to_string()`：

```text
m=<m>|input=<input_dtype>|weight=<weight_dtype>|output=<output_dtype>|in_features=<in>|out_features=<out>
```

如果新增 op kind 也需要 per-shape 选择，优先定义稳定、可序列化、不会轻易破坏兼容性的字符串格式。

## registry 与缓存模式

- `ActivationLayer`、`RMSNormLayer`、`AttentionLayer` 都按 stage 缓存 `selected_impl_id_` 和已解析的实现指针。
- `LinearLayer` 除了缓存 `selected_impl_id_`，还按 prefill 的 `m` 维护 descriptor cache，并缓存 `selected_impl_params_`、`best_algo_index_` 等。
- 新增算子时，只有当缓存能稳定复用且不会把 shape 或 stage 搞混时才引入缓存。

## 内建默认记录

`src/operators/operator_impl_table.cpp` 里有内建 fallback 记录：

- `qwen2_5` / `qwen2_5_vl`
- `hw_profile = "cuda"`
- 默认映射：
  - `linear -> cublasLt`
  - `attention -> flashinfer_attention`
  - `norm -> flashinfer_norm`
  - `activation -> flashinfer_silu_and_mul`

外部 `operator_impl_table.json` 记录会在这些内建记录之后追加加载。

## 与 whole-graph backend 的关系

- `src/engine/horizon_engine.cpp` 当前只把 `linear` 记录导出到 `graph_tuning.linear_operator_table`
- `src/backends/horizon_module_emitter.cpp` 说明 compile spec 会带上已解析的 operator table 信息
- 这意味着：
  - 新增 `linear` 相关实现时，要检查 Horizon compile spec 是否需要同步感知
  - 新增非 linear op kind 时，不要默认扩展 `graph_tuning`，除非 whole-graph backend 真要消费它

## Python 暴露面

- 当前 layer 级测试主要走 `src/python/pybind_debug.cpp`
- 如果新增 layer 需要 Python 侧直接实例化做对齐测试，通常要在 `pybind_debug.cpp` 加绑定
- `src/python/pybind.cpp` 主要是主运行时接口，不是当前 layer 级测试主入口
