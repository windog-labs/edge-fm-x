# operator_impl_table 规则

## JSON schema

当前示例文件是 `examples/config/operator_impl_table.json`，每条记录形如：

```json
{
  "model_name": "qwen2_5",
  "hw_profile": "cuda_sm80",
  "op_kind": "linear",
  "layer_role": "fused_gate_up",
  "op_name": "",
  "stage": "prefill",
  "shape_sig": "",
  "impl_id": "cutile",
  "impl_params": {
    "artifact_path": "artifacts/cutile/qwen2_5/fused_gate_up_prefill_sm100.json",
    "kernel_name": "fused_gate_up_prefill_bf16_sm100_v1",
    "launcher": "python_generated"
  }
}
```

字段含义：

- `model_name`
模型归一化名，如 `qwen2_5`、`qwen2_5_vl`

- `hw_profile`
硬件画像，如 `cuda`、`cuda_sm80`、`cuda_sm100`

- `op_kind`
算子大类，如 `linear`、`attention`、`norm`、`activation`

- `layer_role`
稳定的语义类别，适合跨 layer 复用

- `op_name`
更细粒度的名字。当前 linear 常用 `layer_prefix`，activation、norm、attention 常用固定 op name

- `stage`
一般为 `prefill` 或 `decode`

- `shape_sig`
形状签名。当前 linear 使用得最完整

- `impl_id`
registry 里实现的稳定 ID，必须能被 `find_impl_by_id()` 找到

- `impl_params`
附加参数。只有实现真的消费它时再写

## 匹配与打分优先级

解析逻辑在 `src/operators/operator_impl_table.cpp`。当前总分优先级从高到低是：

1. `op_name` 精确匹配：`1000000`
2. `layer_role` 精确匹配：`100000`
3. `shape_sig` 精确匹配：`10000`
4. `stage` 精确匹配：`1000`
5. `hw_profile`
   - 精确匹配：`200`
   - 前缀匹配：`120`
   - 例：表里写 `cuda`，查询是 `cuda_sm80`，仍会命中
6. `model_name` 精确匹配：`100`
7. `op_kind` 精确匹配：`10`

空字符串或 `"*"` 视为 wildcard，得分为 `0`，但允许命中。

## 同分覆盖规则

`resolve()` 里使用的是：

```cpp
if (total_score >= best_score) {
    best_match = record;
    best_score = total_score;
}
```

这意味着：

- 分数更高的记录会覆盖更泛化的记录
- 分数相同时，后出现的记录会覆盖先出现的记录
- 外部 JSON 记录在内建记录之后加载，因此即使同分，也会压过内建默认记录

实务建议：

1. 把更具体的记录写得更具体，而不是只依赖“排在后面”。
2. 如果必须做同分覆盖，把 override 记录放在 JSON 更靠后的位置。
3. 不要写两条只差顺序、不差语义的重复记录。

## 归一化规则

当前实现会归一化：

- `model_name`
  - `Qwen2.5`、`qwen25`、`qwen2` 会归一化到 `qwen2_5`
  - `Qwen2.5-VL`、`qwen25_vl` 会归一化到 `qwen2_5_vl`

- `hw_profile`
  - 非字母数字会转成下划线风格

- `stage`
  - `prefill`、`decode` 会归一化到固定 key

新增 query 字段或新命名时，优先复用这些已有归一化习惯。

## 什么时候用哪个字段

### 只想按 op kind 选默认实现

只填：

- `model_name`
- `hw_profile`
- `op_kind`
- `impl_id`

其他字段留空。

### 想按语义类别选

优先用 `layer_role`，适合：

- `fused_qkv`
- `mlp_down`
- `input_norm`

### 想只命中特定 layer

用 `op_name`，适合：

- linear 的特定 `layer_prefix`
- 某个固定名字的 operator API

### 想按 prefill 或 decode 分流

补 `stage`

### 想按形状选最优 kernel

补 `shape_sig`

但只在 shape 真是关键决策维度时使用，否则会把表维护成本拉高。

## impl_params 使用原则

当前 `linear` 已经会消费 `impl_params`。只有在实现明确读取这些字段时才写入，例如：

- `algo_index`
- `artifact_path`
- `kernel_name`
- `launcher`

不要把“方便注释”的信息塞进 `impl_params`；它应该只承载实现真正消费的参数。

## 当前示例中的重要事实

- `operator_impl_table.cpp` 内建了一组 `cuda` 级别的默认记录
- `examples/config/operator_impl_table.json` 额外给出 `cuda_sm80` 的更细粒度 linear role 记录
- 还给了一个 `cuda_sm100 + fused_gate_up + prefill -> cutile` 的更具体 override 示例

新增记录时，优先模仿这些现有模式，而不是重新发明字段组合。
