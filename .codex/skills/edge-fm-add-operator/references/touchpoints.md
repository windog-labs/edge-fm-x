# 文件触点

按改动类型选择最小修改面，不要一开始就同时改 layer、operator、backend、模型组装和所有测试。

## 场景 A：给现有 op kind 增加一个新 impl

适用条件：layer contract 不变，只是新增一个 `impl_id`、新的 kernel 或新的 backend 选择。

优先检查这些文件：

- `src/operators/<existing>_op.h`
- `src/operators/<existing>_op.cu`
- `src/operators/linear_impl.cu`，如果是 linear
- `examples/config/operator_impl_table.json`
- 对应 layer test：
  - `tests/layers/test_activation.py`
  - `tests/layers/test_layernorm.py`
  - `tests/layers/test_attn.py`
  - `tests/layers/test_linear.py`

常见动作：

1. 在 registry 构造函数里注册新实现。
2. 给实现补 `impl_id()` 和 `supports(ctx)`。
3. 让 `default_impl()` 在没有表项时仍然稳定。
4. 只在需要强制路由时添加 `operator_impl_table` 记录。

## 场景 B：新增一个 op kind + layer

适用条件：需要新的 layer API、新的 operator context 或新的模型拼装边界。

通常要新建或修改：

- `src/operators/<name>_op.h`
- `src/operators/<name>_op.cu`
- `src/layers/<name>.h`
- `src/layers/<name>.cu`
- `src/operators/CMakeLists.txt`
- `src/layers/CMakeLists.txt`
- `src/python/pybind_debug.cpp`
- `examples/config/operator_impl_table.json`
- `tests/layers/test_<name>.py`

按需要再看：

- `src/models/...`
如果模型图里要真正接入这个 layer。

- `src/backends/horizon_module_emitter.cpp`
- `src/engine/horizon_engine.cpp`
- `tests/engine/test_from_model_api.py`
如果 whole-graph backend 也要感知该算子。

## 场景 C：linear 专属新增实现

`linear` 和其他 op kind 不同，当前实现集中在 `src/operators/linear_impl.cu`，并且会读写 `impl_params`、`shape_sig`、descriptor cache。

优先检查：

- `src/operators/linear_impl.cu`
- `src/layers/linear.h`
- `src/layers/linear.cu`
- `examples/config/operator_impl_table.json`
- `src/engine/horizon_engine.cpp`

典型额外动作：

1. 如有新的调参字段，在 `selected_impl_params_` 消费它。
2. 如有 per-shape 路由，确认 `shape_sig` 足够稳定。
3. 如有新的 artifact 元数据，决定是否通过 `impl_params` 传入。
4. 如影响 whole-graph 后端，确认 `linear_operator_table` 能携带所需字段。

## 场景 D：修改 operator_impl_table 选择规则

只有在现有字段不够表达时才改 `operator_impl_table` 本身。

需要检查：

- `src/operators/operator_impl_table.h`
- `src/operators/operator_impl_table.cpp`
- `examples/config/operator_impl_table.json`
- 依赖该选择行为的 layer tests
- `tests/engine/test_from_model_api.py`，如果 compile spec 序列化受到影响

优先避免：

- 为单个实验性 kernel 扩 schema
- 在 query 端和 table 端同时发散命名
- 引入无法稳定归一化的动态字段

## 场景 E：只做 layer 级 Python 对齐测试

如果只是为了给新增 layer 写 Python 测试，常见最小路径是：

- 在 `src/python/pybind_debug.cpp` 暴露构造函数和关键 `forward*`
- 仿照现有 `tests/layers/test_*.py`，用临时 `engine_config.json` 和最小模型权重驱动测试
- 通过 DLPack 在 PyTorch 和 `edge_fm.Tensor` 之间互转

## 改动前检查清单

- 先确认这是“新增 impl”还是“新增 op kind”
- 先确认是否真的需要更新 `operator_impl_table`
- 先确认是否真的需要更新 Horizon compile spec
- 先确认测试是否可以复用现有 layer test 风格
