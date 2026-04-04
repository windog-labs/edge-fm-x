---
name: edge-fm-add-operator
description: Add or extend operators in edge-fm using the current layer + operators + operator_impl_table design. Use when Codex needs to add a new operator implementation (`impl_id`) to an existing op kind, introduce a new `op_kind` and layer pair, update the 最优算子表 / `operator_impl_table`, wire operator selection by `model_name`, `hw_profile`, `layer_role`, `op_name`, `stage`, or `shape_sig`, update CMake or pybind touchpoints, or keep Horizon `graph_tuning` aligned with the operator changes.
---

# edge-fm 新增算子

按当前仓库分层工作：`layers/` 负责输入输出契约、校验和 `OperatorQuery` 构造，`operators/` 负责接口、实现和 registry，`operator_impl_table` 负责按模型、硬件、阶段和形状选择最优实现。新增算子时，先判断是“给现有 op kind 增加一个 impl”，还是“新增一个 op kind + layer 边界”。

## 先做判断

1. 如果输入输出 contract 不变，只是增加一个新的 kernel、backend 或调参实现，扩展现有 `operators/*_op.*` 或 `linear_impl.cu`，不要新建 layer。
2. 如果需要新的 `OpContext`、新的 layer public API、新的模型拼装边界，或新的 backend lowering 单元，新建 `src/operators/<name>_op.*` 与 `src/layers/<name>.*`。
3. 如果改动只影响 CUDA 运行时实现，不要默认同步改 Horizon whole-graph 路径；只有 compile spec 或 lowering 真要感知该算子时才扩展 `graph_tuning`。

## 工作流

1. 先读 `references/current-design.md`，确认现有命名约定、query 字段和缓存方式。
2. 再读 `references/touchpoints.md`，按“新增 impl”或“新增 op kind”选择需要改的文件集合。
3. 再读 `references/operator-impl-table.md`，决定是否需要新增 `operator_impl_table` 记录，以及是否需要 `shape_sig` 或 `impl_params`。
4. 先写 operators 侧 contract 与 registry，再写 layers 侧校验、context 和 `OperatorQuery`。
5. 最后补 CMake、pybind、`examples/config/operator_impl_table.json`、tests，以及必要的 Horizon compile-spec 触点。

## 实施要求

- 复用当前约定：layer 做 tensor、device、shape、dtype 校验，并在 `forward*` 中构造 context；operator 做 `impl_id()`、`supports()`、执行函数和默认选择。
- 让 `default_impl()` 在没有表项时也能工作；把 `operator_impl_table` 当成覆盖和定向选择机制，而不是唯一入口。
- 只在字段有稳定区分度时填写 `layer_role`、`op_name`、`stage`、`shape_sig`。不要为了“更精确”随手加字段，否则表会快速碎片化。
- 对现有 op kind 追加 impl 时，优先保持现有 layer API 不变，并让老测试仍能直接复用。
- 对全新的 op kind，补最小可运行路径：header/source、CMake、必要的 pybind_debug 入口、示例表项和 layer test。
- 只有 whole-graph backend 真的要识别该算子时，才同步更新 `src/engine/horizon_engine.cpp` 和 `src/backends/horizon_module_emitter.cpp`。
- 新增 tests 时，优先沿用现有 layer 测试风格；若已有参考实现（FlashInfer 或 PyTorch），复用相同的 correctness 和 benchmark 对齐方式。

## 验证

1. 先重新编译安装，再跑与改动最接近的 layer tests；命令见 `references/validation-workflow.md`。
2. 若改动涉及 `operator_impl_table` 选择逻辑，至少覆盖一条“命中表项”和一条“走 `default_impl()`”路径。
3. 若改动影响 Horizon compile spec 或 graph tuning，补跑 `tests/engine/test_from_model_api.py`。
4. 若引入了新的 query 字段或 registry 逻辑，确认未破坏已有内建默认记录和 `examples/config/operator_impl_table.json`。

## 何时读哪份 reference

- 看总体结构与命名：`references/current-design.md`
- 看具体要改哪些文件：`references/touchpoints.md`
- 看最优算子表字段、优先级和示例：`references/operator-impl-table.md`
- 看编译、测试和回归命令：`references/validation-workflow.md`
