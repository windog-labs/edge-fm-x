# 验证流程

新增算子后，优先做最小闭环验证：重新编译安装，跑最接近改动面的 tests，再决定是否扩大回归范围。

## 重新编译

如果当前 `build/` 已配置好，通常直接执行：

```bash
cd build
make -j
make install
```

如果新增了源文件或改了 CMake 结构，先重新配置：

```bash
mkdir -p build
cd build
cmake .. -DPLATFORM=a100
make -j
make install
```

如果当前环境使用别的 `PLATFORM`，沿用已有配置，不要随意切平台。

## layer 级测试入口

当前仓库现有对齐测试大多直接实例化 layer，并使用 `build/install/python` 下的 `edge_fm` Python 模块。

常用命令：

```bash
pytest -s tests/layers/test_activation.py
pytest -s tests/layers/test_layernorm.py
pytest -s tests/layers/test_attn.py
pytest -s tests/layers/test_linear.py
```

如果新增全新的 layer，优先仿照这些文件写一个 `tests/layers/test_<name>.py`。

## operator_impl_table 回归点

当前多个 layer test 会显式把 `operator_impl_table_path` 指向：

```text
examples/config/operator_impl_table.json
```

所以改这个文件后，至少要确认：

1. 指向新 `impl_id` 的路径能命中
2. 没有对应记录时仍能走 `default_impl()`
3. 没有把现有层的默认行为误改掉

## Horizon compile spec 回归点

如果改动影响 whole-graph backend，补跑：

```bash
pytest -s tests/engine/test_from_model_api.py
```

这个测试会校验：

- compile spec 已生成
- schema 仍是 `edgefm_horizon_compile_spec_v2`
- `graph_tuning` 仍存在
- 当前 `linear_operator_table` 序列化仍可用

## 新增实现时的最小验证建议

### 现有 op kind 新 impl

至少覆盖：

1. 一条命中 `operator_impl_table` 的 case
2. 一条不命中表、退回 `default_impl()` 的 case
3. 一条 correctness case

如果有参考实现，再补一条 benchmark case。

### 全新的 op kind + layer

至少覆盖：

1. Python 侧可实例化
2. 权重可加载，如果有权重
3. `forward*` correctness
4. `operator_impl_table` 选择行为
5. 必要时的 compile-spec 或 backend 兼容

## 写测试时沿用的现有风格

- 用临时目录生成最小 `config.json` 或 `engine_config.json`
- 用 DLPack 在 PyTorch tensor 和 `edge_fm.Tensor` 之间互转
- 有 reference backend 时：
  - activation、norm、attention 对齐 FlashInfer
  - linear 对齐 PyTorch
- benchmark 用现有 `bench_gpu_time` 风格，避免自创计时框架
