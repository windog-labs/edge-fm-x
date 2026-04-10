# Qwen2.5-VL-0.5B 示例

本目录为 `examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b/` 下的模型文件补齐了 EdgeFM 示例入口。

## 目录结构

```text
qwen2.5-vl-0.5b/
├── README.md              # 本说明
├── engine_config.json     # EdgeFM 引擎配置
├── generate.py            # 文本侧 smoke test
└── qwen2.5-vl-0.5b/       # 模型文件目录
    ├── config.json
    ├── tokenizer.json
    ├── preprocessor_config.json
    └── ...
```

## 使用方式

先在项目根目录构建并安装 Python 绑定：

```bash
mkdir -p build && cd build
cmake .. -DPLATFORM=a100 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
make -j && make install
```

然后在本目录执行：

```bash
cd examples/qwen2.5-vl-0.5b
python3 generate.py
```

如果 `edge_fm` 无法导入，请先确认你运行脚本的 Python 版本与 `build/install/python/edge_fm*.so` 的 ABI 一致。例如当前环境若默认是 Python 3.12，而扩展是 `cpython-310`，则需要切到对应 Python 版本重新执行，或重新编译 pybind。

## 配置说明

- `engine_config.json` 复用了现有 VLM 示例的配置形式，`model_name` 为 `Qwen2.5-VL`，算子表使用 `../config/operator_impl_table_vlm.json`。
