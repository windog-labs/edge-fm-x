# Qwen2.5-3B-Instruct 示例

本目录提供与 `qwen2.5-1.5b-instruct` 同结构的 **Qwen2.5-3B-Instruct** 示例，用于 EdgeFM 推理。

## 目录结构

```text
qwen2.5-3b-instruct/
├── README.md           # 本说明
├── download.sh         # 从 HuggingFace 下载模型
├── engine_config.json  # 引擎配置（prefill_model_path 指向子目录）
├── generate.py         # Python 推理示例
└── qwen2.5-3b-instruct/   # 模型文件
    ├── config.json
    ├── model-00001-of-00002.safetensors
    ├── tokenizer.json
    └── ...
```

如果 `qwen2.5-3b-instruct/` 子目录已经有模型文件，则无需再次下载。

## 1. 下载模型

```bash
cd examples/qwen2.5-3b-instruct
bash download.sh
```

模型会下载到当前目录下的 `qwen2.5-3b-instruct/`。

## 2. 构建 EdgeFM

在项目根目录执行：

```bash
mkdir -p build && cd build
cmake .. -DPLATFORM=a800 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
make -j && make install
```

## 3. 运行示例

在**本示例目录**下执行：

```bash
cd examples/qwen2.5-3b-instruct

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/path/to/edge-fm/build-a800/install/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

python3 generate.py
```

将 `/path/to/edge-fm` 替换为项目根目录实际路径。

## 模型与配置说明

- **Qwen2.5-3B-Instruct**：`num_attention_heads=16`，`num_key_value_heads=2`，`group_size=8`。
- `engine_config.json` 中 `prefill_model_path` 使用相对路径 `qwen2.5-3b-instruct`，会相对于配置文件所在目录解析，因此建议在 `examples/qwen2.5-3b-instruct` 下运行 `generate.py`。
- `speculative.enabled=false`，因此即使没有 draft model 也不会影响默认运行。
