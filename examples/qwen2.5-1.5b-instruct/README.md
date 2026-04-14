# Qwen2.5-1.5B-Instruct 示例

本目录提供与 `qwen2.5-0.5b-instruct` 同结构的 **Qwen2.5-1.5B-Instruct** 示例，用于 EdgeFM 推理。

## 目录结构

```
qwen2.5-1.5b-instruct/
├── README.md           # 本说明
├── download.sh         # 从 HuggingFace 下载模型
├── engine_config.json  # 引擎配置（prefill_model_path 指向子目录）
├── generate.py         # Python 推理示例
└── qwen2.5-1.5b-instruct/   # 模型文件（由 download.sh 生成）
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

## 1. 下载模型

```bash
cd examples/qwen2.5-1.5b-instruct
bash download.sh
```

模型会下载到当前目录下的 `qwen2.5-1.5b-instruct/`（约 3GB）。

## 2. 构建 EdgeFM

在项目根目录执行：

```bash
mkdir -p build && cd build
cmake .. -DPLATFORM=a800 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
make -j && make install
```

## 3. 运行示例

在**本示例目录**下执行（保证 `engine_config.json` 中的相对路径 `qwen2.5-1.5b-instruct` 能正确解析）：

```bash
cd examples/qwen2.5-1.5b-instruct

# 按需设置运行时库路径（与 test_all.sh 一致）
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/path/to/edge-fm/build-a800/install/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

python3 generate.py
```

将 `/path/to/edge-fm` 替换为项目根目录实际路径。

## 模型与配置说明

- **Qwen2.5-1.5B-Instruct**：`num_attention_heads=12`，`num_key_value_heads=2`，`group_size=6`。  
  EdgeFM 已为 FlashInfer 增加对 `group_size=6` 的支持，可直接使用本模型。
- `engine_config.json` 中 `prefill_model_path` 使用相对路径 `qwen2.5-1.5b-instruct`，会相对于配置文件所在目录解析，因此建议在 `examples/qwen2.5-1.5b-instruct` 下运行 `generate.py`。
