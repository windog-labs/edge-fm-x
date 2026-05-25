# Qwen3.5-2B 示例

本目录用于放置和验证 `Qwen/Qwen3.5-2B` checkpoint。

## 目录结构

```text
qwen3.5-2b/
├── README.md
├── download.sh
├── engine_config.json
├── generate.py
└── qwen3.5-2b/       # 模型文件目录，由 download.sh 生成
    ├── config.json
    ├── model.safetensors-00001-of-00001.safetensors
    ├── tokenizer.json
    └── ...
```

## 1. 下载模型

```bash
cd examples/qwen3.5-2b
bash download.sh
```

模型会下载到当前目录下的 `qwen3.5-2b/`。

## 2. Transformers smoke test

```bash
cd examples/qwen3.5-2b
python3 generate.py
```

## EdgeFM 支持状态

- `engine_config.json` 预留了 EdgeFM 配置入口，`prefill_model_path` 指向本地模型目录。
- 当前 EdgeFM Qwen loader 支持的是 Qwen2.5/Qwen2.5-VL 的 `self_attn` 权重布局；Qwen3.5 checkpoint 使用 `model.language_model.layers.*.linear_attn.*` 结构，暂不能直接用 EdgeFM `generate()` 原生运行。
- 如需接入 EdgeFM，需要后续新增 Qwen3.5 linear-attention 模型实现与对应算子路径。
