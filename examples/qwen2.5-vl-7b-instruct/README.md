# Qwen2.5-VL-7B-Instruct 示例

本目录提供与 `qwen2.5-vl-3b-instruct` 同结构的 **Qwen2.5-VL-7B-Instruct** 示例，用于 EdgeFM VLM 推理准备与本地 benchmark。

## 目录结构

```text
qwen2.5-vl-7b-instruct/
├── README.md               # 本说明
├── download.sh             # 从 HuggingFace 下载模型
├── engine_config.json      # 引擎配置（prefill_model_path 指向子目录）
├── generate.py             # Python 推理示例
└── qwen2.5-vl-7b-instruct/ # 模型文件目录
    ├── config.json
    ├── tokenizer.json
    ├── preprocessor_config.json
    └── ...
```

## 1. 放置模型

当前仓库期望的模型目录是：

```bash
examples/qwen2.5-vl-7b-instruct/qwen2.5-vl-7b-instruct/
```

如果你现在正在同步的目录是 `examples/Qwen2.5-VL-7B-Instruct`，同步完成后将其内容移动或同步到上面的目标目录即可。

例如：

```bash
rsync -a examples/Qwen2.5-VL-7B-Instruct/ examples/qwen2.5-vl-7b-instruct/qwen2.5-vl-7b-instruct/
```

也可以直接在本目录执行下载脚本：

```bash
cd examples/qwen2.5-vl-7b-instruct
bash download.sh
```

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
cd examples/qwen2.5-vl-7b-instruct

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/path/to/edge-fm/build-a800/install/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

python3 generate.py
```

将 `/path/to/edge-fm` 替换为项目根目录实际路径。

## 模型与配置说明

- `engine_config.json` 中 `prefill_model_path` 使用相对路径 `qwen2.5-vl-7b-instruct`，会相对于配置文件所在目录解析。
- `generate.py` 提供的是纯文本 smoke test，便于快速验证模型路径和引擎配置。
- 真正的 VLM 图像输入、prepared multimodal、benchmark 对比请参考：
  - `tests/engine/test_qwen2_generate.py`
  - `doc/benchmark_reports/qwen_vlm_suite_20260407.md`
