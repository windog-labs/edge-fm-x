# Qwen2.5-VL-7B-Instruct Example

This directory provides a **Qwen2.5-VL-7B-Instruct** example with the same structure as `qwen2.5-vl-3b-instruct`, used for EdgeFM VLM inference preparation and local benchmarking.

## Directory Structure

```text
qwen2.5-vl-7b-instruct/
├── README.md               # This document
├── download.sh             # Download the model from HuggingFace
├── engine_config.json      # Engine configuration (prefill_model_path points to the subdirectory)
├── generate.py             # Python inference example
└── qwen2.5-vl-7b-instruct/ # Model files directory
    ├── config.json
    ├── tokenizer.json
    ├── preprocessor_config.json
    └── ...
```

## 1. Place the Model

The model directory currently expected by the repository is:

```bash
examples/qwen2.5-vl-7b-instruct/qwen2.5-vl-7b-instruct/
```

If the directory you are currently syncing is `examples/Qwen2.5-VL-7B-Instruct`, simply move or sync its contents to the target directory above after syncing completes.

For example:

```bash
rsync -a examples/Qwen2.5-VL-7B-Instruct/ examples/qwen2.5-vl-7b-instruct/qwen2.5-vl-7b-instruct/
```

You can also run the download script directly in this directory:

```bash
cd examples/qwen2.5-vl-7b-instruct
bash download.sh
```

## 2. Build EdgeFM

Run in the project root directory:

```bash
mkdir -p build && cd build
cmake .. -DPLATFORM=a800 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
make -j && make install
```

## 3. Run the Example

Run in **this example directory**:

```bash
cd examples/qwen2.5-vl-7b-instruct

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/path/to/edge-fm/build-a800/install/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

python3 generate.py
```

Replace `/path/to/edge-fm` with the actual path of the project root directory.

## Model and Configuration Notes

- In `engine_config.json`, `prefill_model_path` uses the relative path `qwen2.5-vl-7b-instruct`, which is resolved relative to the directory where the configuration file is located.
- `generate.py` provides a text-only smoke test, making it easy to quickly verify the model path and engine configuration.
- For real VLM image input, prepared multimodal, and benchmark comparison, please refer to:
  - `tests/engine/test_qwen2_generate.py`
  - `doc/benchmark_reports/qwen_vlm_suite_20260407.md`
