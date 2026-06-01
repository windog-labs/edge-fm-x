# Qwen2.5-3B-Instruct Example

This directory provides a **Qwen2.5-3B-Instruct** example with the same structure as `qwen2.5-1.5b-instruct`, used for EdgeFM inference.

## Directory Structure

```text
qwen2.5-3b-instruct/
├── README.md           # This document
├── download.sh         # Download the model from HuggingFace
├── engine_config.json  # Engine configuration (prefill_model_path points to the subdirectory)
├── generate.py         # Python inference example
└── qwen2.5-3b-instruct/   # Model files
    ├── config.json
    ├── model-00001-of-00002.safetensors
    ├── tokenizer.json
    └── ...
```

If the `qwen2.5-3b-instruct/` subdirectory already contains the model files, there is no need to download again.

## 1. Download the Model

```bash
cd examples/qwen2.5-3b-instruct
bash download.sh
```

The model will be downloaded to `qwen2.5-3b-instruct/` under the current directory.

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
cd examples/qwen2.5-3b-instruct

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/path/to/edge-fm/build-a800/install/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

python3 generate.py
```

Replace `/path/to/edge-fm` with the actual path of the project root directory.

## Model and Configuration Notes

- **Qwen2.5-3B-Instruct**: `num_attention_heads=16`, `num_key_value_heads=2`, `group_size=8`.
- In `engine_config.json`, `prefill_model_path` uses the relative path `qwen2.5-3b-instruct`, which is resolved relative to the directory where the configuration file is located, so it is recommended to run `generate.py` under `examples/qwen2.5-3b-instruct`.
- `speculative.enabled=false`, so even without a draft model, the default run is not affected.
