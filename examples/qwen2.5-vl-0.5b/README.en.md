# Qwen2.5-VL-0.5B Example

This directory provides an EdgeFM example entry point for the model files under `examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b/`.

## Directory Structure

```text
qwen2.5-vl-0.5b/
├── README.md              # This document
├── engine_config.json     # EdgeFM engine configuration
├── generate.py            # Text-side smoke test
└── qwen2.5-vl-0.5b/       # Model files directory
    ├── config.json
    ├── tokenizer.json
    ├── preprocessor_config.json
    └── ...
```

## Usage

First, build and install the Python bindings in the project root directory:

```bash
mkdir -p build && cd build
cmake .. -DPLATFORM=a800 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
make -j && make install
```

Then run in this directory:

```bash
cd examples/qwen2.5-vl-0.5b
python3 generate.py
```

If `edge_fm` cannot be imported, first confirm that the Python version you use to run the script has an ABI consistent with `build-<platform>/install/python/edge_fm*.so`. For example, if the current environment defaults to Python 3.12 while the extension is `cpython-310`, you need to switch to the corresponding Python version and re-run, or recompile pybind.

## Configuration Notes

- `engine_config.json` reuses the configuration form of the existing VLM examples, with `model_name` being `Qwen2.5-VL`, and the default operator table is provided by the platform configuration in the install directory.
