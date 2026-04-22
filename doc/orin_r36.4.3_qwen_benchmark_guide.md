# Orin `r36.4.3` 跑 `qwen2.5-vl-0.5b` Benchmark 简版

## 0. 如果你拿到的是交付包

如果你拿到的是这个目录：

- `deliverables/orin_qwen2_5_vl_0_5b_benchmark_bundle`

优先直接使用交付包里的脚本，不要自己再重新拼路径：

```bash
cd deliverables/orin_qwen2_5_vl_0_5b_benchmark_bundle
bash run_test_benchmark_vlm_0_5b.sh
```

这个交付包已经包含：

- 最小可运行 repo 子集
- 已编好的 `build-orin/install`
- `qwen2.5-vl-0.5b` 模型
- 默认图片 `candy.JPG`

这份文档后面的内容仍然有效，但如果只是帮忙测性能，优先使用交付包脚本即可。

## 目标

只覆盖这一条路径：

- 平台：Jetson Orin
- JetPack：`r36.4.3`
- 模型：`examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b`
- 测试入口：[`tests/engine/test_qwen2_generate.py`](../tests/engine/test_qwen2_generate.py) 里的 `test_benchmark_vlm`

推荐做法：

- 编译和 install 放到 Docker 里做
- benchmark 在 Orin host 的 Python 环境里跑

## 1. 需要准备什么

最少准备下面这些东西：

- 仓库源码
- 模型目录：
  - `examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b`
  - 当前仓库里这份模型约 `1.6G`

额外说明：

- `test_benchmark_vlm` 默认会用 `tests/data/candy.JPG` 和默认 prompt，不需要你再准备图片。
- 这个 benchmark 会优先尝试 `Transformers vs EdgeFM(cuda graph) vs TRT-Edge-LLM`。
- 如果没准备 TRT engine / plugin，它会自动退回成 `Transformers vs EdgeFM(cuda graph)`，所以最小闭环不需要先做 TRT。

## 2. Orin 台架依赖

### 2.1 构建依赖

如果按推荐路径走，Orin host 只需要：

- Docker
- Jetson 容器运行环境可用

### 2.2 跑 benchmark 的 Python 依赖

需要一个能正常用 CUDA 的 Python 3.10 环境。

先安装基础工具：

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv
```

再建一个虚拟环境：

```bash
python3 -m venv .venv-orin-bench
source .venv-orin-bench/bin/activate
python -m pip install --upgrade pip
```

然后安装：

1. 一个和 JetPack 6.1 兼容的 Jetson PyTorch
   这一步建议按 NVIDIA 官方文档装，不建议直接随手从 PyPI 拉 `torch`。
   参考：
   https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

2. 其余 Python 包：

```bash
python -m pip install "transformers>=4.57" safetensors pytest-order prettytable numpy pytest pillow
```

## 3. 如何编译和 install

仓库已经有现成脚本 [`scripts/docker/build_orin.sh`](../scripts/docker/build_orin.sh)。

如果你使用的是上面提到的交付包，而且包里已经带了 `build-orin/install`，这一节可以跳过。

在仓库根目录执行：

```bash
export EDGE_FM_DOCKER_IMAGE=edge-fm-orin:r36.4.3-tools
export EDGE_FM_DOCKER_EXTRA_BUILD_ARGS="--build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.3"
export EDGE_FM_BUILD_JOBS=1

bash scripts/docker/build_orin.sh install
```

这一步会完成：

- `cmake --preset orin`
- `cmake --build --preset orin`
- `cmake --install build-orin`

install 完成后，重点看这两个目录：

- `build-orin/install/lib`
- `build-orin/install/python`

## 4. 怎么跑 benchmark

### 4.1 先跑最小闭环

这是最推荐先执行的命令：

```bash
source .venv-orin-bench/bin/activate

export EDGE_FM_BUILD_DIR="$PWD/build-orin"
export EDGE_FM_DEVICE_ID=0
export EDGE_FM_QWEN_VL_0_5B_MODEL_PATH="$PWD/examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b"
export EDGE_FM_BENCH_VLM_MODELS=0.5b
export EDGE_FM_BENCH_PREFILL_LIST=2048
export EDGE_FM_BENCH_DECODE_LIST=32

pytest -s tests/engine/test_qwen2_generate.py -k test_benchmark_vlm -v
```

这条命令的优点是最稳：

- 只测 `qwen2.5-vl-0.5b`
- 只跑一个 case
- 不要求你先准备 TRT 资产

### 4.2 如果最小闭环通了，再放大一点

```bash
source .venv-orin-bench/bin/activate

export EDGE_FM_BUILD_DIR="$PWD/build-orin"
export EDGE_FM_DEVICE_ID=0
export EDGE_FM_QWEN_VL_0_5B_MODEL_PATH="$PWD/examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b"
export EDGE_FM_BENCH_VLM_MODELS=0.5b
export EDGE_FM_BENCH_PREFILL_LIST=1024,2048
export EDGE_FM_BENCH_DECODE_LIST=32,64

pytest -s tests/engine/test_qwen2_generate.py -k test_benchmark_vlm -v
```

说明：

- 对 VLM 尤其是小模型，过短的 prefill 有时会被脚本自动跳过。
- 所以这里建议直接从 `1024,2048` 开始，不建议先上 `512`。

## 4.3 可选：准备 TRT-Edge-LLM 的 VLM 资产

如果你想让 `test_benchmark_vlm` 真正跑到：

- `Transformers`
- `EdgeFM(cuda graph)`
- `TRT-Edge-LLM`

三方对比，还需要额外准备下面这些东西：

- `TRT-Edge-LLM` 的 VLM text engine
- `TRT-Edge-LLM` 的 multimodal / visual engine
- `libNvInfer_edgellm_plugin.so`
- 对应的 `tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/...` 目录

### A. 先准备 VLM ONNX

这一步更推荐在 x86 + NVIDIA GPU 机器上做，然后把 ONNX 目录拷到 Orin。

`TensorRT-Edge-LLM` 本地文档里有 Qwen2.5-VL-3B 的同类流程。对 `0.5B`，路径替换成 `qwen2.5-vl-0.5b` 即可。

先安装 `TensorRT-Edge-LLM` 的 Python 包，然后执行：

```bash
tensorrt-edgellm-export-llm \
  --model_dir /path/to/qwen2.5-vl-0.5b \
  --output_dir /path/to/trt_edgellm_workspace/qwen2.5-vl-0.5b/onnx

tensorrt-edgellm-export-visual \
  --model_dir /path/to/qwen2.5-vl-0.5b \
  --output_dir /path/to/trt_edgellm_workspace/qwen2.5-vl-0.5b/onnx/visual_enc_onnx
```

建议最后把导出的 ONNX 放成下面这个目录结构：

```bash
tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/onnx
tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/onnx/visual_enc_onnx
```

### B. 在 Orin 上构建 TRT plugin 和 engine

先确保你已经完成了仓库的 Orin 构建，至少有：

```bash
build-orin/trt-edgellm
```

然后补编 `TensorRT-Edge-LLM` 的这几个 target：

```bash
cmake --build build-orin/trt-edgellm --parallel 1 \
  --target llm_build visual_build NvInfer_edgellm_plugin
```

接着生成 VLM text engine：

```bash
mkdir -p tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/engines_mxil2048

build-orin/trt-edgellm/examples/llm/llm_build \
  --onnxDir tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/onnx \
  --engineDir tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/engines_mxil2048 \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --vlm \
  --minImageTokens 128 \
  --maxImageTokens 512
```

再生成 visual engine：

```bash
mkdir -p tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/visual_engines_mxil2048

build-orin/trt-edgellm/examples/multimodal/visual_build \
  --onnxDir tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/onnx/visual_enc_onnx \
  --engineDir tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/visual_engines_mxil2048 \
  --minImageTokens 128 \
  --maxImageTokens 512 \
  --maxImageTokensPerImage 512
```

这里的 `128/512` 参数来自 `TensorRT-Edge-LLM` 本地文档里 Qwen2.5-VL 的示例，是一个合理的起步配置。

最终你至少应该能看到：

```bash
build-orin/trt-edgellm/libNvInfer_edgellm_plugin.so
tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/engines_mxil2048/llm.engine
tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/visual_engines_mxil2048/visual.engine
```

### C. 跑 3-way benchmark

准备好上面的资产后，执行：

```bash
source .venv-orin-bench/bin/activate

export EDGE_FM_BUILD_DIR="$PWD/build-orin"
export EDGE_FM_DEVICE_ID=0
export EDGE_FM_QWEN_VL_0_5B_MODEL_PATH="$PWD/examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b"
export EDGE_FM_BENCH_VLM_MODELS=0.5b
export EDGE_FM_BENCH_PREFILL_LIST=2048
export EDGE_FM_BENCH_DECODE_LIST=32
export TRT_EDGELLM_VLM_ENGINE_DIR_0_5B="$PWD/tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/engines_mxil2048"
export TRT_EDGELLM_VLM_MULTIMODAL_ENGINE_DIR_0_5B="$PWD/tests/data/trt_edgellm_workspace/qwen2.5-vl-0.5b/visual_engines_mxil2048"
export TRT_EDGELLM_PLUGIN_PATH="$PWD/build-orin/trt-edgellm/libNvInfer_edgellm_plugin.so"

pytest -s tests/engine/test_qwen2_generate.py -k test_benchmark_vlm -v
```

如果这三个环境变量都设置正确，`test_benchmark_vlm` 就不会再自动退回到 2-way，而是会尝试真正的 3-way 对比：

- `TRT_EDGELLM_VLM_ENGINE_DIR_0_5B`
- `TRT_EDGELLM_VLM_MULTIMODAL_ENGINE_DIR_0_5B`
- `TRT_EDGELLM_PLUGIN_PATH`

## 5. 你现在最该执行的顺序

```bash
# 1) Docker build + install
export EDGE_FM_DOCKER_IMAGE=edge-fm-orin:r36.4.3-tools
export EDGE_FM_DOCKER_EXTRA_BUILD_ARGS="--build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.3"
export EDGE_FM_BUILD_JOBS=1
bash scripts/docker/build_orin.sh install

# 2) benchmark Python env
python3 -m venv .venv-orin-bench
source .venv-orin-bench/bin/activate
python -m pip install --upgrade pip
# 这里先按 NVIDIA 官方文档安装 Jetson PyTorch
python -m pip install "transformers>=4.57" safetensors pytest-order prettytable numpy pytest pillow

# 3) run minimal VLM benchmark
export EDGE_FM_BUILD_DIR="$PWD/build-orin"
export EDGE_FM_DEVICE_ID=0
export EDGE_FM_QWEN_VL_0_5B_MODEL_PATH="$PWD/examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b"
export EDGE_FM_BENCH_VLM_MODELS=0.5b
export EDGE_FM_BENCH_PREFILL_LIST=2048
export EDGE_FM_BENCH_DECODE_LIST=32
pytest -s tests/engine/test_qwen2_generate.py -k test_benchmark_vlm -v
```

## 6. 常见问题

### 6.1 `No VLM model weights found for benchmark`

说明没找到模型路径。先检查：

```bash
examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b/config.json
examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b/model.safetensors
```

并显式导出：

```bash
export EDGE_FM_QWEN_VL_0_5B_MODEL_PATH="$PWD/examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b"
```

### 6.2 `test_benchmark_vlm` 里提示 TRT 不可用

这个不算 blocker。

只要 `Transformers vs EdgeFM(cuda graph)` 能正常跑，最小闭环就已经通了。

### 6.3 benchmark case 被 skip

对 `qwen2.5-vl-0.5b`，部分较短 prefill case 可能会被脚本自动跳过，这是脚本自身的保护逻辑，不一定是环境问题。

所以建议：

- 先用 `prefill=2048, decode=32`
- 再试 `1024,2048 x 32,64`
