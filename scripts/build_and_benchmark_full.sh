#!/bin/bash
# 完整构建 + 性能对比：edge-fm vs TRT-Edge-LLM
# 统一使用 CUDA 12.8
#
# 用法（在项目根目录）:
#   bash scripts/build_and_benchmark_full.sh
#
# 环境变量:
#   EDGE_FM_QWEN_MODEL_PATH  模型路径（默认 examples/qwen2.5-1.5b-instruct/...）
#   EDGE_FM_DEVICE_ID        GPU 设备 ID（默认 0）

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HORIZON_PYTHON="${HORIZON_PYTHON:-/home/zhangzimo/miniconda3/envs/horizon_quant/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TRT_PKG="${TRT_PACKAGE_DIR:-/usr/local/TensorRT-10.15.1.29}"

cd "$PROJECT_ROOT"

# 1. 确保 TRT-Edge-LLM engine 已构建
echo "[1/4] Checking TRT-Edge-LLM setup..."
if ! bash tests/scripts/setup_trt_edgellm_benchmark.sh 2>/dev/null; then
    echo "WARNING: setup_trt_edgellm_benchmark.sh had issues. Continuing..."
fi

# 2. 完整构建 edge-fm（含 edge_fm_trt）
echo ""
echo "[2/4] Building edge-fm with CUDA 12.8 + TRT pybind..."
rm -rf build
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLATFORM=a100 \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DPython3_EXECUTABLE="$HORIZON_PYTHON" \
  -DBUILD_PYTHON=ON \
  -DBUILD_TRT_EDGELLM_PYBIND=ON \
  -DTRT_PACKAGE_DIR="$TRT_PKG"

make -j$(nproc)
make install

# 3. 运行 benchmark
echo ""
echo "[3/4] Running performance benchmark (EdgeFM vs TRT-Edge-LLM)..."
cd "$PROJECT_ROOT"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${TRT_PKG}/lib:${PROJECT_ROOT}/build/lib:${PROJECT_ROOT}/build/install/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}/build/python:${PROJECT_ROOT}/build/install/python:${PYTHONPATH:-}"
export EDGELLM_PLUGIN_PATH="${PROJECT_ROOT}/third_party/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
export EDGE_FM_DEVICE_ID="${EDGE_FM_DEVICE_ID:-0}"
export TRT_EDGELLM_ENGINE_DIR="${PROJECT_ROOT}/tests/data/trt_edgellm_workspace/qwen2.5-1.5b/engines"

"$HORIZON_PYTHON" -m pytest -s tests/engine/test_qwen2_generate.py -k benchmark_trt_edgellm -v 2>&1 || true

echo ""
echo "[4/4] Done."
echo "  edge_fm:      build/install/python/edge_fm*.so"
echo "  edge_fm_trt: build/install/python/edge_fm_trt*.so"
echo "  TRT engine:   tests/data/trt_edgellm_workspace/qwen2.5-1.5b/engines/"
