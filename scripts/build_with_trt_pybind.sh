#!/bin/bash
# edge-fm 构建脚本：启用 TRT-Edge-LLM Python 绑定 (edge_fm_trt)
# 用于 in-process benchmark，避免 subprocess 开销
#
# 用法（在项目根目录）:
#   bash scripts/build_with_trt_pybind.sh
#
# 需先运行: bash tests/scripts/setup_trt_edgellm_benchmark.sh (构建 TRT engine)
# 重要：CUDA 版本需与 TRT-Edge-LLM 构建时一致（通常 12.8），否则可能出现 undefined symbol

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HORIZON_PYTHON="${HORIZON_PYTHON:-/home/zhangzimo/miniconda3/envs/horizon_quant/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TRT_PKG="${TRT_PACKAGE_DIR:-/usr/local/TensorRT-10.15.1.29}"

cd "$PROJECT_ROOT"
mkdir -p build
cd build

# 使用与 TRT-Edge-LLM 相同的 CUDA 版本，避免 cudaGetDeviceProperties_v2 等符号未定义
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

echo ""
echo "Done. edge_fm + edge_fm_trt installed to build/install/python/"
echo "Run benchmark: EDGE_FM_QWEN_MODEL_PATH=... EDGE_FM_DEVICE_ID=0 pytest -s tests/engine/test_qwen2_generate.py -k benchmark_trt_edgellm -v"
