#!/bin/bash
# edge-fm 构建脚本：horizon_quant (Python 3.10) + CUDA 12.8
# 与 PyTorch/CUDA 12 环境兼容，避免 __cudaLaunchKernel 等符号冲突
#
# 用法（在项目根目录）:
#   bash scripts/build_horizon_quant.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HORIZON_PYTHON="${HORIZON_PYTHON:-/home/zhangzimo/miniconda3/envs/horizon_quant/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

cd "$PROJECT_ROOT"
rm -rf build
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLATFORM=a100 \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DPython3_EXECUTABLE="$HORIZON_PYTHON" \
  -DBUILD_PYTHON=ON

make -j$(nproc)
make install

echo ""
echo "Done. edge_fm installed to build/install/python/"
echo "Run benchmark: EDGE_FM_QWEN_MODEL_PATH=... EDGE_FM_DEVICE_ID=0 pytest -s tests/engine/test_qwen2_generate.py -k benchmark -v"
