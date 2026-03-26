#!/bin/bash
# Edge-FM vs TRT-Edge-LLM 算子耗时对比
# 使用 nsys 采集 GPU kernel 耗时，对比两个框架的算子分布
#
# 用法（在项目根目录）:
#   bash scripts/profile_operator_comparison.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/ncu_reports"
PYTHON="${HORIZON_PYTHON:-$(which python)}"
NSYS="${NSYS:-/usr/local/cuda-12.8/bin/nsys}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TRT_PKG="${TRT_PACKAGE_DIR:-/usr/local/TensorRT-10.15.1.29}"

cd "$PROJECT_ROOT"
mkdir -p "$REPORT_DIR"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${TRT_PKG}/lib:${PROJECT_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}/build/python:${PROJECT_ROOT}/build/install/python:${PYTHONPATH:-}"
export EDGELLM_PLUGIN_PATH="${PROJECT_ROOT}/third_party/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
export TRT_EDGELLM_ENGINE_DIR="${PROJECT_ROOT}/tests/data/trt_edgellm_workspace/qwen2.5-1.5b/engines"
export EDGE_FM_DEVICE_ID="${EDGE_FM_DEVICE_ID:-0}"

echo "[1/4] Profiling Edge-FM..."
"$NSYS" profile -o "${REPORT_DIR}/edgefm_profile" --stats=true --force-overwrite=true \
    "$PYTHON" scripts/profile_operator_comparison.py edgefm 2>&1 | tail -5

echo ""
echo "[2/4] Profiling TRT-Edge-LLM..."
"$NSYS" profile -o "${REPORT_DIR}/trt_profile" --stats=true --force-overwrite=true \
    "$PYTHON" scripts/profile_operator_comparison.py trt 2>&1 | tail -5

echo ""
echo "[3/4] Extracting kernel summaries..."
"$NSYS" stats "${REPORT_DIR}/edgefm_profile.nsys-rep" --report gpukernsum -f csv -o "${REPORT_DIR}/edgefm_kernels" 2>/dev/null || true
"$NSYS" stats "${REPORT_DIR}/trt_profile.nsys-rep" --report gpukernsum -f csv -o "${REPORT_DIR}/trt_kernels" 2>/dev/null || true

echo ""
echo "[4/4] Generating comparison report..."
"$PYTHON" scripts/profile_operator_comparison.py analyze

echo ""
echo "Done. Reports:"
echo "  Edge-FM:    ${REPORT_DIR}/edgefm_profile.nsys-rep"
echo "  TRT-Edge:   ${REPORT_DIR}/trt_profile.nsys-rep"
echo "  Kernels:    ${REPORT_DIR}/edgefm_kernels.csv, ${REPORT_DIR}/trt_kernels.csv"
echo ""
echo "View in Nsight Systems: nsys-ui ${REPORT_DIR}/edgefm_profile.nsys-rep"
