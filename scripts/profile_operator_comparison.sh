#!/bin/bash
# Edge-FM(cuda graph) vs TRT-Edge-LLM 算子耗时对比
# 使用 nsys 采集 GPU kernel 耗时，对比两个框架的算子分布。
# 默认 workload:
#   prefill=2048
#   decode=64
# 可通过 EDGE_FM_PROFILE_PREFILL_LEN / EDGE_FM_PROFILE_DECODE_LEN 覆盖。
#
# 用法（在项目根目录）:
#   bash scripts/profile_operator_comparison.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/scripts/edge_fm_env.sh"
REPORT_DIR="${PROJECT_ROOT}/ncu_reports"
PYTHON="${HORIZON_PYTHON:-$(which python)}"
if [ -z "${NSYS:-}" ]; then
    if command -v nsys >/dev/null 2>&1; then
        NSYS="$(command -v nsys)"
    else
        NSYS="$(ls -1d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | sort -V | tail -1)"
    fi
fi
CUDA_HOME="$(edgefm_resolve_cuda_home || true)"
CUDA_LIB_DIR=""
if [ -n "$CUDA_HOME" ]; then
    CUDA_LIB_DIR="$(edgefm_resolve_cuda_library_dir "$CUDA_HOME" || true)"
fi
TRT_PKG="$(edgefm_resolve_tensorrt_package_dir "$PROJECT_ROOT" "$PROJECT_ROOT/tests/data/trt_edgellm_workspace" || true)"
EDGE_FM_BUILD_DIR_RESOLVED="$(edgefm_resolve_build_dir "$PROJECT_ROOT" || true)"

if [ -z "${NSYS:-}" ] || [ ! -x "${NSYS}" ]; then
    echo "ERROR: nsys not found. Set NSYS=/path/to/nsys and retry." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$REPORT_DIR"

if [ -n "$CUDA_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
if [ -n "$TRT_PKG" ]; then
    export LD_LIBRARY_PATH="${TRT_PKG}/lib:${LD_LIBRARY_PATH:-}"
fi
if [ -n "$EDGE_FM_BUILD_DIR_RESOLVED" ]; then
    export EDGE_FM_BUILD_DIR="${EDGE_FM_BUILD_DIR_RESOLVED}"
    export LD_LIBRARY_PATH="${EDGE_FM_BUILD_DIR_RESOLVED}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${EDGE_FM_BUILD_DIR_RESOLVED}/python:${EDGE_FM_BUILD_DIR_RESOLVED}/install/python:${PYTHONPATH:-}"
fi
export EDGELLM_PLUGIN_PATH="${PROJECT_ROOT}/third_party/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
export TRT_EDGELLM_ENGINE_DIR="${PROJECT_ROOT}/tests/data/trt_edgellm_workspace/qwen2.5-1.5b/engines"
export EDGE_FM_DEVICE_ID="${EDGE_FM_DEVICE_ID:-1}"
export EDGE_FM_PROFILE_PREFILL_LEN="${EDGE_FM_PROFILE_PREFILL_LEN:-2048}"
export EDGE_FM_PROFILE_DECODE_LEN="${EDGE_FM_PROFILE_DECODE_LEN:-64}"

echo "[1/4] Profiling Edge-FM..."
"$NSYS" profile -o "${REPORT_DIR}/edgefm_profile" --stats=true --force-overwrite=true \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    "$PYTHON" scripts/profile_operator_comparison.py edgefm 2>&1 | tail -5

echo ""
echo "[2/4] Profiling TRT-Edge-LLM..."
"$NSYS" profile -o "${REPORT_DIR}/trt_profile" --stats=true --force-overwrite=true \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    "$PYTHON" scripts/profile_operator_comparison.py trt 2>&1 | tail -5

echo ""
echo "[3/4] Extracting kernel summaries..."
"$NSYS" stats --force-export=true --report cuda_gpu_kern_sum --format csv \
    --output "${REPORT_DIR}/edgefm_kernels" "${REPORT_DIR}/edgefm_profile.nsys-rep" 2>/dev/null || true
"$NSYS" stats --force-export=true --report cuda_gpu_kern_sum --format csv \
    --output "${REPORT_DIR}/trt_kernels" "${REPORT_DIR}/trt_profile.nsys-rep" 2>/dev/null || true

echo ""
echo "[4/4] Generating comparison report..."
"$PYTHON" scripts/profile_operator_comparison.py analyze

echo ""
echo "Done. Reports:"
echo "  Edge-FM:    ${REPORT_DIR}/edgefm_profile.nsys-rep"
echo "  TRT-Edge:   ${REPORT_DIR}/trt_profile.nsys-rep"
echo "  Kernels:    ${REPORT_DIR}/edgefm_kernels_cuda_gpu_kern_sum.csv, ${REPORT_DIR}/trt_kernels_cuda_gpu_kern_sum.csv"
echo ""
echo "View in Nsight Systems: nsys-ui ${REPORT_DIR}/edgefm_profile.nsys-rep"
