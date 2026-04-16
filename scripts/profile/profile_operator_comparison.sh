#!/bin/bash
# Edge-FM(cuda graph) vs TRT-Edge-LLM 算子耗时对比
# 使用 nsys 采集 GPU kernel 耗时，对比两个框架的算子分布。
# 默认 workload:
#   prefill=2048
#   decode=64
# 可通过 EDGE_FM_PROFILE_PREFILL_LEN / EDGE_FM_PROFILE_DECODE_LEN 覆盖。
#
# 用法（在项目根目录）:
#   bash scripts/profile/profile_operator_comparison.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/ncu_reports"
TRT_PKG="${TRT_PACKAGE_DIR:-/usr/local/TensorRT}"
CUDA_HOME_RESOLVED="${CUDA_HOME:-/usr/local/cuda}"
PYTHON="${HORIZON_PYTHON:-$(which python)}"
BUILD_DIR="$("$PYTHON" "$PROJECT_ROOT/scripts/edge_fm_build_paths.py" --print-build-dir --project-root "$PROJECT_ROOT" --strict)"
if [ -z "$BUILD_DIR" ]; then
    echo "ERROR: no resolved build directory. Set EDGE_FM_BUILD_DIR or build one preset first." >&2
    exit 1
fi
export EDGE_FM_BUILD_DIR="${BUILD_DIR}"
TRT_BUILD_DIR="${EDGE_FM_TRT_BUILD_DIR:-${BUILD_DIR}/trt-edgellm}"
if [ -z "${NSYS:-}" ]; then
    if command -v nsys >/dev/null 2>&1; then
        NSYS="$(command -v nsys)"
    else
        NSYS="$(ls -1d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | sort -V | tail -1)"
    fi
fi
CUDA_LIB_DIR=""
if [ -x "${CUDA_HOME_RESOLVED}/bin/nvcc" ]; then
    if [ -d "${CUDA_HOME_RESOLVED}/lib64" ]; then
        CUDA_LIB_DIR="${CUDA_HOME_RESOLVED}/lib64"
    elif [[ "$(uname -m)" =~ ^(aarch64|arm64)$ ]] && [ -d "${CUDA_HOME_RESOLVED}/targets/aarch64-linux/lib" ]; then
        CUDA_LIB_DIR="${CUDA_HOME_RESOLVED}/targets/aarch64-linux/lib"
    elif [[ "$(uname -m)" =~ ^(x86_64|amd64)$ ]] && [ -d "${CUDA_HOME_RESOLVED}/targets/x86_64-linux/lib" ]; then
        CUDA_LIB_DIR="${CUDA_HOME_RESOLVED}/targets/x86_64-linux/lib"
    fi
fi

if [ -z "${NSYS:-}" ] || [ ! -x "${NSYS}" ]; then
    echo "ERROR: nsys not found. Set NSYS=/path/to/nsys and retry." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$REPORT_DIR"

if [ -n "$CUDA_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
if [ -d "${TRT_PKG}/lib" ]; then
    export LD_LIBRARY_PATH="${TRT_PKG}/lib:${LD_LIBRARY_PATH:-}"
fi
if [ -d "${BUILD_DIR}/lib" ]; then
    export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi
if [ -d "${BUILD_DIR}/python" ] || [ -d "${BUILD_DIR}/install/python" ]; then
    export PYTHONPATH="${BUILD_DIR}/python:${BUILD_DIR}/install/python:${PYTHONPATH:-}"
fi
export EDGE_FM_TRT_BUILD_DIR="${TRT_BUILD_DIR}"
export EDGELLM_PLUGIN_PATH="${TRT_BUILD_DIR}/libNvInfer_edgellm_plugin.so"
export TRT_EDGELLM_ENGINE_DIR="${PROJECT_ROOT}/tests/data/trt_edgellm_workspace/qwen2.5-1.5b/engines"
export EDGE_FM_DEVICE_ID="${EDGE_FM_DEVICE_ID:-1}"
export EDGE_FM_PROFILE_PREFILL_LEN="${EDGE_FM_PROFILE_PREFILL_LEN:-2048}"
export EDGE_FM_PROFILE_DECODE_LEN="${EDGE_FM_PROFILE_DECODE_LEN:-64}"

echo "[1/4] Profiling Edge-FM..."
"$NSYS" profile -o "${REPORT_DIR}/edgefm_profile" --stats=true --force-overwrite=true \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    "$PYTHON" "$PROJECT_ROOT/scripts/profile/profile_operator_comparison.py" edgefm 2>&1 | tail -5

echo ""
echo "[2/4] Profiling TRT-Edge-LLM..."
"$NSYS" profile -o "${REPORT_DIR}/trt_profile" --stats=true --force-overwrite=true \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    "$PYTHON" "$PROJECT_ROOT/scripts/profile/profile_operator_comparison.py" trt 2>&1 | tail -5

echo ""
echo "[3/4] Extracting kernel summaries..."
"$NSYS" stats --force-export=true --report cuda_gpu_kern_sum --format csv \
    --output "${REPORT_DIR}/edgefm_kernels" "${REPORT_DIR}/edgefm_profile.nsys-rep" 2>/dev/null || true
"$NSYS" stats --force-export=true --report cuda_gpu_kern_sum --format csv \
    --output "${REPORT_DIR}/trt_kernels" "${REPORT_DIR}/trt_profile.nsys-rep" 2>/dev/null || true

echo ""
echo "[4/4] Generating comparison report..."
"$PYTHON" "$PROJECT_ROOT/scripts/profile/profile_operator_comparison.py" analyze

echo ""
echo "Done. Reports:"
echo "  Edge-FM:    ${REPORT_DIR}/edgefm_profile.nsys-rep"
echo "  TRT-Edge:   ${REPORT_DIR}/trt_profile.nsys-rep"
echo "  Kernels:    ${REPORT_DIR}/edgefm_kernels_cuda_gpu_kern_sum.csv, ${REPORT_DIR}/trt_kernels_cuda_gpu_kern_sum.csv"
echo ""
echo "View in Nsight Systems: nsys-ui ${REPORT_DIR}/edgefm_profile.nsys-rep"
