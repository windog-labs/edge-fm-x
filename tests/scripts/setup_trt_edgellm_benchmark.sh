#!/bin/bash
# TensorRT-Edge-LLM benchmark 预置脚本
# 导出指定 Qwen2.5 LLM ONNX，构建 engine。首次运行 benchmark 前执行一次。
#
# 用法（在项目根目录）:
#   bash tests/scripts/setup_trt_edgellm_benchmark.sh
#   EDGE_FM_TRT_MODEL_SIZE=0.5b bash tests/scripts/setup_trt_edgellm_benchmark.sh
#   EDGE_FM_TRT_MODEL_SIZE=3b EDGE_FM_QWEN_3B_MODEL_PATH=/path/to/model bash tests/scripts/setup_trt_edgellm_benchmark.sh
#
# 需先: conda activate horizon_quant, pip install third_party/TensorRT-Edge-LLM

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/scripts/edge_fm_env.sh"
TRT_EDGELLM="$PROJECT_ROOT/third_party/TensorRT-Edge-LLM"
WORKSPACE="$PROJECT_ROOT/tests/data/trt_edgellm_workspace"
MODEL_SIZE="${EDGE_FM_TRT_MODEL_SIZE:-1.5b}"
MAX_INPUT_LEN="${EDGE_FM_TRT_MAX_INPUT_LEN:-2048}"
MAX_KV_CACHE_CAPACITY="${EDGE_FM_TRT_MAX_KV_CACHE_CAPACITY:-4096}"
CONDA_ENV_PREFIX="${EDGE_FM_TRT_CONDA_PREFIX:-}"
if [[ -n "$CONDA_ENV_PREFIX" && -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
    PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$CONDA_ENV_PREFIX/bin/python}"
else
    PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
fi
TRT_EXPORT_DEVICE="${EDGE_FM_TRT_EXPORT_DEVICE:-cuda}"
case "$MODEL_SIZE" in
    0.5b)
        MODEL_NAME="qwen2.5-0.5b"
        MODEL_PATH="${EDGE_FM_QWEN_0_5B_MODEL_PATH:-$PROJECT_ROOT/examples/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct}"
        ;;
    1.5b)
        MODEL_NAME="qwen2.5-1.5b"
        MODEL_PATH="${EDGE_FM_QWEN_1_5B_MODEL_PATH:-${EDGE_FM_QWEN_MODEL_PATH:-$PROJECT_ROOT/examples/qwen2.5-1.5b-instruct/qwen2.5-1.5b-instruct}}"
        ;;
    3b)
        MODEL_NAME="qwen2.5-3b"
        MODEL_PATH="${EDGE_FM_QWEN_3B_MODEL_PATH:-$PROJECT_ROOT/examples/qwen2.5-3b-instruct/qwen2.5-3b-instruct}"
        ;;
    *)
        echo "ERROR: unsupported EDGE_FM_TRT_MODEL_SIZE=$MODEL_SIZE (supported: 0.5b, 1.5b, 3b)"
        exit 1
        ;;
esac
ONNX_DIR="$WORKSPACE/$MODEL_NAME/onnx"
ENGINE_DIR="$WORKSPACE/$MODEL_NAME/engines_mxil${MAX_INPUT_LEN}"
TRT_PKG="$(edgefm_resolve_tensorrt_package_dir "$PROJECT_ROOT" "$WORKSPACE")" || {
    echo "ERROR: TensorRT headers/libraries not found. Set TRT_PACKAGE_DIR and retry."
    exit 1
}
CUDA_HOME_RESOLVED="$(edgefm_resolve_cuda_home)" || {
    echo "ERROR: unable to locate CUDA toolkit. Set CUDA_HOME and retry."
    exit 1
}
CUDA_VERSION_RESOLVED="${EDGE_FM_TRT_CUDA_VERSION:-$(edgefm_resolve_cuda_version "$CUDA_HOME_RESOLVED")}"
if [[ -z "$CUDA_VERSION_RESOLVED" ]]; then
    echo "ERROR: unable to determine CUDA version from ${CUDA_HOME_RESOLVED}. Set EDGE_FM_TRT_CUDA_VERSION and retry."
    exit 1
fi
CUDA_NVCC="${EDGE_FM_TRT_CMAKE_CUDA_COMPILER:-${CUDA_HOME_RESOLVED}/bin/nvcc}"
CUDA_ARCHS="${EDGE_FM_TRT_CMAKE_CUDA_ARCHITECTURES:-$(edgefm_default_trt_cuda_architectures || true)}"
TRT_BUILD_DIR="${EDGE_FM_TRT_BUILD_DIR:-$TRT_EDGELLM/build}"
TRT_BUILD_LLM_BINARY="${TRT_BUILD_DIR}/examples/llm/llm_build"
if [[ -n "${EDGE_FM_TRT_BUILD_JOBS:-}" ]]; then
    TRT_BUILD_JOBS="${EDGE_FM_TRT_BUILD_JOBS}"
elif [[ "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
    TRT_BUILD_JOBS=1
else
    TRT_BUILD_JOBS="$(nproc)"
fi

if [[ "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
    TRT_CUDA_DIR_DEFAULT="${CUDA_HOME_RESOLVED}/targets/$(edgefm_resolve_cuda_target_triple)"
else
    TRT_CUDA_DIR_DEFAULT="${CUDA_HOME_RESOLVED}"
fi
TRT_CUDA_DIR="${EDGE_FM_TRT_CUDA_DIR:-$TRT_CUDA_DIR_DEFAULT}"

resolve_export_llm_cmd() {
    if [[ -n "$CONDA_ENV_PREFIX" && -x "$PYTHON_EXECUTABLE" ]]; then
        if "$PYTHON_EXECUTABLE" - <<'PY' &>/dev/null
import sys
sys.path.insert(0, "third_party/TensorRT-Edge-LLM")
from tensorrt_edgellm.scripts.export_llm import main  # noqa: F401
print("ok")
PY
        then
            echo "PYTHONPATH=\"$TRT_EDGELLM\" \"$PYTHON_EXECUTABLE\" \"$TRT_EDGELLM/tensorrt_edgellm/scripts/export_llm.py\""
            return 0
        fi
    fi

    if command -v conda >/dev/null 2>&1; then
        local conda_prefix
        conda_prefix="$(conda env list 2>/dev/null | awk '$1 == "horizon_quant" {print $2}' | head -1)"
        if [[ -n "$conda_prefix" ]] && conda run -n horizon_quant which tensorrt-edgellm-export-llm &>/dev/null; then
            echo "conda run -n horizon_quant tensorrt-edgellm-export-llm"
            return 0
        fi
    fi

    if "$PYTHON_EXECUTABLE" - <<'PY' &>/dev/null
import sys
sys.path.insert(0, "third_party/TensorRT-Edge-LLM")
from tensorrt_edgellm.scripts.export_llm import main  # noqa: F401
print("ok")
PY
    then
        echo "PYTHONPATH=\"$TRT_EDGELLM\" \"$PYTHON_EXECUTABLE\" \"$TRT_EDGELLM/tensorrt_edgellm/scripts/export_llm.py\""
        return 0
    fi

    return 1
}

cd "$PROJECT_ROOT"

# 1. 初始化子模块与 nlohmann json
echo "[1/4] Initializing TensorRT-Edge-LLM submodules..."
cd "$TRT_EDGELLM"
git submodule update --init --recursive 2>/dev/null || true
# nlohmannJson 子模块可能为空，使用 edge-fm 的 json 作为后备
if [[ ! -f "$TRT_EDGELLM/3rdParty/nlohmannJson/include/nlohmann/json.hpp" ]]; then
    mkdir -p "$TRT_EDGELLM/3rdParty/nlohmannJson/include"
    ln -sf "$PROJECT_ROOT/third_party/json/include/nlohmann" "$TRT_EDGELLM/3rdParty/nlohmannJson/include/" 2>/dev/null || true
fi
cd "$PROJECT_ROOT"

# 2. 导出 ONNX（FP16，小模型无需量化）
if [[ ! -d "$ONNX_DIR" ]] || [[ -z "$(ls -A "$ONNX_DIR" 2>/dev/null)" ]]; then
    echo "[2/4] Exporting ONNX from $MODEL_PATH..."
    mkdir -p "$ONNX_DIR"
    EXPORT_CMD="$(resolve_export_llm_cmd)" || {
        echo "ERROR: unable to find a usable TensorRT-Edge-LLM export command."
        echo "  Tried:"
        echo "    1. conda run -n horizon_quant tensorrt-edgellm-export-llm"
        echo "    2. repo-local export_llm.py via PYTHONPATH=$TRT_EDGELLM"
        exit 1
    }
    eval "$EXPORT_CMD" \
        --model_dir "\"$MODEL_PATH\"" \
        --output_dir "\"$ONNX_DIR\"" \
        --device "\"$TRT_EXPORT_DEVICE\""
else
    echo "[2/4] ONNX already exists at $ONNX_DIR, skipping export."
fi

# 3. 构建 C++ 运行时
if [[ ! -f "$TRT_BUILD_LLM_BINARY" ]]; then
    echo "[3/4] Building TensorRT-Edge-LLM C++ runtime..."
    mkdir -p "$TRT_BUILD_DIR"
    cmake_args=(
        -DCMAKE_BUILD_TYPE=Release
        -DTRT_PACKAGE_DIR="$TRT_PKG"
        -DCUDA_VERSION="$CUDA_VERSION_RESOLVED"
        -DCUDA_DIR="$TRT_CUDA_DIR"
        -DCMAKE_CUDA_COMPILER="$CUDA_NVCC"
    )
    if [[ -n "$CUDA_ARCHS" ]]; then
        cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS")
    fi
    cmake -S "$TRT_EDGELLM" -B "$TRT_BUILD_DIR" "${cmake_args[@]}"
    cmake --build "$TRT_BUILD_DIR" --parallel "$TRT_BUILD_JOBS"
else
    echo "[3/4] C++ runtime already built, skipping."
fi

# 4. 构建 Engine
echo "[4/4] Building TensorRT engine..."
mkdir -p "$ENGINE_DIR"
export EDGELLM_PLUGIN_PATH="$TRT_BUILD_DIR/libNvInfer_edgellm_plugin.so"
export LD_LIBRARY_PATH="${TRT_PKG}/lib:${LD_LIBRARY_PATH:-}"
"$TRT_BUILD_LLM_BINARY" \
    --onnxDir "$ONNX_DIR" \
    --engineDir "$ENGINE_DIR" \
    --maxBatchSize 1 \
    --maxInputLen "$MAX_INPUT_LEN" \
    --maxKVCacheCapacity "$MAX_KV_CACHE_CAPACITY"

echo ""
echo "Done. Engine at: $ENGINE_DIR"
echo "Run benchmark: EDGE_FM_BENCH_LLM_MODELS=$MODEL_SIZE pytest -s tests/engine/test_qwen2_generate.py -k benchmark_trt_edgellm -v"
