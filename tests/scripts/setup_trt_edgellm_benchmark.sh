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
TRT_EDGELLM="$PROJECT_ROOT/third_party/TensorRT-Edge-LLM"
WORKSPACE="$PROJECT_ROOT/tests/data/trt_edgellm_workspace"
MODEL_SIZE="${EDGE_FM_TRT_MODEL_SIZE:-1.5b}"
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
ENGINE_DIR="$WORKSPACE/$MODEL_NAME/engines"
TRT_PKG="${TRT_PACKAGE_DIR:-/usr/local/TensorRT-10.15.1.29}"

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
    if ! conda run -n horizon_quant which tensorrt-edgellm-export-llm &>/dev/null; then
        echo "ERROR: tensorrt-edgellm not installed in horizon_quant."
        echo "  Run: conda activate horizon_quant && pip install third_party/TensorRT-Edge-LLM"
        echo "  (May take several minutes due to nvidia-modelopt dependencies)"
        exit 1
    fi
    conda run -n horizon_quant tensorrt-edgellm-export-llm \
        --model_dir "$MODEL_PATH" \
        --output_dir "$ONNX_DIR"
else
    echo "[2/4] ONNX already exists at $ONNX_DIR, skipping export."
fi

# 3. 构建 C++ 运行时
if [[ ! -f "$TRT_EDGELLM/build/examples/llm/llm_build" ]]; then
    echo "[3/4] Building TensorRT-Edge-LLM C++ runtime..."
    mkdir -p "$TRT_EDGELLM/build"
    cd "$TRT_EDGELLM/build"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR="$TRT_PKG" \
        -DCUDA_VERSION=12.8 \
        -DCMAKE_CUDA_COMPILER="${CUDA_HOME:-/usr/local/cuda-12.8}/bin/nvcc" \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
    make -j$(nproc)
    cd "$PROJECT_ROOT"
else
    echo "[3/4] C++ runtime already built, skipping."
fi

# 4. 构建 Engine
echo "[4/4] Building TensorRT engine..."
mkdir -p "$ENGINE_DIR"
export EDGELLM_PLUGIN_PATH="$TRT_EDGELLM/build/libNvInfer_edgellm_plugin.so"
export LD_LIBRARY_PATH="${TRT_PKG}/lib:${LD_LIBRARY_PATH:-}"
"$TRT_EDGELLM/build/examples/llm/llm_build" \
    --onnxDir "$ONNX_DIR" \
    --engineDir "$ENGINE_DIR" \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 4096

echo ""
echo "Done. Engine at: $ENGINE_DIR"
echo "Run benchmark: EDGE_FM_BENCH_LLM_MODELS=$MODEL_SIZE pytest -s tests/engine/test_qwen2_generate.py -k benchmark_trt_edgellm -v"
