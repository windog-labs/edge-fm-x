#!/bin/bash

set -u

# 支持 horizon_quant (Python 3.10)：export HORIZON_PYTHON=/path/to/python 则使用该 python
PYEXEC="${HORIZON_PYTHON:-pytest}"
PYTHON_FOR_HELPER="${HORIZON_PYTHON:-python3}"
if [ -n "${HORIZON_PYTHON:-}" ]; then
    PYEXEC="$HORIZON_PYTHON -m pytest"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$("$PYTHON_FOR_HELPER" "$PROJECT_ROOT/scripts/edge_fm_build_paths.py" --print-build-dir --project-root "$PROJECT_ROOT" --strict)"
if [ -z "$BUILD_DIR" ]; then
    echo "ERROR: no resolved build directory. Set EDGE_FM_BUILD_DIR or build one preset first." >&2
    exit 1
fi
export EDGE_FM_BUILD_DIR="${BUILD_DIR}"
CUDA_HOME_RESOLVED="${CUDA_HOME:-/usr/local/cuda}"
case "$(uname -m)" in
    aarch64|arm64)
        MULTIARCH_TRIPLET="aarch64-linux-gnu"
        CUDA_TARGET_TRIPLE="aarch64-linux"
        ;;
    x86_64|amd64)
        MULTIARCH_TRIPLET="x86_64-linux-gnu"
        CUDA_TARGET_TRIPLE="x86_64-linux"
        ;;
    *)
        MULTIARCH_TRIPLET=""
        CUDA_TARGET_TRIPLE=""
        ;;
esac

MULTIARCH_LIB_DIR=""
if [ -n "$MULTIARCH_TRIPLET" ]; then
    MULTIARCH_LIB_DIR="/usr/lib/${MULTIARCH_TRIPLET}"
fi
if [ -n "$MULTIARCH_LIB_DIR" ] && [ -f "${MULTIARCH_LIB_DIR}/libstdc++.so.6" ]; then
    export LD_PRELOAD="${MULTIARCH_LIB_DIR}/libstdc++.so.6:${LD_PRELOAD:-}"
fi
if [ -n "$MULTIARCH_LIB_DIR" ] && [ -d "${MULTIARCH_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${MULTIARCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

if [ -d "${BUILD_DIR}/lib" ]; then
    export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

cd "$PROJECT_ROOT"

OVERALL_STATUS=0
run_pytest() {
    "$@" || OVERALL_STATUS=$?
}

# 若用 horizon_quant，需 PYTHONPATH 包含 edge_fm，以及 CUDA lib 供 edge_fm 加载
if [ -n "${HORIZON_PYTHON:-}" ]; then
    EDGE_FM_PY=""
    EDGE_FM_PY="${BUILD_DIR}/install/python"
    [ -d "$EDGE_FM_PY" ] || EDGE_FM_PY="${BUILD_DIR}/python"
    if [ -n "$EDGE_FM_PY" ] && [ -d "$EDGE_FM_PY" ]; then
        export PYTHONPATH="${EDGE_FM_PY}:${PYTHONPATH:-}"
    fi

    if [ -x "${CUDA_HOME_RESOLVED}/bin/nvcc" ]; then
        CUDA_LIB_DIR=""
        if [ -d "${CUDA_HOME_RESOLVED}/lib64" ]; then
            CUDA_LIB_DIR="${CUDA_HOME_RESOLVED}/lib64"
        elif [ -n "$CUDA_TARGET_TRIPLE" ] && [ -d "${CUDA_HOME_RESOLVED}/targets/${CUDA_TARGET_TRIPLE}/lib" ]; then
            CUDA_LIB_DIR="${CUDA_HOME_RESOLVED}/targets/${CUDA_TARGET_TRIPLE}/lib"
        fi
        if [ -n "$CUDA_LIB_DIR" ]; then
            export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        fi
    fi
fi

# engine
run_pytest $PYEXEC -s tests/engine/test_kvcache.py
run_pytest $PYEXEC -s tests/engine/test_qwen2_generate.py

# layers
run_pytest $PYEXEC -s tests/layers/test_attn.py
run_pytest $PYEXEC -s tests/layers/test_activation.py
run_pytest $PYEXEC -s tests/layers/test_layernorm.py
run_pytest $PYEXEC -s tests/layers/test_sampler.py
run_pytest $PYEXEC -s tests/layers/test_embed.py
run_pytest $PYEXEC -s tests/layers/test_linear.py

# utils
run_pytest $PYEXEC -s tests/utils/test_tensor.py
run_pytest $PYEXEC -s tests/utils/test_weight_loader.py

exit "$OVERALL_STATUS"
