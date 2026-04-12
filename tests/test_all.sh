#!/bin/bash

# 支持 horizon_quant (Python 3.10)：export HORIZON_PYTHON=/path/to/python 则使用该 python
PYEXEC="${HORIZON_PYTHON:-pytest}"
if [ -n "$HORIZON_PYTHON" ]; then
    PYEXEC="$HORIZON_PYTHON -m pytest"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/scripts/edge_fm_env.sh"

MULTIARCH_LIB_DIR="/usr/lib/$(edgefm_resolve_multiarch_triplet)"
if [ -f "${MULTIARCH_LIB_DIR}/libstdc++.so.6" ]; then
    export LD_PRELOAD="${MULTIARCH_LIB_DIR}/libstdc++.so.6:${LD_PRELOAD:-}"
fi
if [ -d "${MULTIARCH_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${MULTIARCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

EDGE_FM_BUILD_DIR_RESOLVED="$(edgefm_resolve_build_dir "$PROJECT_ROOT" || true)"
if [ -n "$EDGE_FM_BUILD_DIR_RESOLVED" ]; then
    export EDGE_FM_BUILD_DIR="${EDGE_FM_BUILD_DIR_RESOLVED}"
    if [ -d "${EDGE_FM_BUILD_DIR_RESOLVED}/lib" ]; then
        export LD_LIBRARY_PATH="${EDGE_FM_BUILD_DIR_RESOLVED}/lib:${LD_LIBRARY_PATH:-}"
    fi
fi

cd "$PROJECT_ROOT"

# 若用 horizon_quant，需 PYTHONPATH 包含 edge_fm，以及 CUDA lib 供 edge_fm 加载
if [ -n "$HORIZON_PYTHON" ]; then
    EDGE_FM_PY=""
    if [ -n "$EDGE_FM_BUILD_DIR_RESOLVED" ]; then
        EDGE_FM_PY="${EDGE_FM_BUILD_DIR_RESOLVED}/install/python"
        [ -d "$EDGE_FM_PY" ] || EDGE_FM_PY="${EDGE_FM_BUILD_DIR_RESOLVED}/python"
    fi
    if [ -n "$EDGE_FM_PY" ] && [ -d "$EDGE_FM_PY" ]; then
        export PYTHONPATH="${EDGE_FM_PY}:${PYTHONPATH:-}"
    fi

    CUDA_HOME_RESOLVED="$(edgefm_resolve_cuda_home || true)"
    if [ -n "$CUDA_HOME_RESOLVED" ]; then
        CUDA_LIB_DIR="$(edgefm_resolve_cuda_library_dir "$CUDA_HOME_RESOLVED" || true)"
        if [ -n "$CUDA_LIB_DIR" ]; then
            export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        fi
    fi
fi

# engine
$PYEXEC -s tests/engine/test_kvcache.py
$PYEXEC -s tests/engine/test_qwen2_generate.py

# layers
$PYEXEC -s tests/layers/test_attn.py
$PYEXEC -s tests/layers/test_activation.py
$PYEXEC -s tests/layers/test_layernorm.py
$PYEXEC -s tests/layers/test_sampler.py
$PYEXEC -s tests/layers/test_embed.py
$PYEXEC -s tests/layers/test_linear.py

# utils
$PYEXEC -s tests/utils/test_tensor.py
$PYEXEC -s tests/utils/test_weight_loader.py
