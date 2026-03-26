#!/bin/bash

# 优先使用系统的 libstdc++
if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:${LD_PRELOAD:-}
fi
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# 支持 horizon_quant (Python 3.10)：export HORIZON_PYTHON=/path/to/python 则使用该 python
PYEXEC="${HORIZON_PYTHON:-pytest}"
if [ -n "$HORIZON_PYTHON" ]; then
    PYEXEC="$HORIZON_PYTHON -m pytest"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 若用 horizon_quant，需 PYTHONPATH 包含 edge_fm，以及 CUDA lib 供 edge_fm 加载
if [ -n "$HORIZON_PYTHON" ]; then
    EDGE_FM_PY="$PROJECT_ROOT/build/install/python"
    [ -d "$EDGE_FM_PY" ] || EDGE_FM_PY="$PROJECT_ROOT/build/python"
    export PYTHONPATH="${EDGE_FM_PY}:${PYTHONPATH:-}"
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
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
