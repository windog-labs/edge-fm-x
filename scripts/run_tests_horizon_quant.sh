#!/bin/bash
# 使用 horizon_quant (Python 3.10) 运行 edge-fm 测试
# 需先: bash scripts/build_horizon_quant.sh
#
# 用法（在项目根目录）:
#   bash scripts/run_tests_horizon_quant.sh
#   bash scripts/run_tests_horizon_quant.sh -k benchmark   # 仅跑 benchmark
#   EDGE_FM_DEVICE_ID=0 bash scripts/run_tests_horizon_quant.sh   # 单 GPU 时指定 device 0

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HORIZON_PYTHON="${HORIZON_PYTHON:-/home/zhangzimo/miniconda3/envs/horizon_quant/bin/python}"

# 优先使用系统的 libstdc++（与 test_all.sh 一致）
if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:${LD_PRELOAD:-}
fi
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_ROOT"

EDGE_FM_PY="$PROJECT_ROOT/build/install/python"
[ -d "$EDGE_FM_PY" ] || EDGE_FM_PY="$PROJECT_ROOT/build/python"
export PYTHONPATH="${EDGE_FM_PY}:${PYTHONPATH:-}"

if ! "$HORIZON_PYTHON" -c "import edge_fm" 2>/dev/null; then
    echo "ERROR: edge_fm not importable with $HORIZON_PYTHON"
    echo "  Run: bash scripts/build_horizon_quant.sh"
    exit 1
fi

# 默认跑 tests/；传具体路径时只跑该路径（如 tests/engine/）
"$HORIZON_PYTHON" -m pytest -s "${@:-tests/}"
