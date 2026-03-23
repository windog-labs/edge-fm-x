#!/bin/bash
# NCU 自动化性能分析 (edge-fm)
# 用法: ./auto_profile.sh <report_prefix> [pytest_args...]
# 示例: ./auto_profile.sh attn_profile tests/layers/test_attn.py -v -k "test_attn"
# 示例: ./auto_profile.sh linear_profile python tests/model_qwen2_one_layer.py

set -e
PREFIX=${1:?"Usage: $0 <report_prefix> [pytest_args...]"}
shift || true
REPORT_DIR="ncu_reports"
mkdir -p "$REPORT_DIR"

if [ $# -eq 0 ]; then
    set -- python -m pytest tests/layers/test_attn.py -v
elif [ "$1" != "python" ] && [ "$1" != "python3" ]; then
    set -- python -m pytest "$@"
fi

echo "🚀 NCU 采集: $PREFIX"
echo "命令: $*"
echo ""

ncu --set full -o "${REPORT_DIR}/${PREFIX}" --target-processes all --force-overwrite "$@" 2>&1 | tee "${REPORT_DIR}/${PREFIX}_log.txt"

echo ""
echo "📈 提取指标..."
ncu --import "${REPORT_DIR}/${PREFIX}.ncu-rep" --page raw --csv > "${REPORT_DIR}/${PREFIX}_raw.csv" 2>/dev/null || true
ncu --import "${REPORT_DIR}/${PREFIX}.ncu-rep" --print-summary per-kernel > "${REPORT_DIR}/${PREFIX}_summary.txt" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/ncu_analyzer.py" ]; then
    python3 "${SCRIPT_DIR}/ncu_analyzer.py" --import "${REPORT_DIR}/${PREFIX}.ncu-rep" -o "${REPORT_DIR}/${PREFIX}_analysis.md" 2>/dev/null || true
fi

echo "✅ 完成: ${REPORT_DIR}/${PREFIX}.ncu-rep"
