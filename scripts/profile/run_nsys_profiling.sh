#!/bin/bash
# Nsys profiling script for Edge-FM

set -e

# Activate conda environment
source ~/miniconda3/bin/activate horizon_quant

# Set Python path
export PYTHONPATH="/home/zhangzimo/Repos/private/edge-fm-x/build-3060/install/python:$PYTHONPATH"

# Create output directory
mkdir -p .tmp_codex/nsys

# Default parameters
MODEL_PATH="examples/qwen2.5-1.5b-instruct/qwen2.5-1.5b-instruct"
PREFILL_LEN=${1:-1024}
DECODE_LEN=${2:-32}
OUTPUT_NAME="edgefm_prefill_${PREFILL_LEN}_decode_${DECODE_LEN}"

echo "Running nsys profiling: prefill=$PREFILL_LEN, decode=$DECODE_LEN"

nsys profile -o .tmp_codex/nsys/${OUTPUT_NAME} \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python scripts/profile/profile_edgefm_generate_case.py \
    --model-path ${MODEL_PATH} \
    --prefill-len ${PREFILL_LEN} \
    --decode-len ${DECODE_LEN} \
    --profile-range \
    --use-cuda-graph

echo "Profiling complete: .tmp_codex/nsys/${OUTPUT_NAME}.nsys-rep"
