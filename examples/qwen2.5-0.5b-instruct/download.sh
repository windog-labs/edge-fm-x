#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

save_dir=./qwen2.5-0.5b-instruct
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir $save_dir

echo "模型已保存至: ${save_dir}"
