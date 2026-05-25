#!/bin/bash

set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

save_dir=./qwen3.5-2b
repo_id=Qwen/Qwen3.5-2B

download_with_wget() {
    local endpoint=${HF_DIRECT_ENDPOINT:-${HF_ENDPOINT}}
    local files=(
        ".gitattributes"
        "LICENSE"
        "README.md"
        "chat_template.jinja"
        "config.json"
        "merges.txt"
        "model.safetensors-00001-of-00001.safetensors"
        "model.safetensors.index.json"
        "preprocessor_config.json"
        "tokenizer.json"
        "tokenizer_config.json"
        "video_preprocessor_config.json"
        "vocab.json"
    )

    mkdir -p "${save_dir}"
    for file in "${files[@]}"; do
        echo "下载 ${file}"
        wget -c --tries=5 --timeout=30 \
            -O "${save_dir}/${file}" \
            "${endpoint%/}/${repo_id}/resolve/main/${file}"
    done
}

if hf download "${repo_id}" --local-dir "${save_dir}"; then
    :
else
    status=$?
    if [[ ${status} -eq 130 ]]; then
        exit "${status}"
    fi
    echo "hf download 失败，使用 wget 直链 fallback..."
    download_with_wget
fi

echo "模型已保存至: ${save_dir}"
