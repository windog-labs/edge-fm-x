#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

EDGE_FM_PLATFORM="orin"
EDGE_FM_DOCKERFILE="${EDGE_FM_DOCKERFILE:-${EDGE_FM_PROJECT_ROOT}/docker/orin-l4t-jetpack-r36.4.0.dockerfile}"
EDGE_FM_DOCKER_IMAGE="${EDGE_FM_DOCKER_IMAGE:-edge-fm-orin:r36.4.0-tools}"
EDGE_FM_DOCKER_PLATFORM="${EDGE_FM_DOCKER_PLATFORM:-linux/arm64}"
EDGE_FM_DOCKER_CONTEXT="${EDGE_FM_DOCKER_CONTEXT:-${EDGE_FM_PROJECT_ROOT}}"
EDGE_FM_BOOTSTRAP_PACKAGES="${EDGE_FM_BOOTSTRAP_PACKAGES:-0}"
EDGE_FM_BUILD_TRT_EDGELLM_PYBIND="${EDGE_FM_BUILD_TRT_EDGELLM_PYBIND:-1}"

edge_fm_configure_args() {
    if [[ "${EDGE_FM_BUILD_TRT_EDGELLM_PYBIND}" == "1" ]]; then
        printf '%s\n' \
            "-DBUILD_TRT_EDGELLM_PYBIND=ON" \
            "-DTRT_PACKAGE_DIR=/usr" \
            "-DTRT_EDGELLM_BUILD_DIR=/workspace/edge-fm/build-orin/trt-edgellm"
    fi
}

edge_fm_pre_configure_commands() {
    if [[ "${EDGE_FM_BUILD_TRT_EDGELLM_PYBIND}" != "1" ]]; then
        return 0
    fi

    local trt_build_jobs="${EDGE_FM_TRT_BUILD_JOBS:-${EDGE_FM_BUILD_JOBS:-1}}"

    cat <<EOF
TRT_EDGELLM_ROOT="\${PROJECT_ROOT}/third_party/TensorRT-Edge-LLM"
TRT_EDGELLM_BUILD_DIR="\${BUILD_DIR}/trt-edgellm"
TRT_NLOHMANN_DIR="\${TRT_EDGELLM_ROOT}/3rdParty/nlohmannJson/include/nlohmann"
TRT_CORE_LIB="\${TRT_EDGELLM_BUILD_DIR}/cpp/libedgellmCore.a"
TRT_TOKENIZER_LIB="\${TRT_EDGELLM_BUILD_DIR}/cpp/libedgellmTokenizer.a"
TRT_UTILS_LIB="\${TRT_EDGELLM_BUILD_DIR}/examples/utils/libexampleUtils.a"

mkdir -p "\${TRT_EDGELLM_ROOT}/3rdParty/nlohmannJson/include"
if [[ -L "\${TRT_NLOHMANN_DIR}" ]] || [[ -e "\${TRT_NLOHMANN_DIR}" ]]; then
    rm -rf "\${TRT_NLOHMANN_DIR}"
fi
ln -s "../../../../json/include/nlohmann" "\${TRT_NLOHMANN_DIR}"

if [[ -f "\${TRT_EDGELLM_BUILD_DIR}/CMakeCache.txt" ]]; then
    TRT_CACHED_SOURCE_DIR="\$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "\${TRT_EDGELLM_BUILD_DIR}/CMakeCache.txt" | tail -n 1)"
    if [[ -n "\${TRT_CACHED_SOURCE_DIR}" && "\${TRT_CACHED_SOURCE_DIR}" != "\${TRT_EDGELLM_ROOT}" ]]; then
        rm -rf "\${TRT_EDGELLM_BUILD_DIR}"
    fi
fi

cmake -S "\${TRT_EDGELLM_ROOT}" -B "\${TRT_EDGELLM_BUILD_DIR}" -G Ninja \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCUDA_VERSION=12.6 \
    -DCUDA_DIR=/usr/local/cuda/targets/aarch64-linux
if [[ ! -f "\${TRT_CORE_LIB}" || ! -f "\${TRT_TOKENIZER_LIB}" || ! -f "\${TRT_UTILS_LIB}" ]]; then
    cmake --build "\${TRT_EDGELLM_BUILD_DIR}" --parallel "${trt_build_jobs}" --target edgellmCore edgellmTokenizer exampleUtils
fi
EOF
}

edge_fm_verify_commands() {
    if [[ "${EDGE_FM_BUILD_TRT_EDGELLM_PYBIND}" != "1" ]]; then
        return 0
    fi

    cat <<'EOF'
PYTHONPATH="${BUILD_DIR}/install/python:${PYTHONPATH:-}" EDGE_FM_BUILD_DIR="${BUILD_DIR}" "${PYTHON_EXECUTABLE}" - <<'PY'
import edge_fm_trt
print(edge_fm_trt.__file__)
PY
EOF
}

edge_fm_docker_main "${1:-all}"
