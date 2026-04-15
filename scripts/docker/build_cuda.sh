#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

EDGE_FM_PLATFORM="${EDGE_FM_PLATFORM:-3060}"
case "${EDGE_FM_PLATFORM}" in
    3060|a800)
        ;;
    *)
        echo "ERROR: EDGE_FM_PLATFORM must be 3060 or a800 for build_cuda.sh" >&2
        exit 1
        ;;
esac

EDGE_FM_DOCKERFILE="${EDGE_FM_DOCKERFILE:-${EDGE_FM_PROJECT_ROOT}/docker/cuda12.6.3_cudnn_trt10.15.dockerfile}"
EDGE_FM_DOCKER_IMAGE="${EDGE_FM_DOCKER_IMAGE:-edge-fm-cuda12.6.3-trt10.15:latest}"
EDGE_FM_DOCKER_CONTEXT="${EDGE_FM_DOCKER_CONTEXT:-${EDGE_FM_PROJECT_ROOT}}"
EDGE_FM_BOOTSTRAP_PACKAGES="${EDGE_FM_BOOTSTRAP_PACKAGES:-0}"
EDGE_FM_BUILD_TRT_EDGELLM_PYBIND="${EDGE_FM_BUILD_TRT_EDGELLM_PYBIND:-1}"
EDGE_FM_HOST_TRT_DIR="${EDGE_FM_HOST_TRT_DIR:-/usr/local/TensorRT-10.15.1.29}"
EDGE_FM_CUDA_TRT_MIN_VERSION_MAJOR=10
EDGE_FM_CUDA_TRT_MIN_VERSION_MINOR=15
EDGE_FM_CUDA_TRT_MIN_VERSION="${EDGE_FM_CUDA_TRT_MIN_VERSION_MAJOR}.${EDGE_FM_CUDA_TRT_MIN_VERSION_MINOR}"

EDGE_FM_HOST_TRT_VERSION_HEADER=""
EDGE_FM_HOST_TRT_VERSION_MAJOR=""
EDGE_FM_HOST_TRT_VERSION_MINOR=""
EDGE_FM_HOST_TRT_VERSION_PATCH=""
EDGE_FM_HOST_TRT_VERSION_BUILD=""
EDGE_FM_HOST_TRT_VERSION_STRING=""

edge_fm_read_trt_version_macro() {
    local version_header="${1:?version header is required}"
    local macro_regex="${2:?macro regex is required}"

    sed -n -E "s/^#define (${macro_regex})[[:space:]]+([0-9]+).*/\\2/p" "${version_header}" | head -n 1
}

edge_fm_detect_host_trt_version() {
    local version_header=""
    local version_header_candidates=(
        "${EDGE_FM_HOST_TRT_DIR}/include/NvInferVersion.h"
        "${EDGE_FM_HOST_TRT_DIR}/NvInferVersion.h"
    )

    EDGE_FM_HOST_TRT_VERSION_HEADER=""
    EDGE_FM_HOST_TRT_VERSION_MAJOR=""
    EDGE_FM_HOST_TRT_VERSION_MINOR=""
    EDGE_FM_HOST_TRT_VERSION_PATCH=""
    EDGE_FM_HOST_TRT_VERSION_BUILD=""
    EDGE_FM_HOST_TRT_VERSION_STRING=""

    for version_header in "${version_header_candidates[@]}"; do
        if [[ ! -f "${version_header}" ]]; then
            continue
        fi

        EDGE_FM_HOST_TRT_VERSION_MAJOR="$(edge_fm_read_trt_version_macro "${version_header}" 'NV_TENSORRT_MAJOR|TRT_MAJOR_ENTERPRISE')"
        EDGE_FM_HOST_TRT_VERSION_MINOR="$(edge_fm_read_trt_version_macro "${version_header}" 'NV_TENSORRT_MINOR|TRT_MINOR_ENTERPRISE')"
        EDGE_FM_HOST_TRT_VERSION_PATCH="$(edge_fm_read_trt_version_macro "${version_header}" 'NV_TENSORRT_PATCH|TRT_PATCH_ENTERPRISE')"
        EDGE_FM_HOST_TRT_VERSION_BUILD="$(edge_fm_read_trt_version_macro "${version_header}" 'NV_TENSORRT_BUILD|TRT_BUILD_ENTERPRISE')"

        if [[ -n "${EDGE_FM_HOST_TRT_VERSION_MAJOR}" && -n "${EDGE_FM_HOST_TRT_VERSION_MINOR}" && -n "${EDGE_FM_HOST_TRT_VERSION_PATCH}" && -n "${EDGE_FM_HOST_TRT_VERSION_BUILD}" ]]; then
            EDGE_FM_HOST_TRT_VERSION_HEADER="${version_header}"
            EDGE_FM_HOST_TRT_VERSION_STRING="${EDGE_FM_HOST_TRT_VERSION_MAJOR}.${EDGE_FM_HOST_TRT_VERSION_MINOR}.${EDGE_FM_HOST_TRT_VERSION_PATCH}.${EDGE_FM_HOST_TRT_VERSION_BUILD}"
            return 0
        fi
    done

    return 1
}

edge_fm_require_host_trt_version() {
    if ! edge_fm_detect_host_trt_version; then
        echo "ERROR: failed to determine TensorRT version from ${EDGE_FM_HOST_TRT_DIR}." >&2
        echo "       Expected version header under ${EDGE_FM_HOST_TRT_DIR}/include/NvInferVersion.h." >&2
        echo "       TensorRT-Edge-LLM on CUDA/x86 requires TensorRT >= ${EDGE_FM_CUDA_TRT_MIN_VERSION}." >&2
        exit 1
    fi

    if (( EDGE_FM_HOST_TRT_VERSION_MAJOR < EDGE_FM_CUDA_TRT_MIN_VERSION_MAJOR )) || \
       (( EDGE_FM_HOST_TRT_VERSION_MAJOR == EDGE_FM_CUDA_TRT_MIN_VERSION_MAJOR && EDGE_FM_HOST_TRT_VERSION_MINOR < EDGE_FM_CUDA_TRT_MIN_VERSION_MINOR )); then
        echo "ERROR: TensorRT ${EDGE_FM_HOST_TRT_VERSION_STRING} found in ${EDGE_FM_HOST_TRT_DIR}," >&2
        echo "       but CUDA/x86 build_cuda.sh requires TensorRT >= ${EDGE_FM_CUDA_TRT_MIN_VERSION}." >&2
        echo "       Update EDGE_FM_HOST_TRT_DIR to a newer TensorRT package and retry." >&2
        exit 1
    fi
}

edge_fm_print_host_trt_info() {
    echo "[cuda] Host TensorRT package: ${EDGE_FM_HOST_TRT_DIR}"
    echo "[cuda] Host TensorRT version: ${EDGE_FM_HOST_TRT_VERSION_STRING} (from ${EDGE_FM_HOST_TRT_VERSION_HEADER})"
    echo "[cuda] Required minimum version for CUDA/x86 TRT-Edge-LLM: ${EDGE_FM_CUDA_TRT_MIN_VERSION}"
}

edge_fm_validate_host_trt_dir() {
    if [[ ! -d "${EDGE_FM_HOST_TRT_DIR}" ]]; then
        echo "ERROR: EDGE_FM_HOST_TRT_DIR does not exist: ${EDGE_FM_HOST_TRT_DIR}" >&2
        exit 1
    fi
    if [[ ! -f "${EDGE_FM_HOST_TRT_DIR}/include/NvInfer.h" ]]; then
        echo "ERROR: TensorRT header missing from ${EDGE_FM_HOST_TRT_DIR}/include/NvInfer.h" >&2
        exit 1
    fi
    if ! compgen -G "${EDGE_FM_HOST_TRT_DIR}/lib/libnvinfer.so*" >/dev/null; then
        echo "ERROR: TensorRT library missing from ${EDGE_FM_HOST_TRT_DIR}/lib/libnvinfer.so*" >&2
        exit 1
    fi
    if [[ ! -f "${EDGE_FM_HOST_TRT_DIR}/include/NvOnnxParser.h" ]]; then
        echo "ERROR: TensorRT ONNX parser header missing from ${EDGE_FM_HOST_TRT_DIR}/include/NvOnnxParser.h" >&2
        exit 1
    fi
    if ! compgen -G "${EDGE_FM_HOST_TRT_DIR}/lib/libnvonnxparser.so*" >/dev/null; then
        echo "ERROR: TensorRT ONNX parser library missing from ${EDGE_FM_HOST_TRT_DIR}/lib/libnvonnxparser.so*" >&2
        exit 1
    fi

    edge_fm_require_host_trt_version
}

edge_fm_cuda_arch() {
    case "${EDGE_FM_PLATFORM}" in
        3060)
            printf '%s\n' "86"
            ;;
        a800)
            printf '%s\n' "80"
            ;;
        *)
            echo "ERROR: unsupported CUDA platform: ${EDGE_FM_PLATFORM}" >&2
            exit 1
            ;;
    esac
}

edge_fm_image_build_args() {
    edge_fm_validate_host_trt_dir
    printf '%s\n' \
        "--build-context" \
        "host_trt=${EDGE_FM_HOST_TRT_DIR}"
}

edge_fm_run_args() {
    printf '%s\n' \
        "--gpus" \
        "all"
}

edge_fm_configure_args() {
    if [[ "${EDGE_FM_BUILD_TRT_EDGELLM_PYBIND}" != "1" ]]; then
        return 0
    fi

    local build_dir
    build_dir="$(edge_fm_build_dir_for_platform "${EDGE_FM_PLATFORM}")"
    printf '%s\n' \
        "-DBUILD_TRT_EDGELLM_PYBIND=ON" \
        "-DTRT_PACKAGE_DIR=/usr/local/TensorRT" \
        "-DTRT_EDGELLM_BUILD_DIR=/workspace/edge-fm/${build_dir}/trt-edgellm"
}

edge_fm_pre_configure_commands() {
    if [[ "${EDGE_FM_BUILD_TRT_EDGELLM_PYBIND}" != "1" ]]; then
        return 0
    fi

    local cuda_arch
    local trt_build_jobs
    cuda_arch="$(edge_fm_cuda_arch)"
    trt_build_jobs="${EDGE_FM_TRT_BUILD_JOBS:-${EDGE_FM_BUILD_JOBS:-1}}"

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
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT \
    -DONNX_PARSER_INCLUDE_DIR=/usr/local/TensorRT/include \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=${cuda_arch} \
    -DCUDA_VERSION=12.6 \
    -DCUDA_DIR=/usr/local/cuda/targets/x86_64-linux
if [[ ! -f "\${TRT_CORE_LIB}" || ! -f "\${TRT_TOKENIZER_LIB}" || ! -f "\${TRT_UTILS_LIB}" ]]; then
    cmake --build "\${TRT_EDGELLM_BUILD_DIR}" --parallel "${trt_build_jobs}" --target edgellmCore edgellmTokenizer exampleUtils
fi
EOF
}

edge_fm_verify_commands() {
    local platform_name="${EDGE_FM_PLATFORM}"
    local correctness_expr='not benchmark'
    local bench_prefill_list='512,1024'
    if [[ "${platform_name}" == "3060" ]]; then
        correctness_expr='not benchmark and not vl'
        bench_prefill_list='512'
    fi

    cat <<EOF
PY_RUNTIME_DIR="\${BUILD_DIR}/.python-runtime"
mkdir -p "\${PY_RUNTIME_DIR}"
"\${PYTHON_EXECUTABLE}" -m pip install --disable-pip-version-check --target "\${PY_RUNTIME_DIR}" --upgrade 'pytest>=8.3,<9'
export PYTHONPATH="\${PY_RUNTIME_DIR}:\${BUILD_DIR}/install/python:\${PYTHONPATH:-}"

EDGE_FM_BUILD_DIR="\${BUILD_DIR}" "\${PYTHON_EXECUTABLE}" - <<'PY'
import edge_fm_trt
print(edge_fm_trt.__file__)
PY
TRT_BUILD_DIR="\${BUILD_DIR}/trt-edgellm"
export TRT_PACKAGE_DIR=/usr/local/TensorRT
export EDGE_FM_BUILD_DIR="\${BUILD_DIR}"
export EDGE_FM_DEVICE_ID="\${EDGE_FM_DEVICE_ID:-0}"
export EDGE_FM_PLATFORM="${platform_name}"
export TRT_EDGELLM_PLUGIN_PATH="\${TRT_BUILD_DIR}/libNvInfer_edgellm_plugin.so"
export EDGELLM_PLUGIN_PATH="\${TRT_EDGELLM_PLUGIN_PATH}"

"\${PYTHON_EXECUTABLE}" -m pytest -s tests/engine/test_qwen2_generate.py -k "${correctness_expr}"

env EDGE_FM_TRT_MODEL_SIZE=1.5b EDGE_FM_TRT_MAX_INPUT_LEN=2048 \
    bash tests/scripts/setup_trt_edgellm_benchmark.sh

env EDGE_FM_BENCH_LLM_MODELS=0.5b,1.5b EDGE_FM_BENCH_PREFILL_LIST=${bench_prefill_list} EDGE_FM_BENCH_DECODE_LIST=32 \
    "\${PYTHON_EXECUTABLE}" -m pytest -s tests/engine/test_qwen2_generate.py -q -k test_benchmark_llm

env EDGE_FM_BENCH_LLM_MODELS=1.5b EDGE_FM_BENCH_PREFILL_LIST=${bench_prefill_list} EDGE_FM_BENCH_DECODE_LIST=32 \
    "\${PYTHON_EXECUTABLE}" -m pytest -s tests/engine/test_qwen2_generate.py -q -k test_benchmark_trt_edgellm
EOF
}

edge_fm_action="${1:-all}"
case "${edge_fm_action}" in
    -h|--help|help)
        ;;
    *)
        edge_fm_validate_host_trt_dir
        edge_fm_print_host_trt_info
        ;;
esac

edge_fm_docker_main "${edge_fm_action}"
