#!/bin/bash

edgefm_resolve_multiarch_triplet() {
    local machine="${1:-$(uname -m)}"
    case "$machine" in
        aarch64|arm64)
            echo "aarch64-linux-gnu"
            ;;
        x86_64|amd64)
            echo "x86_64-linux-gnu"
            ;;
        *)
            echo "${machine}-linux-gnu"
            ;;
    esac
}

edgefm_resolve_cuda_target_triple() {
    local machine="${1:-$(uname -m)}"
    case "$machine" in
        aarch64|arm64)
            echo "aarch64-linux"
            ;;
        x86_64|amd64)
            echo "x86_64-linux"
            ;;
        *)
            return 1
            ;;
    esac
}

edgefm_resolve_cuda_home() {
    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        echo "${CUDA_HOME}"
        return 0
    fi

    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path
        nvcc_path="$(command -v nvcc)"
        dirname "$(dirname "$(readlink -f "${nvcc_path}")")"
        return 0
    fi

    for candidate in /usr/local/cuda /usr/local/cuda-*; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

edgefm_resolve_cuda_version() {
    local cuda_home="$1"
    local nvcc_path="${cuda_home}/bin/nvcc"
    if [[ -x "${nvcc_path}" ]]; then
        "${nvcc_path}" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1
        return 0
    fi

    basename "${cuda_home}" | sed -n 's/^cuda-\([0-9][0-9]*\.[0-9][0-9]*\)$/\1/p'
}

edgefm_resolve_cuda_library_dir() {
    local cuda_home="$1"
    if [[ -d "${cuda_home}/lib64" ]]; then
        echo "${cuda_home}/lib64"
        return 0
    fi

    local target_triple
    target_triple="$(edgefm_resolve_cuda_target_triple)" || return 1
    if [[ -d "${cuda_home}/targets/${target_triple}/lib" ]]; then
        echo "${cuda_home}/targets/${target_triple}/lib"
        return 0
    fi

    return 1
}

edgefm_default_trt_cuda_architectures() {
    local machine="${1:-$(uname -m)}"
    case "$machine" in
        aarch64|arm64)
            echo "87"
            ;;
        x86_64|amd64)
            echo "80;86;89"
            ;;
        *)
            return 1
            ;;
    esac
}

edgefm_resolve_build_dir() {
    local project_root="$1"

    if [[ -n "${EDGE_FM_BUILD_DIR:-}" && -d "${EDGE_FM_BUILD_DIR}" ]]; then
        echo "${EDGE_FM_BUILD_DIR}"
        return 0
    fi

    local candidate
    for candidate in "${project_root}/build" "${project_root}"/build-*; do
        [[ -d "${candidate}" ]] || continue
        if [[ -d "${candidate}/python" || -d "${candidate}/install/python" || -d "${candidate}/lib" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

edgefm_resolve_tensorrt_package_dir() {
    local project_root="$1"
    local workspace_root="$2"

    if [[ -n "${TRT_PACKAGE_DIR:-}" && -f "${TRT_PACKAGE_DIR}/include/NvInfer.h" ]]; then
        echo "${TRT_PACKAGE_DIR}"
        return 0
    fi

    local candidate
    for candidate in /usr/local/TensorRT /usr/local/TensorRT-*; do
        if [[ -f "${candidate}/include/NvInfer.h" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    local multiarch
    multiarch="$(edgefm_resolve_multiarch_triplet)"

    local include_dir=""
    local lib_dir=""

    for candidate in "/usr/include/${multiarch}" "/usr/include"; do
        if [[ -f "${candidate}/NvInfer.h" ]]; then
            include_dir="${candidate}"
            break
        fi
    done

    for candidate in "/usr/lib/${multiarch}" "/usr/lib64" "/usr/lib"; do
        if [[ -f "${candidate}/libnvinfer.so" ]]; then
            lib_dir="${candidate}"
            break
        fi
    done

    if [[ -n "${include_dir}" && -n "${lib_dir}" ]]; then
        local overlay_dir="${workspace_root}/_deps/tensorrt-package-${multiarch}"
        mkdir -p "${overlay_dir}"
        ln -sfn "${include_dir}" "${overlay_dir}/include"
        ln -sfn "${lib_dir}" "${overlay_dir}/lib"
        echo "${overlay_dir}"
        return 0
    fi

    return 1
}
