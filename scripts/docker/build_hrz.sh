#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

EDGE_FM_PLATFORM="j6m"
EDGE_FM_DOCKERFILE="${EDGE_FM_DOCKERFILE:-${EDGE_FM_PROJECT_ROOT}/docker/hrz-j6m.dockerfile}"
EDGE_FM_DOCKER_IMAGE="${EDGE_FM_DOCKER_IMAGE:-openexplorer/ai_toolchain_ubuntu_22_j6_gpu:v3.5.0}"
EDGE_FM_DOCKER_CONTEXT="${EDGE_FM_DOCKER_CONTEXT:-${EDGE_FM_PROJECT_ROOT}}"
EDGE_FM_BOOTSTRAP_PACKAGES="${EDGE_FM_BOOTSTRAP_PACKAGES:-1}"
EDGE_FM_DEFAULT_HOST_HORIZON_DEPS_ROOT="${HOME}/Packages/horizon_j6_open_explorer_v3.5.0-py310_20250927/samples/ucp_tutorial/deps_aarch64"
EDGE_FM_HOST_HORIZON_DEPS_ROOT="${EDGE_FM_HOST_HORIZON_DEPS_ROOT:-}"
if [[ -z "${EDGE_FM_HOST_HORIZON_DEPS_ROOT}" && -d "${EDGE_FM_DEFAULT_HOST_HORIZON_DEPS_ROOT}/ucp/include" ]]; then
    EDGE_FM_HOST_HORIZON_DEPS_ROOT="${EDGE_FM_DEFAULT_HOST_HORIZON_DEPS_ROOT}"
fi
EDGE_FM_CONTAINER_HORIZON_DEPS_ROOT="${EDGE_FM_CONTAINER_HORIZON_DEPS_ROOT:-/opt/horizon_deps_aarch64}"
if [[ -z "${EDGE_FM_ENABLE_HORIZON_RUNTIME:-}" && -n "${EDGE_FM_HOST_HORIZON_DEPS_ROOT}" ]]; then
    EDGE_FM_ENABLE_HORIZON_RUNTIME="ON"
fi

edge_fm_run_args() {
    if [[ -n "${EDGE_FM_HOST_HORIZON_DEPS_ROOT}" ]]; then
        if [[ ! -d "${EDGE_FM_HOST_HORIZON_DEPS_ROOT}/ucp/include" ]]; then
            echo "ERROR: EDGE_FM_HOST_HORIZON_DEPS_ROOT does not contain ucp/include: ${EDGE_FM_HOST_HORIZON_DEPS_ROOT}" >&2
            exit 1
        fi
        printf '%s\n' "-v"
        printf '%s\n' "${EDGE_FM_HOST_HORIZON_DEPS_ROOT}:${EDGE_FM_CONTAINER_HORIZON_DEPS_ROOT}:ro"
    fi
}

edge_fm_configure_args() {
    local toolchain_root="${EDGE_FM_J6M_TOOLCHAIN_ROOT:-/arm-gnu-toolchain-12.2.rel1-x86_64-aarch64-none-linux-gnu}"
    local target_triple="${EDGE_FM_J6M_TARGET_TRIPLE:-aarch64-none-linux-gnu}"
    local toolchain_bin="${toolchain_root}/bin"
    local enable_horizon_runtime="${EDGE_FM_ENABLE_HORIZON_RUNTIME:-OFF}"

    printf '%s\n' "-DEDGE_FM_J6M_TOOLCHAIN_ROOT=${toolchain_root}"
    printf '%s\n' "-DEDGE_FM_J6M_TARGET_TRIPLE=${target_triple}"
    printf '%s\n' "-DCMAKE_SYSTEM_NAME=Linux"
    printf '%s\n' "-DCMAKE_SYSTEM_PROCESSOR=aarch64"
    printf '%s\n' "-DCMAKE_CXX_COMPILER=${toolchain_bin}/${target_triple}-g++"
    printf '%s\n' "-DCMAKE_AR=${toolchain_bin}/${target_triple}-ar"
    printf '%s\n' "-DCMAKE_RANLIB=${toolchain_bin}/${target_triple}-ranlib"
    printf '%s\n' "-DCMAKE_STRIP=${toolchain_bin}/${target_triple}-strip"
    printf '%s\n' "-DBUILD_PYTHON=OFF"
    printf '%s\n' "-DENABLE_HORIZON_RUNTIME=${enable_horizon_runtime}"
    if [[ "${enable_horizon_runtime}" == "ON" && -n "${EDGE_FM_HOST_HORIZON_DEPS_ROOT}" ]]; then
        printf '%s\n' "-DHORIZON_DEPS_ROOT=${EDGE_FM_CONTAINER_HORIZON_DEPS_ROOT}"
    fi
}

edge_fm_docker_main "${1:-configure}"
