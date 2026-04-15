#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

EDGE_FM_PLATFORM="${EDGE_FM_PLATFORM:-a800}"
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

TRT_PACKAGE_DIR="${TRT_PACKAGE_DIR:-/usr/local/TensorRT}"
if [[ -d "${TRT_PACKAGE_DIR}" ]]; then
    EDGE_FM_DOCKER_EXTRA_RUN_ARGS="${EDGE_FM_DOCKER_EXTRA_RUN_ARGS:-} -v ${TRT_PACKAGE_DIR}:${TRT_PACKAGE_DIR}:ro"
fi

edge_fm_docker_main "${1:-all}"
