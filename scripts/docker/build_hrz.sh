#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"

EDGE_FM_PLATFORM="j6m"
EDGE_FM_DOCKERFILE="${EDGE_FM_DOCKERFILE:-${EDGE_FM_PROJECT_ROOT}/docker/hrz-j6m.dockerfile}"
EDGE_FM_DOCKER_IMAGE="${EDGE_FM_DOCKER_IMAGE:-edge-fm-hrz-j6m:latest}"
EDGE_FM_DOCKER_CONTEXT="${EDGE_FM_DOCKER_CONTEXT:-${EDGE_FM_PROJECT_ROOT}}"
EDGE_FM_BOOTSTRAP_PACKAGES="${EDGE_FM_BOOTSTRAP_PACKAGES:-0}"

edge_fm_docker_main "${1:-configure}"
