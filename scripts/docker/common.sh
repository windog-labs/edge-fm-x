#!/usr/bin/env bash

set -euo pipefail

EDGE_FM_DOCKER_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGE_FM_PROJECT_ROOT="$(cd "${EDGE_FM_DOCKER_COMMON_DIR}/../.." && pwd)"

edge_fm_split_words() {
    local raw="${1:-}"
    if [[ -z "${raw}" ]]; then
        return 0
    fi
    read -r -a EDGE_FM_WORDS <<<"${raw}"
    printf '%s\n' "${EDGE_FM_WORDS[@]}"
}

edge_fm_has_function() {
    declare -F "${1:-}" >/dev/null 2>&1
}

edge_fm_emit_function_output() {
    local function_name="${1:-}"
    if edge_fm_has_function "${function_name}"; then
        "${function_name}"
    fi
}

edge_fm_docker_usage() {
    cat <<'EOF'
Usage:
  bash scripts/docker/<build-script>.sh [image|configure|build|install|verify|all]

Actions:
  image      Build or refresh the tools image only.
  configure  Run `cmake --preset <platform>` in the container.
  build      Configure + build.
  install    Configure + build + install.
  verify     Configure + build + install + import edge_fm + pytest collect-only.
  all        Same as verify.

Important env vars:
  EDGE_FM_PLATFORM
  EDGE_FM_DOCKERFILE
  EDGE_FM_DOCKER_IMAGE
  EDGE_FM_DOCKER_PLATFORM
  EDGE_FM_DOCKER_CONTEXT
  EDGE_FM_DOCKER_REBUILD_IMAGE
  EDGE_FM_DOCKER_EXTRA_BUILD_ARGS
  EDGE_FM_DOCKER_EXTRA_RUN_ARGS
  EDGE_FM_DOCKER_RUN_AS_ROOT
  EDGE_FM_BUILD_JOBS
  EDGE_FM_BOOTSTRAP_PACKAGES
  EDGE_FM_PYTHON_EXECUTABLE
EOF
}

edge_fm_build_dir_for_platform() {
    case "${1:-}" in
        3060)
            printf '%s\n' "build-3060"
            ;;
        a800)
            printf '%s\n' "build-a800"
            ;;
        orin)
            printf '%s\n' "build-orin"
            ;;
        j6m)
            printf '%s\n' "build-j6m"
            ;;
        *)
            echo "ERROR: unsupported EDGE_FM_PLATFORM: ${1:-}" >&2
            exit 1
            ;;
    esac
}

edge_fm_build_image() {
    local docker_context="${EDGE_FM_DOCKER_CONTEXT:-${EDGE_FM_PROJECT_ROOT}}"
    local image_tag="${EDGE_FM_DOCKER_IMAGE:?EDGE_FM_DOCKER_IMAGE is required}"
    local dockerfile_path="${EDGE_FM_DOCKERFILE:?EDGE_FM_DOCKERFILE is required}"
    local docker_platform="${EDGE_FM_DOCKER_PLATFORM:-}"
    local rebuild_image="${EDGE_FM_DOCKER_REBUILD_IMAGE:-0}"
    local -a build_args=()

    if [[ ! -f "${dockerfile_path}" ]]; then
        echo "ERROR: Dockerfile not found: ${dockerfile_path}" >&2
        exit 1
    fi

    if [[ "${rebuild_image}" == "1" ]] || ! docker image inspect "${image_tag}" >/dev/null 2>&1; then
        while IFS= read -r token; do
            [[ -n "${token}" ]] && build_args+=("${token}")
        done < <(edge_fm_split_words "${EDGE_FM_DOCKER_EXTRA_BUILD_ARGS:-}")

        echo "[image] Building ${image_tag} from ${dockerfile_path}"
        local -a docker_cmd=(docker build)
        if [[ -n "${docker_platform}" ]]; then
            docker_cmd+=(--platform "${docker_platform}")
        fi
        docker_cmd+=(-f "${dockerfile_path}" -t "${image_tag}")
        docker_cmd+=("${build_args[@]}")
        docker_cmd+=("${docker_context}")
        DOCKER_BUILDKIT=1 "${docker_cmd[@]}"
    else
        echo "[image] Reusing local image ${image_tag}"
    fi
}

edge_fm_run_action() {
    local action="${1:?action is required}"
    local platform_name="${EDGE_FM_PLATFORM:?EDGE_FM_PLATFORM is required}"
    local image_tag="${EDGE_FM_DOCKER_IMAGE:?EDGE_FM_DOCKER_IMAGE is required}"
    local docker_platform="${EDGE_FM_DOCKER_PLATFORM:-}"
    local build_jobs="${EDGE_FM_BUILD_JOBS:-1}"
    local build_dir
    local python_executable="${EDGE_FM_PYTHON_EXECUTABLE:-/usr/bin/python3}"
    local bootstrap_packages="${EDGE_FM_BOOTSTRAP_PACKAGES:-0}"
    local docker_run_as_root="${EDGE_FM_DOCKER_RUN_AS_ROOT:-0}"
    local -a run_args=()
    local -a configure_args=()
    local pre_configure_script=""
    local verify_script=""

    build_dir="$(edge_fm_build_dir_for_platform "${platform_name}")"

    while IFS= read -r token; do
        [[ -n "${token}" ]] && run_args+=("${token}")
    done < <(edge_fm_split_words "${EDGE_FM_DOCKER_EXTRA_RUN_ARGS:-}")

    while IFS= read -r token; do
        [[ -n "${token}" ]] && configure_args+=("${token}")
    done < <(edge_fm_emit_function_output edge_fm_configure_args)

    pre_configure_script="$(edge_fm_emit_function_output edge_fm_pre_configure_commands)"
    verify_script="$(edge_fm_emit_function_output edge_fm_verify_commands)"

    local container_script
    container_script=$(
        cat <<EOF
set -euo pipefail

PROJECT_ROOT=/workspace/edge-fm
BUILD_DIR="\${PROJECT_ROOT}/${build_dir}"
PYTHON_EXECUTABLE="${python_executable}"
BOOTSTRAP_PACKAGES="${bootstrap_packages}"

edge_fm_reset_stale_cmake_cache() {
    local cache_file="\${BUILD_DIR}/CMakeCache.txt"
    if [[ ! -f "\${cache_file}" ]]; then
        return 0
    fi

    local cached_source_dir=""
    local cached_build_dir=""
    cached_source_dir="\$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "\${cache_file}" | tail -n 1)"
    cached_build_dir="\$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "\${cache_file}" | tail -n 1)"

    if [[ -n "\${cached_source_dir}" && "\${cached_source_dir}" != "\${PROJECT_ROOT}" ]]; then
        echo "[configure] Removing stale build cache with source dir \${cached_source_dir}"
        rm -rf "\${BUILD_DIR}"
        return 0
    fi

    if [[ -n "\${cached_build_dir}" && "\${cached_build_dir}" != "\${BUILD_DIR}" ]]; then
        echo "[configure] Removing stale build cache with build dir \${cached_build_dir}"
        rm -rf "\${BUILD_DIR}"
    fi
}

if [[ "\${BOOTSTRAP_PACKAGES}" == "1" ]]; then
    missing_packages=()
    if ! command -v cmake >/dev/null 2>&1; then
        missing_packages+=(cmake)
    fi
    if ! command -v ninja >/dev/null 2>&1; then
        missing_packages+=(ninja-build)
    fi
    if ! command -v make >/dev/null 2>&1; then
        missing_packages+=(make)
    fi
    if ! "\${PYTHON_EXECUTABLE}" -m pytest --version >/dev/null 2>&1; then
        missing_packages+=(python3-pytest)
    fi
    if (( \${#missing_packages[@]} > 0 )); then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "\${missing_packages[@]}"
    fi
fi

edge_fm_reset_stale_cmake_cache
EOF
    )

    if [[ -n "${pre_configure_script}" ]]; then
        container_script+=$'\n'
        container_script+="${pre_configure_script}"
    fi

    container_script+=$'\n'
    container_script+="cmake --preset \"${platform_name}\""
    if (( ${#configure_args[@]} > 0 )); then
        local arg
        for arg in "${configure_args[@]}"; do
            container_script+=" $(printf '%q' "${arg}")"
        done
    fi

    case "${action}" in
        configure)
            ;;
        build)
            container_script+=$'\n'
            container_script+="cmake --build --preset \"${platform_name}\" --parallel \"${build_jobs}\""
            ;;
        install)
            container_script+=$'\n'
            container_script+="cmake --build --preset \"${platform_name}\" --parallel \"${build_jobs}\""$'\n'
            container_script+="cmake --install \"\${BUILD_DIR}\""
            ;;
        verify|all)
            container_script+=$'\n'
            container_script+="cmake --build --preset \"${platform_name}\" --parallel \"${build_jobs}\""$'\n'
            container_script+="cmake --install \"\${BUILD_DIR}\""$'\n'
            container_script+="PYTHONPATH=\"\${BUILD_DIR}/install/python:\${PYTHONPATH:-}\" EDGE_FM_BUILD_DIR=\"\${BUILD_DIR}\" \"\${PYTHON_EXECUTABLE}\" - <<'PY'"$'\n'
            container_script+="import edge_fm"$'\n'
            container_script+="print(edge_fm.__file__)"$'\n'
            container_script+="PY"$'\n'
            if [[ -n "${verify_script}" ]]; then
                container_script+="${verify_script}"$'\n'
            fi
            container_script+="EDGE_FM_BUILD_DIR=\"\${BUILD_DIR}\" \"\${PYTHON_EXECUTABLE}\" -m pytest --collect-only tests/engine/test_qwen2_generate.py -q"
            ;;
        *)
            echo "ERROR: unsupported action: ${action}" >&2
            exit 1
            ;;
    esac

    local -a docker_run_cmd=(docker run --rm)
    if [[ -n "${docker_platform}" ]]; then
        docker_run_cmd+=(--platform "${docker_platform}")
    fi
    if [[ "${docker_run_as_root}" != "1" && "${bootstrap_packages}" != "1" ]]; then
        docker_run_cmd+=(--user "$(id -u):$(id -g)")
    fi
    docker_run_cmd+=(-v "${EDGE_FM_PROJECT_ROOT}:/workspace/edge-fm" -w /workspace/edge-fm)
    docker_run_cmd+=("${run_args[@]}")
    docker_run_cmd+=("${image_tag}" bash -lc "${container_script}")
    "${docker_run_cmd[@]}"
}

edge_fm_docker_main() {
    local action="${1:-all}"
    case "${action}" in
        image)
            edge_fm_build_image
            ;;
        configure|build|install|verify|all)
            edge_fm_build_image
            edge_fm_run_action "${action}"
            ;;
        -h|--help|help)
            edge_fm_docker_usage
            ;;
        *)
            echo "ERROR: unsupported action: ${action}" >&2
            edge_fm_docker_usage >&2
            exit 1
            ;;
    esac
}
