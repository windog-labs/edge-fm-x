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
        cat <<'EOF'
EDGE_FM_BUILD_DIR="${BUILD_DIR}" "${PYTHON_EXECUTABLE}" -m pytest --collect-only tests/engine/test_qwen2_generate.py -q
EOF
        return 0
    fi

    cat <<'EOF'
PY_RUNTIME_DIR="${BUILD_DIR}/.python-runtime"
ORIN_COMPAT_DIR="/tmp/edge-fm-orin-runtime-$(id -u)"
mkdir -p "${PY_RUNTIME_DIR}"
"${PYTHON_EXECUTABLE}" -m pip install --disable-pip-version-check --target "${PY_RUNTIME_DIR}" --upgrade 'pytest>=8.3,<9'
mkdir -p "${ORIN_COMPAT_DIR}"

if [[ ! -e /usr/lib/aarch64-linux-gnu/libnvos.so && ! -e /usr/lib/aarch64-linux-gnu/nvidia/libnvos.so ]]; then
    cat > "${ORIN_COMPAT_DIR}/libnvos_stub.c" <<'C_EOF'
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void* NvOsAlloc(size_t size) { return calloc(1, size ? size : 1); }
uintptr_t NvOsClosedir(void) { return 0; }
uintptr_t NvOsFcloseEx(void) { return 0; }
uintptr_t NvOsFgetc(void) { return 0; }
uintptr_t NvOsFopen(void) { return 0; }
uintptr_t NvOsFread(void) { return 0; }
void NvOsFree(void* p) { free(p); }
uintptr_t NvOsFremove(void) { return 0; }
uintptr_t NvOsFseek(void) { return 0; }
uintptr_t NvOsFstat(void) { return 0; }
uintptr_t NvOsFwrite(void) { return 0; }
uintptr_t NvOsGetOsInformation(void) { return 0; }
uintptr_t NvOsGetTimeMS(void) { return 0; }
void* NvOsMemset(void* dst, int c, size_t n) { return memset(dst, c, n); }
uintptr_t NvOsMkdir(void) { return 0; }
uintptr_t NvOsOpendir(void) { return 0; }
uintptr_t NvOsReaddir(void) { return 0; }
uintptr_t NvOsSleepMS(void) { return 0; }
uintptr_t NvOsStat(void) { return 0; }
uintptr_t NvOsThreadCreate(void) { return 0; }
uintptr_t NvOsThreadJoinEx(void) { return 0; }
C_EOF
    gcc -shared -fPIC "${ORIN_COMPAT_DIR}/libnvos_stub.c" \
        -Wl,-soname,libnvos.so \
        -o "${ORIN_COMPAT_DIR}/libnvos.so"
fi

if [[ ! -e /usr/lib/aarch64-linux-gnu/libnvdla_runtime.so && ! -e /usr/lib/aarch64-linux-gnu/nvidia/libnvdla_runtime.so ]]; then
    cat > "${ORIN_COMPAT_DIR}/libnvdla_runtime_stub.cpp" <<'CPP_EOF'
struct NvDlaFenceRec;
struct NvDlaSemaphoreRec;
struct NvDlaRuntimeTaskStatisticsDesc;
enum NvDlaSyncEventType : int {};
enum NvDlaAccessType : int {};
struct NvDlaMemDescRec { unsigned long long data[8]; };
namespace nvdla {
struct ISync {};
struct IRuntime {};
struct CsvStatisticsContainer {};
}

extern "C" nvdla::IRuntime* createRuntime_stub() asm("_ZN5nvdla13createRuntimeEv");
extern "C" nvdla::IRuntime* createRuntime_stub() { return nullptr; }

extern "C" void destroyRuntime_stub(nvdla::IRuntime*) asm("_ZN5nvdla14destroyRuntimeEPNS_8IRuntimeE");
extern "C" void destroyRuntime_stub(nvdla::IRuntime*) {}

extern "C" nvdla::ISync* createSyncSemaphore_stub(const NvDlaSemaphoreRec*)
    asm("_ZN5nvdla19createSyncSemaphoreEPK17NvDlaSemaphoreRec");
extern "C" nvdla::ISync* createSyncSemaphore_stub(const NvDlaSemaphoreRec*) { return nullptr; }

extern "C" nvdla::ISync* createSyncSyncpoint_stub(const NvDlaFenceRec*)
    asm("_ZN5nvdla19createSyncSyncpointEPK13NvDlaFenceRec");
extern "C" nvdla::ISync* createSyncSyncpoint_stub(const NvDlaFenceRec*) { return nullptr; }

extern "C" nvdla::ISync* createSyncStrideSemaphore_stub(const NvDlaSemaphoreRec*, unsigned)
    asm("_ZN5nvdla25createSyncStrideSemaphoreEPK17NvDlaSemaphoreRecj");
extern "C" nvdla::ISync* createSyncStrideSemaphore_stub(const NvDlaSemaphoreRec*, unsigned) { return nullptr; }

extern "C" void destroySync_stub(nvdla::ISync*) asm("_ZN5nvdla11destroySyncEPNS_5ISyncE");
extern "C" void destroySync_stub(nvdla::ISync*) {}

extern "C" int bindSubmitEvent_stub(void*, int, NvDlaSyncEventType, nvdla::ISync*, int*)
    asm("_ZN5nvdla8IRuntime15bindSubmitEventEi18NvDlaSyncEventTypePNS_5ISyncEPi");
extern "C" int bindSubmitEvent_stub(void*, int, NvDlaSyncEventType, nvdla::ISync*, int*) { return -1; }

extern "C" int registerTaskStatistics_stub(void*, NvDlaMemDescRec, NvDlaAccessType)
    asm("_ZN5nvdla8IRuntime22registerTaskStatisticsE15NvDlaMemDescRec15NvDlaAccessType");
extern "C" int registerTaskStatistics_stub(void*, NvDlaMemDescRec, NvDlaAccessType) { return -1; }

extern "C" int translateRawStatsToCsv_stub(void*, const char*, float, nvdla::CsvStatisticsContainer&)
    asm("_ZN5nvdla8IRuntime22translateRawStatsToCsvEPKcfRNS_22CsvStatisticsContainerE");
extern "C" int translateRawStatsToCsv_stub(void*, const char*, float, nvdla::CsvStatisticsContainer&) { return -1; }

extern "C" int appendDiagnosticLoadable_stub(void*, int*)
    asm("_ZN5nvdla8IRuntime24appendDiagnosticLoadableEPi");
extern "C" int appendDiagnosticLoadable_stub(void*, int*) { return -1; }

extern "C" int bindOutputTaskStatistics_stub(void*, int, NvDlaMemDescRec)
    asm("_ZN5nvdla8IRuntime24bindOutputTaskStatisticsEi15NvDlaMemDescRec");
extern "C" int bindOutputTaskStatistics_stub(void*, int, NvDlaMemDescRec) { return -1; }

extern "C" int unregisterTaskStatistics_stub(void*, NvDlaMemDescRec)
    asm("_ZN5nvdla8IRuntime24unregisterTaskStatisticsE15NvDlaMemDescRec");
extern "C" int unregisterTaskStatistics_stub(void*, NvDlaMemDescRec) { return -1; }

extern "C" int getNumOutputTaskStatistics_stub(void*, int*)
    asm("_ZN5nvdla8IRuntime26getNumOutputTaskStatisticsEPi");
extern "C" int getNumOutputTaskStatistics_stub(void*, int*) { return -1; }

extern "C" int getOutputTaskStatisticsDesc_stub(void*, int, NvDlaRuntimeTaskStatisticsDesc*)
    asm("_ZN5nvdla8IRuntime27getOutputTaskStatisticsDescEiP30NvDlaRuntimeTaskStatisticsDesc");
extern "C" int getOutputTaskStatisticsDesc_stub(void*, int, NvDlaRuntimeTaskStatisticsDesc*) { return -1; }

extern "C" int submit_stub(void*, bool, bool, unsigned, unsigned, nvdla::ISync**)
    asm("_ZN5nvdla8IRuntime6submitEbbjjPPNS_5ISyncE");
extern "C" int submit_stub(void*, bool, bool, unsigned, unsigned, nvdla::ISync**) { return -1; }
CPP_EOF
    g++ -shared -fPIC "${ORIN_COMPAT_DIR}/libnvdla_runtime_stub.cpp" \
        -Wl,-soname,libnvdla_runtime.so \
        -o "${ORIN_COMPAT_DIR}/libnvdla_runtime.so"
fi

export PYTHONPATH="${PY_RUNTIME_DIR}:${BUILD_DIR}/install/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ORIN_COMPAT_DIR}:${BUILD_DIR}/install/lib:/usr/local/cuda-12.6/targets/aarch64-linux/lib/stubs:/usr/local/cuda/targets/aarch64-linux/lib/stubs:/usr/local/cuda-12.6/compat:/usr/local/cuda/compat:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

EDGE_FM_BUILD_DIR="${BUILD_DIR}" "${PYTHON_EXECUTABLE}" - <<'PY'
import edge_fm_trt
print(edge_fm_trt.__file__)
PY
EDGE_FM_BUILD_DIR="${BUILD_DIR}" "${PYTHON_EXECUTABLE}" -m pytest --collect-only tests/engine/test_qwen2_generate.py -q
EOF
}

edge_fm_docker_main "${1:-all}"
