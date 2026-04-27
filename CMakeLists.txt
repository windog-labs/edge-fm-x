cmake_minimum_required(VERSION 3.15)
project(edge-fm
    VERSION 0.1.0
    LANGUAGES CXX
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not supported. "
        "Use `cmake --preset <platform>` or `cmake -S . -B build-<platform>` instead.")
endif()

include(CheckLanguage)
include(CTest)

# ============================================================================
# 构建类型配置
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 根据构建类型设置编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
    if(MSVC)
        add_compile_options(/Od /Zi)
    else()
        add_compile_options(-g -O0)
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
endif()

# ============================================================================
# C++ 标准配置
# ============================================================================
# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 生成 compile_commands.json（用于 IDE 和 clangd）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 静态库需要 -fPIC 以便链接到共享库
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# 编译选项
# ============================================================================
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================================================
# 安装路径配置
# ============================================================================
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix" FORCE)
endif()

# ============================================================================
# 输出目录配置
# ============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================================================
# 平台选择
# ============================================================================
set(PLATFORM_OPTIONS "3060" "a800" "orin" "j6m")
set(PLATFORM "a800" CACHE STRING "Target platform")
set_property(CACHE PLATFORM PROPERTY STRINGS ${PLATFORM_OPTIONS})

message(STATUS "Target platform: ${PLATFORM}")

set(EDGE_FM_MULTIARCH_TRIPLET "")
if(CMAKE_LIBRARY_ARCHITECTURE)
    set(EDGE_FM_MULTIARCH_TRIPLET "${CMAKE_LIBRARY_ARCHITECTURE}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(EDGE_FM_MULTIARCH_TRIPLET "aarch64-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(EDGE_FM_MULTIARCH_TRIPLET "x86_64-linux-gnu")
endif()

set(_edge_fm_cuda_target_candidates "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    list(APPEND _edge_fm_cuda_target_candidates "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    list(APPEND _edge_fm_cuda_target_candidates "x86_64-linux")
endif()
list(APPEND _edge_fm_cuda_target_candidates "aarch64-linux" "x86_64-linux")
list(REMOVE_DUPLICATES _edge_fm_cuda_target_candidates)

# 加载平台特定配置
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/platform_${PLATFORM}.cmake)
    include(cmake/platform_${PLATFORM}.cmake)
    message(STATUS "Platform configuration loaded: platform_${PLATFORM}.cmake")
else()
    message(WARNING "Platform configuration file not found: cmake/platform_${PLATFORM}.cmake")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release"
   AND (PLATFORM STREQUAL "3060"
        OR PLATFORM STREQUAL "a800"
        OR PLATFORM STREQUAL "orin"))
    set(_edge_fm_cuda_release_flags "${CMAKE_CUDA_FLAGS_RELEASE}")
    if(NOT _edge_fm_cuda_release_flags MATCHES "(^| )-use_fast_math($| )")
        string(APPEND _edge_fm_cuda_release_flags " -use_fast_math")
        string(STRIP "${_edge_fm_cuda_release_flags}" _edge_fm_cuda_release_flags)
        set(CMAKE_CUDA_FLAGS_RELEASE "${_edge_fm_cuda_release_flags}" CACHE STRING "CUDA release flags" FORCE)
    endif()
    message(STATUS "CUDA release flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

# ============================================================================
# CUDA 和 cuDNN 检查和配置（仅适用于 NVIDIA 平台）
# ============================================================================
set(ENABLE_CUDA OFF)
if(PLATFORM STREQUAL "3060" OR PLATFORM STREQUAL "a800" OR PLATFORM STREQUAL "orin")
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found. Please install CUDA toolkit or set CMAKE_CUDA_COMPILER.")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA toolkit root: ${CUDAToolkit_ROOT_DIR}")
        # CUDA 找到后，启用 CUDA device utilities
        set(ENABLE_CUDA ON)
        # FlashInfer sampling.cuh 需要 libcu++ (cuda/functional)。
        set(CUDA_LIBCPP_INCLUDE_DIR "")
        foreach(_base "${CUDAToolkit_ROOT_DIR}" "$ENV{CUDA_PATH}" "/usr/local/cuda" "/usr/local/cuda-12.6" "/usr/local/cuda-12")
            foreach(_cuda_target IN LISTS _edge_fm_cuda_target_candidates)
                if(_base AND EXISTS "${_base}/targets/${_cuda_target}/include/cuda/functional")
                    set(CUDA_LIBCPP_INCLUDE_DIR "${_base}/targets/${_cuda_target}/include" CACHE INTERNAL "libcu++ include for FlashInfer")
                    break()
                endif()
            endforeach()
            if(CUDA_LIBCPP_INCLUDE_DIR)
                break()
            endif()
        endforeach()
        if(CUDA_LIBCPP_INCLUDE_DIR)
            include_directories(${CUDA_LIBCPP_INCLUDE_DIR})
            message(STATUS "CUDA libcu++ include: ${CUDA_LIBCPP_INCLUDE_DIR}")
        endif()
    else()
        message(FATAL_ERROR "CUDA not found. Please install CUDA toolkit.")
    endif()

    # 查找 cuDNN
    find_path(CUDNN_INCLUDE_DIR
        NAMES cudnn.h
        PATHS
            ${CUDAToolkit_INCLUDE_DIRS}
            /usr/local/cuda/include
            /usr/local/cuda/targets/aarch64-linux/include
            /usr/local/cuda/targets/x86_64-linux/include
            /usr/include
            /usr/include/${EDGE_FM_MULTIARCH_TRIPLET}
            /usr/include/aarch64-linux-gnu
            /usr/include/x86_64-linux-gnu
        PATH_SUFFIXES cudnn
    )

    find_library(CUDNN_LIBRARY
        NAMES cudnn
        PATHS
            ${CUDAToolkit_LIBRARY_DIR}
            /usr/local/cuda/lib64
            /usr/local/cuda/targets/aarch64-linux/lib
            /usr/local/cuda/targets/x86_64-linux/lib
            /lib/${EDGE_FM_MULTIARCH_TRIPLET}
            /lib/aarch64-linux-gnu
            /lib/x86_64-linux-gnu
            /usr/lib/${EDGE_FM_MULTIARCH_TRIPLET}
            /usr/lib/aarch64-linux-gnu
            /usr/lib/x86_64-linux-gnu
    )

    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        message(STATUS "cuDNN found")
        message(STATUS "  Include: ${CUDNN_INCLUDE_DIR}")
        message(STATUS "  Library: ${CUDNN_LIBRARY}")
    else()
        message(FATAL_ERROR "cuDNN not found. Please install cuDNN library.")
    endif()
else()
    message(STATUS "CUDA and cuDNN checks skipped for platform: ${PLATFORM}")
endif()

# 输出 CUDA 状态
if(ENABLE_CUDA)
    message(STATUS "CUDA: ENABLED")
    add_compile_definitions(EDGE_FM_ENABLE_CUDA=1)
else()
    message(STATUS "CUDA: DISABLED")
endif()

# ============================================================================
# 第三方依赖
# ============================================================================
set(EDGE_FM_ACTIVE_CONFIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples/config/platform/${PLATFORM}")
if(NOT EXISTS "${EDGE_FM_ACTIVE_CONFIG_DIR}/engine_default.json")
    message(FATAL_ERROR "Platform config directory is incomplete: ${EDGE_FM_ACTIVE_CONFIG_DIR}")
endif()

# ============================================================================
# 构建选项
# ============================================================================
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TRT_EDGELLM_PYBIND "Build Python bindings for TRT-Edge-LLM runtime (in-process benchmark)" OFF)
option(ENABLE_HORIZON_RUNTIME "Build Horizon HBM runtime backend when Horizon SDK is available" OFF)
set(HORIZON_DEPS_ROOT "" CACHE PATH "Horizon SDK/deps root; expected to contain ucp/include")
set(HORIZON_INCLUDE_DIR "" CACHE PATH "Explicit Horizon runtime include directory override")
set(HORIZON_LIBRARY_DIR "" CACHE PATH "Explicit Horizon runtime library directory override")
set(HORIZON_LIBRARIES "" CACHE STRING "Explicit Horizon runtime libraries, separated by semicolons")

set(EDGE_FM_ENABLE_HORIZON_RUNTIME OFF)
set(EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIRS "")
set(EDGE_FM_HORIZON_RUNTIME_LIBRARIES "")
if(ENABLE_HORIZON_RUNTIME)
    set(_edge_fm_horizon_include_hints "")
    if(HORIZON_INCLUDE_DIR)
        list(APPEND _edge_fm_horizon_include_hints "${HORIZON_INCLUDE_DIR}")
    endif()
    if(HORIZON_DEPS_ROOT)
        list(APPEND _edge_fm_horizon_include_hints
            "${HORIZON_DEPS_ROOT}/ucp/include"
            "${HORIZON_DEPS_ROOT}/include"
        )
    endif()

    find_path(EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIR
        NAMES hobot/dnn/hb_dnn.h
        HINTS ${_edge_fm_horizon_include_hints}
    )

    if(NOT EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR
            "ENABLE_HORIZON_RUNTIME=ON but Horizon headers were not found. "
            "Set HORIZON_DEPS_ROOT or HORIZON_INCLUDE_DIR.")
    endif()

    list(APPEND EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIRS "${EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIR}")
    if(HORIZON_LIBRARIES)
        set(EDGE_FM_HORIZON_RUNTIME_LIBRARIES ${HORIZON_LIBRARIES})
    elseif(HORIZON_LIBRARY_DIR)
        link_directories("${HORIZON_LIBRARY_DIR}")
    endif()
    set(EDGE_FM_ENABLE_HORIZON_RUNTIME ON)
    message(STATUS "Horizon runtime backend: ENABLED")
    message(STATUS "Horizon include: ${EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIR}")
else()
    message(STATUS "Horizon runtime backend: DISABLED")
endif()

# nlohmann/json
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/CMakeLists.txt)
    add_subdirectory(third_party/json)
else()
    message(WARNING "nlohmann/json not found in third_party/json. Please add it manually.")
endif()

# flashinfer (header-only library)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer/include/flashinfer)
    # flashinfer 是 header-only 库，只需要添加包含路径
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer/include)
    # CUDA 12+ CUB 移除了 FlagHeads，使用 SubtractLeft API
    add_compile_definitions(FLASHINFER_CUB_SUBTRACTLEFT_DEFINED)
    message(STATUS "FlashInfer found (header-only)")
else()
    message(WARNING "flashinfer not found in third_party/flashinfer/include/flashinfer. Please add it manually.")
endif()

# pybind11
if(BUILD_PYTHON OR BUILD_TRT_EDGELLM_PYBIND)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/CMakeLists.txt)
        add_subdirectory(third_party/pybind11)
    else()
        message(WARNING "pybind11 not found in third_party/pybind11. Please add it manually.")
    endif()
endif()

# spdlog
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/CMakeLists.txt)
    set(SPDLOG_BUILD_PIC ON CACHE BOOL "Build position independent code (-fPIC)" FORCE)
    add_subdirectory(third_party/spdlog)
else()
    message(WARNING "spdlog not found in third_party/spdlog. Please add it manually.")
endif()

# dlpack
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dlpack/include/dlpack/dlpack.h)
    # DLPack is a header-only library, just add include directory
    add_library(dlpack INTERFACE)
    target_include_directories(dlpack INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dlpack/include
    )
    message(STATUS "DLPack found in third_party/dlpack")
    # Add dlpack include directory globally since it's used in header files
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/dlpack/include)
else()
    message(WARNING "dlpack not found in third_party/dlpack. Please add it manually.")
    message(WARNING "You can clone it with: git clone https://github.com/dmlc/dlpack.git third_party/dlpack")
endif()

# safetensors-cpp (header-only library)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/safetensors-cpp/safetensors.hh)
    # safetensors-cpp is a header-only library, just add include directory
    add_library(safetensors_cpp INTERFACE)
    target_include_directories(safetensors_cpp INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/safetensors-cpp
    )
    message(STATUS "safetensors-cpp found (header-only)")
    # Add safetensors-cpp include directory globally
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/safetensors-cpp)
else()
    message(WARNING "safetensors-cpp not found in third_party/safetensors-cpp. Please add it manually.")
    message(WARNING "You can clone it with: git clone https://github.com/syoyo/safetensors-cpp.git third_party/safetensors-cpp")
endif()

set(EDGE_FM_TRT_MIN_VERSION_MAJOR 10)
# JetPack r36.4.0 ships TensorRT 10.3 in /usr on Orin, while the x86 benchmark
# path currently expects newer TensorRT 10.15 builds.
if(PLATFORM STREQUAL "orin")
    set(EDGE_FM_TRT_MIN_VERSION_MINOR 3)
else()
    set(EDGE_FM_TRT_MIN_VERSION_MINOR 15)
endif()
set(EDGE_FM_TRT_MIN_VERSION "${EDGE_FM_TRT_MIN_VERSION_MAJOR}.${EDGE_FM_TRT_MIN_VERSION_MINOR}")
set(EDGE_FM_DEFAULT_TRT_PACKAGE_DIR "/usr/local/TensorRT")

function(edge_fm_read_tensorrt_version include_dir out_major out_minor out_patch out_build)
    set(_version_header_candidates
        "${include_dir}/NvInferVersion.h"
        "${include_dir}/include/NvInferVersion.h"
    )

    foreach(_version_header IN LISTS _version_header_candidates)
        if(NOT EXISTS "${_version_header}")
            continue()
        endif()

        file(STRINGS "${_version_header}" _major_line REGEX "^#define (NV_TENSORRT_MAJOR|TRT_MAJOR_ENTERPRISE) +[0-9]+")
        file(STRINGS "${_version_header}" _minor_line REGEX "^#define (NV_TENSORRT_MINOR|TRT_MINOR_ENTERPRISE) +[0-9]+")
        file(STRINGS "${_version_header}" _patch_line REGEX "^#define (NV_TENSORRT_PATCH|TRT_PATCH_ENTERPRISE) +[0-9]+")
        file(STRINGS "${_version_header}" _build_line REGEX "^#define (NV_TENSORRT_BUILD|TRT_BUILD_ENTERPRISE) +[0-9]+")

        if(_major_line AND _minor_line AND _patch_line AND _build_line)
            string(REGEX REPLACE ".* ([0-9]+).*" "\\1" _major "${_major_line}")
            string(REGEX REPLACE ".* ([0-9]+).*" "\\1" _minor "${_minor_line}")
            string(REGEX REPLACE ".* ([0-9]+).*" "\\1" _patch "${_patch_line}")
            string(REGEX REPLACE ".* ([0-9]+).*" "\\1" _build "${_build_line}")
            set(${out_major} "${_major}" PARENT_SCOPE)
            set(${out_minor} "${_minor}" PARENT_SCOPE)
            set(${out_patch} "${_patch}" PARENT_SCOPE)
            set(${out_build} "${_build}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${out_major} "" PARENT_SCOPE)
    set(${out_minor} "" PARENT_SCOPE)
    set(${out_patch} "" PARENT_SCOPE)
    set(${out_build} "" PARENT_SCOPE)
endfunction()

set(TRT_PACKAGE_DIR "${EDGE_FM_DEFAULT_TRT_PACKAGE_DIR}" CACHE PATH "TensorRT package directory")
if(TRT_PACKAGE_DIR STREQUAL "")
    set(TRT_PACKAGE_DIR "${EDGE_FM_DEFAULT_TRT_PACKAGE_DIR}" CACHE PATH "TensorRT package directory" FORCE)
endif()

set(_edge_fm_trt_root_hints "")
if(TRT_PACKAGE_DIR)
    list(APPEND _edge_fm_trt_root_hints "${TRT_PACKAGE_DIR}")
endif()
file(GLOB _edge_fm_trt_root_globs "/usr/local/TensorRT-*")
list(APPEND _edge_fm_trt_root_hints
    ${_edge_fm_trt_root_globs}
    "/usr"
)
list(REMOVE_DUPLICATES _edge_fm_trt_root_hints)

find_path(TRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_edge_fm_trt_root_hints}
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES include include/${EDGE_FM_MULTIARCH_TRIPLET} ${EDGE_FM_MULTIARCH_TRIPLET}
)

find_library(TRT_NVINFER_LIBRARY
    NAMES nvinfer
    HINTS ${_edge_fm_trt_root_hints}
    PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64
    PATH_SUFFIXES lib lib64 lib/${EDGE_FM_MULTIARCH_TRIPLET} ${EDGE_FM_MULTIARCH_TRIPLET}
)

if(TRT_NVINFER_LIBRARY)
    get_filename_component(TRT_LIBRARY_DIR "${TRT_NVINFER_LIBRARY}" DIRECTORY)
else()
    set(TRT_LIBRARY_DIR "")
endif()

set(TRT_VERSION_MAJOR "")
set(TRT_VERSION_MINOR "")
set(TRT_VERSION_PATCH "")
set(TRT_VERSION_BUILD "")
set(TRT_VERSION_STRING "")
if(TRT_INCLUDE_DIR)
    edge_fm_read_tensorrt_version("${TRT_INCLUDE_DIR}" TRT_VERSION_MAJOR TRT_VERSION_MINOR TRT_VERSION_PATCH TRT_VERSION_BUILD)
    if(TRT_VERSION_MAJOR AND TRT_VERSION_MINOR)
        set(TRT_VERSION_STRING "${TRT_VERSION_MAJOR}.${TRT_VERSION_MINOR}.${TRT_VERSION_PATCH}.${TRT_VERSION_BUILD}")
    endif()
endif()

if(BUILD_PYTHON)
    message(STATUS "Python bindings: ENABLED")
else()
    message(STATUS "Python bindings: DISABLED")
endif()

if(BUILD_TRT_EDGELLM_PYBIND)
    message(STATUS "TRT-Edge-LLM Python bindings: ENABLED")
else()
    message(STATUS "TRT-Edge-LLM Python bindings: DISABLED")
endif()

# ============================================================================
# TensorRT-Edge-LLM (optional, for edge_fm_trt Python module)
# 使用预构建的 TRT-Edge-LLM：需先运行 bash tests/scripts/setup_trt_edgellm_benchmark.sh
# ============================================================================
if(BUILD_TRT_EDGELLM_PYBIND)
    set(TRT_EDGELLM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/TensorRT-Edge-LLM)
    set(TRT_EDGELLM_BUILD_DIR "${CMAKE_BINARY_DIR}/trt-edgellm" CACHE PATH "TensorRT-Edge-LLM build directory")

    set(_trt_core "${TRT_EDGELLM_BUILD_DIR}/cpp/libedgellmCore.a")
    set(_trt_tokenizer "${TRT_EDGELLM_BUILD_DIR}/cpp/libedgellmTokenizer.a")
    set(_trt_utils "${TRT_EDGELLM_BUILD_DIR}/examples/utils/libexampleUtils.a")

    if(TRT_INCLUDE_DIR AND TRT_NVINFER_LIBRARY)
        if(NOT TRT_VERSION_STRING)
            message(FATAL_ERROR
                "Failed to determine TensorRT version from ${TRT_INCLUDE_DIR}. "
                "TensorRT-Edge-LLM requires TensorRT >= ${EDGE_FM_TRT_MIN_VERSION}.")
        endif()
        if(TRT_VERSION_MAJOR LESS EDGE_FM_TRT_MIN_VERSION_MAJOR
           OR (TRT_VERSION_MAJOR EQUAL EDGE_FM_TRT_MIN_VERSION_MAJOR
               AND TRT_VERSION_MINOR LESS EDGE_FM_TRT_MIN_VERSION_MINOR))
            message(FATAL_ERROR
                "TensorRT ${TRT_VERSION_STRING} found in ${TRT_INCLUDE_DIR}, "
                "but TensorRT-Edge-LLM requires TensorRT >= ${EDGE_FM_TRT_MIN_VERSION}. "
                "Set TRT_PACKAGE_DIR to a newer TensorRT installation.")
        endif()
        message(STATUS "TensorRT version: ${TRT_VERSION_STRING}")
    endif()

    if(EXISTS "${_trt_core}" AND EXISTS "${_trt_tokenizer}" AND EXISTS "${_trt_utils}" AND TRT_INCLUDE_DIR AND TRT_NVINFER_LIBRARY)
        set(TRT_EDGELLM_AVAILABLE TRUE)
        set(TRT_EDGELLM_CORE_LIB "${_trt_core}")
        set(TRT_EDGELLM_TOKENIZER_LIB "${_trt_tokenizer}")
        set(TRT_EDGELLM_UTILS_LIB "${_trt_utils}")
        message(STATUS "Using pre-built TensorRT-Edge-LLM from ${TRT_EDGELLM_BUILD_DIR}")
        message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
        message(STATUS "TensorRT library: ${TRT_NVINFER_LIBRARY}")
    else()
        if(NOT TRT_INCLUDE_DIR OR NOT TRT_NVINFER_LIBRARY)
            message(WARNING
                "TensorRT headers/libraries were not found. "
                "Set TRT_PACKAGE_DIR (default: ${EDGE_FM_DEFAULT_TRT_PACKAGE_DIR}) "
                "or install TensorRT system-wide.")
        endif()
        message(WARNING "TensorRT-Edge-LLM not built. Run: bash tests/scripts/setup_trt_edgellm_benchmark.sh")
        set(BUILD_TRT_EDGELLM_PYBIND OFF)
    endif()
endif()

# ============================================================================
# 全局包含目录
# ============================================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# ============================================================================
# 添加源代码目录
# ============================================================================
add_subdirectory(src)
if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()
