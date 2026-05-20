# ============================================================================
# Platform Configuration: NVIDIA RTX 3060 (x86)
# ============================================================================

set(PLATFORM_NAME "NVIDIA RTX 3060")
set(PLATFORM_ARCH "x86_64")
set(PLATFORM_GPU "RTX 3060")

# CUDA 架构设置（Ampere 架构，Compute Capability 8.6）
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")

# CUDA 编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math" CACHE STRING "CUDA release flags")
else()
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G" CACHE STRING "CUDA debug flags")
endif()

# 平台特定的编译定义
add_compile_definitions(
    PLATFORM_RTX3060=1
    PLATFORM_X86=1
)

message(STATUS "Platform: ${PLATFORM_NAME}")
message(STATUS "  Architecture: ${PLATFORM_ARCH}")
message(STATUS "  GPU: ${PLATFORM_GPU}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

include(CheckLanguage)

set(ENABLE_CUDA ON CACHE BOOL "CUDA is enabled for NVIDIA platforms" FORCE)

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

check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found. Please install CUDA toolkit or set CMAKE_CUDA_COMPILER.")
endif()

enable_language(CUDA)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA toolkit root: ${CUDAToolkit_ROOT_DIR}")

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

if(EXISTS ${PROJECT_SOURCE_DIR}/third_party/flashinfer/include/flashinfer)
    add_library(edge_fm_flashinfer_headers INTERFACE)
    target_include_directories(edge_fm_flashinfer_headers INTERFACE
        ${PROJECT_SOURCE_DIR}/third_party/flashinfer/include
    )
    # CUDA 12+ CUB 移除了 FlagHeads，使用 SubtractLeft API。
    target_compile_definitions(edge_fm_flashinfer_headers INTERFACE
        FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
    )
    message(STATUS "FlashInfer found (header-only)")
else()
    message(FATAL_ERROR
        "flashinfer not found in third_party/flashinfer/include/flashinfer. "
        "CUDA builds require the flashinfer submodule.")
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/third_party/safetensors-cpp/safetensors.hh)
    add_library(safetensors_cpp INTERFACE)
    target_include_directories(safetensors_cpp INTERFACE
        ${PROJECT_SOURCE_DIR}/third_party/safetensors-cpp
    )
    message(STATUS "safetensors-cpp found (header-only)")
else()
    message(FATAL_ERROR
        "safetensors-cpp not found in third_party/safetensors-cpp. "
        "CUDA builds require the safetensors-cpp submodule.")
endif()

message(STATUS "CUDA: ENABLED")
add_compile_definitions(EDGE_FM_ENABLE_CUDA=1)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "CUDA release flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
endif()

set(EDGE_FM_ENABLE_TRT_FEATURES OFF)
set(EDGE_FM_ENABLE_TRT_PLUGIN_OPS OFF)
set(TRT_LIBRARY_DIR "")
set(TRT_VERSION_MAJOR "")
set(TRT_VERSION_MINOR "")
set(TRT_VERSION_PATCH "")
set(TRT_VERSION_BUILD "")
set(TRT_VERSION_STRING "")
set(TRT_EDGELLM_AVAILABLE FALSE)
set(TRT_EDGELLM_PLUGIN_LIBRARY "")
set(TRT_EDGELLM_PLUGIN_DIR "")
set(EDGE_FM_TRT_MIN_VERSION "")
set(EDGE_FM_DEFAULT_TRT_PACKAGE_DIR "")

if(BUILD_TRT_EDGELLM_PYBIND OR BUILD_TRT_PLUGIN_OPS)
    set(EDGE_FM_ENABLE_TRT_FEATURES ON)
    set(EDGE_FM_TRT_MIN_VERSION_MAJOR 10)
    set(EDGE_FM_TRT_MIN_VERSION_MINOR 15)
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

    if(TRT_INCLUDE_DIR)
        edge_fm_read_tensorrt_version("${TRT_INCLUDE_DIR}" TRT_VERSION_MAJOR TRT_VERSION_MINOR TRT_VERSION_PATCH TRT_VERSION_BUILD)
        if(TRT_VERSION_MAJOR AND TRT_VERSION_MINOR)
            set(TRT_VERSION_STRING "${TRT_VERSION_MAJOR}.${TRT_VERSION_MINOR}.${TRT_VERSION_PATCH}.${TRT_VERSION_BUILD}")
        endif()
    endif()

    if(BUILD_TRT_EDGELLM_PYBIND)
        set(TRT_EDGELLM_ROOT ${PROJECT_SOURCE_DIR}/third_party/TensorRT-Edge-LLM)
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

    if(BUILD_TRT_PLUGIN_OPS)
        set(TRT_EDGELLM_PLUGIN_CANDIDATE_DIRS
            "${CMAKE_BINARY_DIR}/trt-edgellm"
            "${PROJECT_SOURCE_DIR}/third_party/TensorRT-Edge-LLM/build")
        if(NOT TRT_INCLUDE_DIR OR NOT TRT_NVINFER_LIBRARY)
            message(FATAL_ERROR
                "BUILD_TRT_PLUGIN_OPS=ON requires TensorRT headers/libraries. "
                "Set TRT_PACKAGE_DIR (default: ${EDGE_FM_DEFAULT_TRT_PACKAGE_DIR}) "
                "or install TensorRT system-wide.")
        endif()
        if(NOT TRT_VERSION_STRING)
            message(FATAL_ERROR
                "Failed to determine TensorRT version from ${TRT_INCLUDE_DIR}. "
                "BUILD_TRT_PLUGIN_OPS requires TensorRT >= ${EDGE_FM_TRT_MIN_VERSION}.")
        endif()
        if(TRT_VERSION_MAJOR LESS EDGE_FM_TRT_MIN_VERSION_MAJOR
           OR (TRT_VERSION_MAJOR EQUAL EDGE_FM_TRT_MIN_VERSION_MAJOR
               AND TRT_VERSION_MINOR LESS EDGE_FM_TRT_MIN_VERSION_MINOR))
            message(FATAL_ERROR
                "TensorRT ${TRT_VERSION_STRING} found in ${TRT_INCLUDE_DIR}, "
                "but BUILD_TRT_PLUGIN_OPS requires TensorRT >= ${EDGE_FM_TRT_MIN_VERSION}. "
                "Set TRT_PACKAGE_DIR to a newer TensorRT installation.")
        endif()
        unset(TRT_EDGELLM_PLUGIN_LIBRARY CACHE)
        unset(TRT_EDGELLM_PLUGIN_LIBRARY)
        find_library(TRT_EDGELLM_PLUGIN_LIBRARY
            NAMES NvInfer_edgellm_plugin
            HINTS ${TRT_EDGELLM_PLUGIN_CANDIDATE_DIRS}
            NO_DEFAULT_PATH
        )
        if(NOT TRT_EDGELLM_PLUGIN_LIBRARY)
            message(FATAL_ERROR
                "BUILD_TRT_PLUGIN_OPS=ON requires third_party/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so. "
                "Run bash tests/scripts/setup_trt_edgellm_benchmark.sh first. "
                "Searched: ${TRT_EDGELLM_PLUGIN_CANDIDATE_DIRS}")
        endif()
        get_filename_component(TRT_EDGELLM_PLUGIN_DIR "${TRT_EDGELLM_PLUGIN_LIBRARY}" DIRECTORY)
        set(EDGE_FM_ENABLE_TRT_PLUGIN_OPS ON)
        add_compile_definitions(EDGE_FM_ENABLE_TRT_PLUGIN_OPS=1)
        message(STATUS "TensorRT plugin operators include: ${TRT_INCLUDE_DIR}")
        message(STATUS "TensorRT plugin operators library: ${TRT_EDGELLM_PLUGIN_LIBRARY}")
    endif()
endif()
