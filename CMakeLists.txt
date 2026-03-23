cmake_minimum_required(VERSION 3.15)
project(edge-fm 
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

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
set(PLATFORM_OPTIONS "5090" "4050" "3060" "a100" "thor" "j6m")
set(PLATFORM "a100" CACHE STRING "Target platform")
set_property(CACHE PLATFORM PROPERTY STRINGS ${PLATFORM_OPTIONS})

message(STATUS "Target platform: ${PLATFORM}")

# 加载平台特定配置
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/platform_${PLATFORM}.cmake)
    include(cmake/platform_${PLATFORM}.cmake)
    message(STATUS "Platform configuration loaded: platform_${PLATFORM}.cmake")
else()
    message(WARNING "Platform configuration file not found: cmake/platform_${PLATFORM}.cmake")
endif()

# ============================================================================
# CUDA 和 cuDNN 检查和配置（仅适用于 NVIDIA 平台）
# ============================================================================
set(ENABLE_CUDA OFF)
if(PLATFORM STREQUAL "5090" OR PLATFORM STREQUAL "4050" OR PLATFORM STREQUAL "3060" OR PLATFORM STREQUAL "a100" OR PLATFORM STREQUAL "thor")
    # 查找 CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA toolkit root: ${CUDAToolkit_ROOT_DIR}")
        enable_language(CUDA)
        # CUDA 找到后，启用 CUDA device utilities
        set(ENABLE_CUDA ON)
        # FlashInfer sampling.cuh 需要 libcu++ (cuda/functional)，位于 targets/x86_64-linux/include
        set(CUDA_LIBCPP_INCLUDE_DIR "")
        foreach(_base "${CUDAToolkit_ROOT_DIR}" "$ENV{CUDA_PATH}" "/usr/local/cuda" "/usr/local/cuda-12.6" "/usr/local/cuda-12")
            if(_base AND EXISTS "${_base}/targets/x86_64-linux/include/cuda/functional")
                set(CUDA_LIBCPP_INCLUDE_DIR "${_base}/targets/x86_64-linux/include" CACHE INTERNAL "libcu++ include for FlashInfer")
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
            /usr/local/cuda/targets/x86_64-linux/include
            /usr/include
        PATH_SUFFIXES cudnn
    )

    find_library(CUDNN_LIBRARY
        NAMES cudnn
        PATHS
            ${CUDAToolkit_LIBRARY_DIR}
            /usr/local/cuda/lib64
            /usr/local/cuda/targets/x86_64-linux/lib
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
else()
    message(STATUS "CUDA: DISABLED")
endif()

# ============================================================================
# 第三方依赖
# ============================================================================
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
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/CMakeLists.txt)
    add_subdirectory(third_party/pybind11)
else()
    message(WARNING "pybind11 not found in third_party/pybind11. Please add it manually.")
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


# ============================================================================
# 构建选项
# ============================================================================
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TRT_EDGELLM_PYBIND "Build Python bindings for TRT-Edge-LLM runtime (in-process benchmark)" OFF)

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
    set(TRT_EDGELLM_BUILD_DIR "${TRT_EDGELLM_ROOT}/build" CACHE PATH "TensorRT-Edge-LLM build directory")
    set(TRT_PACKAGE_DIR "${TRT_PACKAGE_DIR}" CACHE PATH "TensorRT package directory (e.g. /usr/local/TensorRT-10.15.1.29)")
    if(NOT TRT_PACKAGE_DIR)
        set(TRT_PACKAGE_DIR "/usr/local/TensorRT-10.15.1.29" CACHE PATH "TensorRT package directory" FORCE)
    endif()

    set(_trt_core "${TRT_EDGELLM_BUILD_DIR}/cpp/libedgellmCore.a")
    set(_trt_tokenizer "${TRT_EDGELLM_BUILD_DIR}/cpp/libedgellmTokenizer.a")
    set(_trt_utils "${TRT_EDGELLM_BUILD_DIR}/examples/utils/libexampleUtils.a")

    if(EXISTS "${_trt_core}" AND EXISTS "${_trt_tokenizer}" AND EXISTS "${_trt_utils}" AND EXISTS "${TRT_PACKAGE_DIR}/include/NvInfer.h")
        set(TRT_EDGELLM_AVAILABLE TRUE)
        set(TRT_EDGELLM_CORE_LIB "${_trt_core}")
        set(TRT_EDGELLM_TOKENIZER_LIB "${_trt_tokenizer}")
        set(TRT_EDGELLM_UTILS_LIB "${_trt_utils}")
        message(STATUS "Using pre-built TensorRT-Edge-LLM from ${TRT_EDGELLM_BUILD_DIR}")
    else()
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