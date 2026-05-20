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
    add_compile_definitions(EDGE_FM_DEBUG=1)
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

# ============================================================================
# 构建选项
# ============================================================================
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TRT_EDGELLM_PYBIND "Build Python bindings for TRT-Edge-LLM runtime (in-process benchmark)" OFF)
option(BUILD_TRT_PLUGIN_OPS "Build default-off EdgeFM operators that directly reuse TRT-Edge-LLM plugin kernels without TensorRT engines" OFF)
option(ENABLE_HORIZON_RUNTIME "Build Horizon HBM runtime backend when Horizon SDK is available" OFF)
set(HORIZON_DEPS_ROOT "" CACHE PATH "Horizon SDK/deps root; expected to contain ucp/include")
set(HORIZON_INCLUDE_DIR "" CACHE PATH "Explicit Horizon runtime include directory override")
set(HORIZON_LIBRARY_DIR "" CACHE PATH "Explicit Horizon runtime library directory override")
set(HORIZON_LIBRARIES "" CACHE STRING "Explicit Horizon runtime libraries, separated by semicolons")

set(ENABLE_CUDA OFF CACHE BOOL "CUDA is enabled by CUDA platform configuration")
set(EDGE_FM_ENABLE_HORIZON_RUNTIME OFF)
set(EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIRS "")
set(EDGE_FM_HORIZON_RUNTIME_LIBRARY_DIRS "")
set(EDGE_FM_HORIZON_RUNTIME_LIBRARIES "")
set(EDGE_FM_ENABLE_TRT_PLUGIN_OPS OFF)
set(TRT_LIBRARY_DIR "")
set(TRT_EDGELLM_PLUGIN_DIR "")
set(TRT_EDGELLM_PLUGIN_LIBRARY "")
set(TRT_EDGELLM_AVAILABLE FALSE)

# 加载平台特定配置；CUDA/cuDNN/TensorRT/Horizon 细节由对应平台配置负责。
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/platform_${PLATFORM}.cmake)
    include(cmake/platform_${PLATFORM}.cmake)
    message(STATUS "Platform configuration loaded: platform_${PLATFORM}.cmake")
else()
    message(WARNING "Platform configuration file not found: cmake/platform_${PLATFORM}.cmake")
endif()

if((BUILD_TRT_EDGELLM_PYBIND OR BUILD_TRT_PLUGIN_OPS) AND NOT ENABLE_CUDA)
    message(FATAL_ERROR
        "BUILD_TRT_EDGELLM_PYBIND/BUILD_TRT_PLUGIN_OPS require a CUDA platform. "
        "Disable these options for PLATFORM=${PLATFORM}.")
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

if(BUILD_TRT_PLUGIN_OPS)
    message(STATUS "TensorRT plugin operators: ENABLED")
else()
    message(STATUS "TensorRT plugin operators: DISABLED")
endif()

# ============================================================================
# 第三方依赖
# ============================================================================
set(EDGE_FM_ACTIVE_CONFIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples/config/platform/${PLATFORM}")
if(NOT EXISTS "${EDGE_FM_ACTIVE_CONFIG_DIR}/engine_default.json")
    message(FATAL_ERROR "Platform config directory is incomplete: ${EDGE_FM_ACTIVE_CONFIG_DIR}")
endif()

# nlohmann/json
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/CMakeLists.txt)
    add_subdirectory(third_party/json)
else()
    message(WARNING "nlohmann/json not found in third_party/json. Please add it manually.")
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
