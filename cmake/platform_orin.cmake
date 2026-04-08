# ============================================================================
# Platform Configuration: NVIDIA Jetson Orin
# ============================================================================

set(PLATFORM_NAME "NVIDIA Jetson Orin")
set(PLATFORM_ARCH "aarch64")
set(PLATFORM_GPU "Orin")

# CUDA 架构设置（Ampere 架构，Compute Capability 8.7）
set(CMAKE_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA architectures")

# CUDA 编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math" CACHE STRING "CUDA release flags")
else()
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G" CACHE STRING "CUDA debug flags")
endif()

# 平台特定的编译定义
add_compile_definitions(
    PLATFORM_ORIN=1
    PLATFORM_JETSON=1
    PLATFORM_AARCH64=1
)

message(STATUS "Platform: ${PLATFORM_NAME}")
message(STATUS "  Architecture: ${PLATFORM_ARCH}")
message(STATUS "  GPU: ${PLATFORM_GPU}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
