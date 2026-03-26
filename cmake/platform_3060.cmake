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
