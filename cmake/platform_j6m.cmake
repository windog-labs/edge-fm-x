# ============================================================================
# Platform Configuration: Horizon J6M
# ============================================================================

set(PLATFORM_NAME "Horizon J6M")
set(PLATFORM_ARCH "aarch64")
set(PLATFORM_ACCELERATOR "BPU Nash-M")

set(EDGE_FM_J6M_TOOLCHAIN_ROOT
    "/arm-gnu-toolchain-12.2.rel1-x86_64-aarch64-none-linux-gnu"
    CACHE PATH
    "Horizon J6M aarch64 GCC toolchain root")
set(EDGE_FM_J6M_TARGET_TRIPLE
    "aarch64-none-linux-gnu"
    CACHE STRING
    "Horizon J6M aarch64 toolchain target triple")
set(EDGE_FM_J6M_MARCH
    "nash-m"
    CACHE STRING
    "Horizon BPU march used by model compilation helpers")

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

if(EXISTS "${EDGE_FM_J6M_TOOLCHAIN_ROOT}")
    list(APPEND CMAKE_FIND_ROOT_PATH "${EDGE_FM_J6M_TOOLCHAIN_ROOT}/${EDGE_FM_J6M_TARGET_TRIPLE}")
endif()

set(BUILD_PYTHON OFF CACHE BOOL "Python bindings are disabled for the J6M target build")
set(ENABLE_CUDA OFF CACHE BOOL "CUDA is not used by the Horizon J6M whole-model backend")

if(EXISTS "/usr/ucp")
    set(HORIZON_DEPS_ROOT "/usr/ucp" CACHE PATH "Horizon SDK/deps root")
    set(HORIZON_LIBRARY_DIR "/usr/ucp" CACHE PATH "Horizon runtime library directory")
endif()

add_compile_definitions(
    PLATFORM_HORIZON=1
    PLATFORM_J6M=1
    PLATFORM_AARCH64=1
)

message(STATUS "Platform: ${PLATFORM_NAME}")
message(STATUS "  Architecture: ${PLATFORM_ARCH}")
message(STATUS "  Accelerator: ${PLATFORM_ACCELERATOR}")
message(STATUS "  BPU march: ${EDGE_FM_J6M_MARCH}")
message(STATUS "  Toolchain root: ${EDGE_FM_J6M_TOOLCHAIN_ROOT}")
message(STATUS "  Target triple: ${EDGE_FM_J6M_TARGET_TRIPLE}")
