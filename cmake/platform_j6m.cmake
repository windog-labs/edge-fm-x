# ============================================================================
# Platform Configuration: Horizon J6M
# ============================================================================

set(PLATFORM_NAME "Horizon J6M")
set(PLATFORM_ARCH "x86_64")
set(PLATFORM_GPU "J6M")

add_compile_definitions(
    PLATFORM_HORIZON=1
    PLATFORM_J6M=1
)

message(STATUS "Platform: ${PLATFORM_NAME}")
message(STATUS "  Architecture: ${PLATFORM_ARCH}")
message(STATUS "  Accelerator: ${PLATFORM_GPU}")
