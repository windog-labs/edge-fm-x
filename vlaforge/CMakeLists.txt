cmake_minimum_required(VERSION 3.18)
project(vlaforge_runtime VERSION 0.2.0 LANGUAGES C CXX)

include(CTest)
include(CMakePackageConfigHelpers)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VLAFORGE_ENABLE_SANITIZERS "Enable address and undefined sanitizers" OFF)
option(VLAFORGE_BUILD_AOTI_BACKEND
       "Build the optional CPU/CUDA AOTInductor RegionExecutable backend" OFF)
option(VLAFORGE_BUILD_TORCHSCRIPT_BACKEND
       "Build the optional CPU TorchScript RegionExecutable backend" OFF)

add_library(vlaforge_runtime STATIC
    runtime/artifact_verifier.cpp
    runtime/device_copy.cpp
    runtime/region_executable.c
    runtime/session_c.c
    runtime/static_arena.cpp
    runtime/state_store.cpp
    runtime/transaction.cpp
)
target_link_libraries(vlaforge_runtime PUBLIC OpenSSL::Crypto)
target_include_directories(vlaforge_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
if(MSVC)
    target_compile_options(vlaforge_runtime PRIVATE /W4 /WX)
else()
    target_compile_options(vlaforge_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(VLAFORGE_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(vlaforge_runtime PUBLIC
        -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(vlaforge_runtime PUBLIC -fsanitize=address,undefined)
endif()

if(VLAFORGE_BUILD_AOTI_BACKEND OR VLAFORGE_BUILD_TORCHSCRIPT_BACKEND)
    find_package(Torch REQUIRED CONFIG)
    separate_arguments(VLAFORGE_TORCH_CXX_FLAGS
        NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
endif()

if(VLAFORGE_BUILD_AOTI_BACKEND)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(vlaforge_runtime
        PUBLIC VLAFORGE_ENABLE_CUDA_ARENA=1)
    target_link_libraries(vlaforge_runtime PUBLIC CUDA::cudart)
    add_library(vlaforge_aoti_backend STATIC
        backends/aoti_region_executable.cpp
    )
    target_include_directories(vlaforge_aoti_backend
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(vlaforge_aoti_backend
        PUBLIC vlaforge_runtime ${TORCH_LIBRARIES}
    )
    target_compile_options(vlaforge_aoti_backend
        PRIVATE ${VLAFORGE_TORCH_CXX_FLAGS}
    )
    if(MSVC)
        target_compile_options(vlaforge_aoti_backend PRIVATE /W4 /WX)
    else()
        target_compile_options(vlaforge_aoti_backend PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VLAFORGE_BUILD_TORCHSCRIPT_BACKEND)
    add_library(vlaforge_torchscript_backend STATIC
        backends/torchscript_region_executable.cpp
    )
    target_include_directories(vlaforge_torchscript_backend
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(vlaforge_torchscript_backend
        PUBLIC vlaforge_runtime ${TORCH_LIBRARIES}
    )
    target_compile_options(vlaforge_torchscript_backend
        PRIVATE ${VLAFORGE_TORCH_CXX_FLAGS}
    )
    if(MSVC)
        target_compile_options(vlaforge_torchscript_backend PRIVATE /W4 /WX)
    else()
        target_compile_options(vlaforge_torchscript_backend PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()

install(TARGETS vlaforge_runtime
    EXPORT VLAForgeRuntimeTargets
    ARCHIVE DESTINATION lib
)
if(TARGET vlaforge_aoti_backend)
    install(TARGETS vlaforge_aoti_backend
        EXPORT VLAForgeRuntimeTargets
        ARCHIVE DESTINATION lib
    )
endif()
if(TARGET vlaforge_torchscript_backend)
    install(TARGETS vlaforge_torchscript_backend
        EXPORT VLAForgeRuntimeTargets
        ARCHIVE DESTINATION lib
    )
endif()
install(DIRECTORY include/ DESTINATION include)
set(VLAFORGE_CONFIG_HAS_CUDA ${VLAFORGE_BUILD_AOTI_BACKEND})
if(VLAFORGE_BUILD_AOTI_BACKEND OR VLAFORGE_BUILD_TORCHSCRIPT_BACKEND)
    set(VLAFORGE_CONFIG_HAS_TORCH ON)
else()
    set(VLAFORGE_CONFIG_HAS_TORCH OFF)
endif()
configure_package_config_file(
    cmake/VLAForgeRuntimeConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/VLAForgeRuntimeConfig.cmake"
    INSTALL_DESTINATION lib/cmake/VLAForgeRuntime
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/VLAForgeRuntimeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(EXPORT VLAForgeRuntimeTargets
    NAMESPACE VLAForge::
    DESTINATION lib/cmake/VLAForgeRuntime
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/VLAForgeRuntimeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/VLAForgeRuntimeConfigVersion.cmake"
    DESTINATION lib/cmake/VLAForgeRuntime
)
