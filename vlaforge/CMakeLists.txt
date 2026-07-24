cmake_minimum_required(VERSION 3.18)
project(vlaforge_runtime VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

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
    runtime/action_queue.cpp
    runtime/region_executable.c
    runtime/static_arena.cpp
    runtime/state_store.cpp
    runtime/transaction.cpp
)
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
install(EXPORT VLAForgeRuntimeTargets
    NAMESPACE VLAForge::
    DESTINATION lib/cmake/VLAForgeRuntime
)
