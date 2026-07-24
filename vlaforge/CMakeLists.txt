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

add_library(vlaforge_runtime STATIC
    runtime/region_executable.c
    runtime/static_arena.cpp
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

if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()

install(TARGETS vlaforge_runtime
    EXPORT VLAForgeRuntimeTargets
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT VLAForgeRuntimeTargets
    NAMESPACE VLAForge::
    DESTINATION lib/cmake/VLAForgeRuntime
)
