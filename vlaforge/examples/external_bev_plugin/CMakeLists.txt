cmake_minimum_required(VERSION 3.18)
project(vlaforge_external_bev_plugin LANGUAGES CXX)

if(NOT DEFINED VLAFORGE_INCLUDE_DIR)
  message(FATAL_ERROR "set VLAFORGE_INCLUDE_DIR to VLAForge/include")
endif()
if(NOT DEFINED VLAFORGE_EXPECTED_SCHEMA_DIGEST)
  message(FATAL_ERROR "set VLAFORGE_EXPECTED_SCHEMA_DIGEST")
endif()

add_library(vlaforge_external_bev_plugin SHARED plugin.cpp)
target_include_directories(vlaforge_external_bev_plugin PRIVATE
    "${VLAFORGE_INCLUDE_DIR}")
target_compile_features(vlaforge_external_bev_plugin PRIVATE cxx_std_17)
target_compile_definitions(vlaforge_external_bev_plugin PRIVATE
    VLAFORGE_EXPECTED_SCHEMA_DIGEST="${VLAFORGE_EXPECTED_SCHEMA_DIGEST}")
target_compile_options(vlaforge_external_bev_plugin PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
set_target_properties(vlaforge_external_bev_plugin PROPERTIES
    CXX_EXTENSIONS OFF
    OUTPUT_NAME "vlaforge_external_bev_plugin")
