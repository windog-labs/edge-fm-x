cmake_minimum_required(VERSION 3.18)
project(vlaforge_install_consumer LANGUAGES CXX)

find_package(VLAForgeRuntime REQUIRED CONFIG)

add_executable(vlaforge_install_consumer main.cpp)
target_link_libraries(vlaforge_install_consumer PRIVATE
    VLAForge::vlaforge_runtime)
if(TARGET VLAForge::vlaforge_aoti_backend)
  target_link_libraries(vlaforge_install_consumer PRIVATE
      VLAForge::vlaforge_aoti_backend)
  target_compile_definitions(vlaforge_install_consumer PRIVATE
      VLAFORGE_CONSUMER_HAS_AOTI=1)
endif()
target_compile_features(vlaforge_install_consumer PRIVATE cxx_std_17)
