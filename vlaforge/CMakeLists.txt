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
set(VLAFORGE_AOTI_PACKAGE_EXTRACTION_ROOT "" CACHE STRING
    "Explicit private AOTI extraction root; empty retains the upstream loader")
if(NOT VLAFORGE_AOTI_PACKAGE_EXTRACTION_ROOT STREQUAL "")
    if(NOT VLAFORGE_BUILD_AOTI_BACKEND OR NOT UNIX OR APPLE)
        message(FATAL_ERROR "AOTI extraction root requires the Linux AOTI backend")
    endif()
    if(NOT IS_ABSOLUTE "${VLAFORGE_AOTI_PACKAGE_EXTRACTION_ROOT}" OR
       VLAFORGE_AOTI_PACKAGE_EXTRACTION_ROOT MATCHES "[\"\\\\;\n\r\t$]")
        message(FATAL_ERROR "AOTI extraction root must be an absolute literal path")
    endif()
endif()
option(VLAFORGE_BUILD_TORCHSCRIPT_BACKEND
       "Build the optional TorchScript RegionExecutable backend" OFF)
option(VLAFORGE_TORCHSCRIPT_ENABLE_CUDA
       "Enable CUDA and optional shared-context TorchScript variants" OFF)
option(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM
       "Experimental LibTorch 2.10 owned graph-pool reclaim after checked destruction" OFF)
option(VLAFORGE_BUILD_TENSORRT_BACKEND
       "Build the optional CUDA TensorRT RegionExecutable backend" OFF)
option(VLAFORGE_TENSORRT_STANDARD_PLUGINS
       "Register installed standard TensorRT plugins before loading engines" OFF)
option(VLAFORGE_TENSORRT_BFLOAT16_PLUGIN
       "Build/register the optional TensorRT 8 FLOAT-carrier BF16 rounding plugin" OFF)
option(VLAFORGE_BUILD_HF_TOKENIZER "Build the optional native Hugging Face tokenizer bridge" OFF)
option(VLAFORGE_BUILD_PILLOW_IMAGE "Build the optional native image preprocessing bridge" OFF)
set(VLAFORGE_PILLOW_CORE_LIBRARY "" CACHE FILEPATH "Prebuilt standalone pinned Pillow imaging core")
set(VLAFORGE_PILLOW_CORE_INCLUDE "" CACHE PATH "Matching pinned imaging core headers")
set(VLAFORGE_PILLOW_PYTHON_INCLUDE "" CACHE PATH "Compile-time platform headers for the standalone image C library")
set(VLAFORGE_HF_TOKENIZER_LIBRARY "" CACHE FILEPATH "Prebuilt native Rust tokenizer static library")
option(VLAFORGE_TENSORRT_DRIVERLESS_COMPILE_ONLY
       "Do not register TensorRT runtime tests in a driverless L4T container"
       OFF)
if(VLAFORGE_TENSORRT_STANDARD_PLUGINS AND NOT VLAFORGE_BUILD_TENSORRT_BACKEND)
    message(FATAL_ERROR "VLAFORGE_TENSORRT_STANDARD_PLUGINS requires the TensorRT backend")
endif()
if(VLAFORGE_BUILD_PILLOW_IMAGE)
    if(NOT IS_ABSOLUTE "${VLAFORGE_PILLOW_CORE_LIBRARY}" OR
       NOT EXISTS "${VLAFORGE_PILLOW_CORE_LIBRARY}" OR
       NOT IS_ABSOLUTE "${VLAFORGE_PILLOW_CORE_INCLUDE}" OR
       NOT EXISTS "${VLAFORGE_PILLOW_CORE_INCLUDE}/Imaging.h" OR
       NOT IS_ABSOLUTE "${VLAFORGE_PILLOW_PYTHON_INCLUDE}" OR
       NOT EXISTS "${VLAFORGE_PILLOW_PYTHON_INCLUDE}/pyconfig.h")
        message(FATAL_ERROR "Set the matching standalone Pillow core archive and headers")
    endif()
    find_package(Threads REQUIRED)
    add_library(vlaforge_pillow_image_backend SHARED
        backends/pillow_image.cpp backends/pillow_image_c.c)
    target_include_directories(vlaforge_pillow_image_backend PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_include_directories(vlaforge_pillow_image_backend SYSTEM PRIVATE
        "${VLAFORGE_PILLOW_CORE_INCLUDE}" "${VLAFORGE_PILLOW_PYTHON_INCLUDE}")
    target_compile_definitions(vlaforge_pillow_image_backend PRIVATE
        VLAFORGE_PILLOW_IMAGE_BUILD=1 HAVE_PROTOTYPES=1 STDC_HEADERS=1)
    set_target_properties(vlaforge_pillow_image_backend PROPERTIES
        C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    target_link_libraries(vlaforge_pillow_image_backend PRIVATE
        "${VLAFORGE_PILLOW_CORE_LIBRARY}" Threads::Threads)
    if(UNIX AND NOT APPLE)
        target_link_libraries(vlaforge_pillow_image_backend PRIVATE m)
    endif()
    if(MSVC)
        target_compile_options(vlaforge_pillow_image_backend PRIVATE /W4 /WX)
    else()
        target_compile_options(vlaforge_pillow_image_backend PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()
if(VLAFORGE_TENSORRT_BFLOAT16_PLUGIN AND NOT VLAFORGE_BUILD_TENSORRT_BACKEND)
    message(FATAL_ERROR "VLAFORGE_TENSORRT_BFLOAT16_PLUGIN requires the TensorRT backend")
endif()

if(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM AND NOT
   (VLAFORGE_BUILD_AOTI_BACKEND OR
    (VLAFORGE_BUILD_TORCHSCRIPT_BACKEND AND VLAFORGE_TORCHSCRIPT_ENABLE_CUDA)))
    message(FATAL_ERROR "VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM requires the LibTorch CUDA graph backend")
endif()

add_library(vlaforge_runtime STATIC
    runtime/artifact_verifier.cpp
    runtime/bounded_replay.cpp
    runtime/cuda_graph.cpp
    runtime/device_copy.cpp
    runtime/execution_context.cpp
    runtime/external_region_plugin.cpp
    runtime/host_preprocessing.cpp
    runtime/numerical.cpp
    runtime/region_executable.c
    runtime/session_c.c
    runtime/static_arena.cpp
    runtime/state_store.cpp
    runtime/transaction.cpp
)
target_link_libraries(vlaforge_runtime PUBLIC OpenSSL::Crypto ${CMAKE_DL_LIBS})
target_include_directories(vlaforge_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
if(VLAFORGE_BUILD_HF_TOKENIZER)
    if(NOT IS_ABSOLUTE "${VLAFORGE_HF_TOKENIZER_LIBRARY}" OR
       NOT EXISTS "${VLAFORGE_HF_TOKENIZER_LIBRARY}")
        message(FATAL_ERROR "Set VLAFORGE_HF_TOKENIZER_LIBRARY to the native Rust static library")
    endif()
    find_package(Threads REQUIRED)
    add_library(vlaforge_hf_tokenizer_backend STATIC backends/huggingface_tokenizer.cpp)
    target_link_libraries(vlaforge_hf_tokenizer_backend PUBLIC vlaforge_runtime
        "${VLAFORGE_HF_TOKENIZER_LIBRARY}" Threads::Threads ${CMAKE_DL_LIBS})
    if(UNIX AND NOT APPLE)
        target_link_libraries(vlaforge_hf_tokenizer_backend PUBLIC m)
    endif()
endif()
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
    if(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM AND
       NOT Torch_VERSION MATCHES "^2\\.10(\\.|$)")
        message(FATAL_ERROR "VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM requires audited LibTorch 2.10; got ${Torch_VERSION}")
    endif()
    separate_arguments(VLAFORGE_TORCH_CXX_FLAGS
        NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    set_target_properties(vlaforge_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(vlaforge_libtorch_numerical_backend SHARED backends/libtorch_numerical.cpp)
    target_include_directories(vlaforge_libtorch_numerical_backend PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(vlaforge_libtorch_numerical_backend PUBLIC vlaforge_runtime ${TORCH_LIBRARIES})
    target_compile_options(vlaforge_libtorch_numerical_backend PRIVATE ${VLAFORGE_TORCH_CXX_FLAGS})
    if(MSVC)
        target_compile_options(vlaforge_libtorch_numerical_backend PRIVATE /W4 /WX)
    else()
        target_compile_options(vlaforge_libtorch_numerical_backend PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VLAFORGE_BUILD_AOTI_BACKEND OR
   (VLAFORGE_BUILD_TORCHSCRIPT_BACKEND AND VLAFORGE_TORCHSCRIPT_ENABLE_CUDA))
    find_package(CUDAToolkit REQUIRED)
    add_library(vlaforge_libtorch_graph_backend STATIC backends/libtorch_graph.cpp)
    target_include_directories(vlaforge_libtorch_graph_backend PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(vlaforge_libtorch_graph_backend PUBLIC
        vlaforge_runtime ${TORCH_LIBRARIES} CUDA::cudart)
    target_compile_options(vlaforge_libtorch_graph_backend PRIVATE ${VLAFORGE_TORCH_CXX_FLAGS})
    if(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM)
        target_compile_definitions(vlaforge_libtorch_graph_backend PRIVATE
            VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM=1)
    endif()
    if(MSVC)
        target_compile_options(vlaforge_libtorch_graph_backend PRIVATE /W4 /WX)
    else()
        target_compile_options(vlaforge_libtorch_graph_backend PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VLAFORGE_BUILD_AOTI_BACKEND)
    find_package(CUDAToolkit REQUIRED)
    if(NOT VLAFORGE_AOTI_PACKAGE_EXTRACTION_ROOT STREQUAL "" AND
       NOT Torch_VERSION MATCHES "^2\\.10(\\.|$)")
        message(FATAL_ERROR "AOTI extraction root requires audited LibTorch 2.10")
    endif()
    configure_file(backends/aoti_package_config.h.in
        generated/aoti_package_config.h @ONLY)
    target_compile_definitions(vlaforge_runtime
        PUBLIC VLAFORGE_ENABLE_CUDA_ARENA=1)
    target_link_libraries(vlaforge_runtime PUBLIC CUDA::cudart)
    add_library(vlaforge_aoti_backend STATIC
        backends/aoti_callable.cpp
        backends/aoti_extracted_package.cpp
        backends/aoti_materialized_package.cpp
        backends/aoti_graph.cpp
        backends/aoti_region_executable.cpp
        backends/aoti_sequence_runner.cpp
    )
    target_include_directories(vlaforge_aoti_backend
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(vlaforge_aoti_backend
        PUBLIC vlaforge_runtime vlaforge_libtorch_graph_backend vlaforge_libtorch_numerical_backend ${TORCH_LIBRARIES}
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

if(VLAFORGE_BUILD_TENSORRT_BACKEND)
    find_package(CUDAToolkit REQUIRED)
    find_path(VLAFORGE_TENSORRT_INCLUDE_DIR
        NAMES NvInfer.h
        PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu)
    find_library(VLAFORGE_TENSORRT_LIBRARY
        NAMES nvinfer
        PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu)
    if(NOT VLAFORGE_TENSORRT_INCLUDE_DIR OR
       NOT VLAFORGE_TENSORRT_LIBRARY)
        message(FATAL_ERROR
            "VLAFORGE_BUILD_TENSORRT_BACKEND requires TensorRT headers "
            "and libnvinfer")
    endif()
    if(NOT TARGET TensorRT::nvinfer)
        add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_LOCATION "${VLAFORGE_TENSORRT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${VLAFORGE_TENSORRT_INCLUDE_DIR}")
    endif()
    target_compile_definitions(vlaforge_runtime
        PUBLIC VLAFORGE_ENABLE_CUDA_ARENA=1)
    target_link_libraries(vlaforge_runtime PUBLIC CUDA::cudart)
    add_library(vlaforge_tensorrt_backend STATIC
        backends/tensorrt_region_executable.cpp
    )
    target_include_directories(vlaforge_tensorrt_backend
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(vlaforge_tensorrt_backend
        PUBLIC vlaforge_runtime TensorRT::nvinfer CUDA::cudart
    )
    if(VLAFORGE_TENSORRT_BFLOAT16_PLUGIN)
        enable_language(CUDA)
        add_library(vlaforge_tensorrt_precision_plugins SHARED
            backends/tensorrt_precision_plugins.cu)
        set_target_properties(vlaforge_tensorrt_precision_plugins PROPERTIES
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        target_include_directories(vlaforge_tensorrt_precision_plugins PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        target_link_libraries(vlaforge_tensorrt_precision_plugins PUBLIC
            TensorRT::nvinfer CUDA::cudart)
        target_link_libraries(vlaforge_tensorrt_precision_plugins PRIVATE CUDA::cublas)
        target_link_libraries(vlaforge_tensorrt_backend PUBLIC vlaforge_tensorrt_precision_plugins)
        target_compile_definitions(vlaforge_tensorrt_backend PRIVATE VLAFORGE_TENSORRT_BFLOAT16_PLUGIN=1)
    endif()
    if(VLAFORGE_TENSORRT_STANDARD_PLUGINS)
        find_library(VLAFORGE_TENSORRT_PLUGIN_LIBRARY
            NAMES nvinfer_plugin
            PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu)
        if(NOT VLAFORGE_TENSORRT_PLUGIN_LIBRARY OR
           NOT EXISTS "${VLAFORGE_TENSORRT_INCLUDE_DIR}/NvInferPlugin.h")
            message(FATAL_ERROR
                "VLAFORGE_TENSORRT_STANDARD_PLUGINS requires NvInferPlugin.h and libnvinfer_plugin")
        endif()
        if(NOT TARGET TensorRT::nvinfer_plugin)
            add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
            set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
                IMPORTED_LOCATION "${VLAFORGE_TENSORRT_PLUGIN_LIBRARY}")
        endif()
        target_link_libraries(vlaforge_tensorrt_backend PUBLIC
            TensorRT::nvinfer_plugin)
        target_compile_definitions(vlaforge_tensorrt_backend PUBLIC
            VLAFORGE_TENSORRT_STANDARD_PLUGINS=1)
    endif()
    # L4T development containers intentionally omit several board driver
    # libraries that satisfy indirect TensorRT/DLA symbols on the real Orin.
    # Permit unresolved symbols in those shared dependencies while still
    # requiring every VLAForge object symbol to resolve at link time.
    if(UNIX AND NOT APPLE AND
       VLAFORGE_TENSORRT_DRIVERLESS_COMPILE_ONLY)
        target_link_options(vlaforge_tensorrt_backend INTERFACE
            "LINKER:--allow-shlib-undefined")
    endif()
    if(MSVC)
        target_compile_options(vlaforge_tensorrt_backend PRIVATE /W4 /WX)
    else()
        target_compile_options(vlaforge_tensorrt_backend PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VLAFORGE_BUILD_TORCHSCRIPT_BACKEND)
    if(VLAFORGE_TORCHSCRIPT_ENABLE_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_compile_definitions(vlaforge_runtime
            PUBLIC VLAFORGE_ENABLE_CUDA_ARENA=1)
        target_link_libraries(vlaforge_runtime PUBLIC CUDA::cudart)
    endif()
    add_library(vlaforge_torchscript_backend STATIC
        backends/torchscript_region_executable.cpp
    )
    target_include_directories(vlaforge_torchscript_backend
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(vlaforge_torchscript_backend
        PUBLIC vlaforge_runtime vlaforge_libtorch_numerical_backend ${TORCH_LIBRARIES}
    )
    target_compile_options(vlaforge_torchscript_backend
        PRIVATE ${VLAFORGE_TORCH_CXX_FLAGS}
    )
    if(VLAFORGE_TORCHSCRIPT_ENABLE_CUDA)
        target_compile_definitions(vlaforge_torchscript_backend
            PRIVATE VLAFORGE_TORCHSCRIPT_CUDA=1)
        target_link_libraries(vlaforge_torchscript_backend PUBLIC vlaforge_libtorch_graph_backend)
    endif()
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
if(TARGET vlaforge_pillow_image_backend)
    install(TARGETS vlaforge_pillow_image_backend
        EXPORT VLAForgeRuntimeTargets
        LIBRARY DESTINATION lib RUNTIME DESTINATION bin ARCHIVE DESTINATION lib)
endif()
if(TARGET vlaforge_aoti_backend)
    install(TARGETS vlaforge_aoti_backend
        EXPORT VLAForgeRuntimeTargets
        ARCHIVE DESTINATION lib
    )
endif()
if(TARGET vlaforge_libtorch_graph_backend)
    install(TARGETS vlaforge_libtorch_graph_backend
        EXPORT VLAForgeRuntimeTargets
        ARCHIVE DESTINATION lib)
endif()
if(TARGET vlaforge_libtorch_numerical_backend)
    install(TARGETS vlaforge_libtorch_numerical_backend
        EXPORT VLAForgeRuntimeTargets
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)
endif()
if(TARGET vlaforge_torchscript_backend)
    install(TARGETS vlaforge_torchscript_backend
        EXPORT VLAForgeRuntimeTargets
        ARCHIVE DESTINATION lib
    )
endif()
if(TARGET vlaforge_tensorrt_backend)
    install(TARGETS vlaforge_tensorrt_backend
        EXPORT VLAForgeRuntimeTargets
        ARCHIVE DESTINATION lib
    )
endif()
if(TARGET vlaforge_tensorrt_precision_plugins)
    install(TARGETS vlaforge_tensorrt_precision_plugins
        EXPORT VLAForgeRuntimeTargets
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)
endif()
install(DIRECTORY include/ DESTINATION include)
if(VLAFORGE_BUILD_AOTI_BACKEND OR VLAFORGE_BUILD_TENSORRT_BACKEND OR
   (VLAFORGE_BUILD_TORCHSCRIPT_BACKEND AND VLAFORGE_TORCHSCRIPT_ENABLE_CUDA))
    set(VLAFORGE_CONFIG_HAS_CUDA ON)
else()
    set(VLAFORGE_CONFIG_HAS_CUDA OFF)
endif()
set(VLAFORGE_CONFIG_HAS_TENSORRT ${VLAFORGE_BUILD_TENSORRT_BACKEND})
set(VLAFORGE_CONFIG_HAS_TENSORRT_PLUGINS ${VLAFORGE_TENSORRT_STANDARD_PLUGINS})
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

# Bundle builders consume actual project targets, not a filesystem/vendor scan.
get_property(_vlaforge_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
set(_vlaforge_runtime_libraries "")
foreach(_vlaforge_target IN LISTS _vlaforge_targets)
    get_target_property(_vlaforge_target_type ${_vlaforge_target} TYPE)
    if(_vlaforge_target_type STREQUAL "SHARED_LIBRARY")
        string(APPEND _vlaforge_runtime_libraries
            "$<TARGET_FILE:${_vlaforge_target}>\n")
        if(UNIX)
            string(APPEND _vlaforge_runtime_libraries
                "$<TARGET_SONAME_FILE:${_vlaforge_target}>\n")
        endif()
    endif()
endforeach()
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/vlaforge-runtime-libraries-$<CONFIG>.txt"
    CONTENT "${_vlaforge_runtime_libraries}")
