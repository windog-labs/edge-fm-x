# ============================================================================
# Python 绑定模块
# ============================================================================
if(TARGET pybind11::module)
    # pybind11_add_module(edge_fm_python pybind.cpp)
    pybind11_add_module(edge_fm_python pybind_debug.cpp)
    target_link_libraries(edge_fm_python PRIVATE
        edge_fm
        edge_fm_utils
        pybind11::module
    )

    set_target_properties(edge_fm_python PROPERTIES
        OUTPUT_NAME "edge_fm"
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH FALSE
        BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    )

    install(TARGETS edge_fm_python
        LIBRARY DESTINATION python
        RUNTIME DESTINATION python
    )
else()
    message(WARNING "pybind11 not available. Python module will not be built.")
endif()

# ============================================================================
# TRT-Edge-LLM Python 绑定 (可选，用于 in-process benchmark)
# 需先: bash tests/scripts/setup_trt_edgellm_benchmark.sh
# ============================================================================
if(BUILD_TRT_EDGELLM_PYBIND AND TARGET pybind11::module AND TRT_EDGELLM_AVAILABLE)
    file(GLOB_RECURSE TRT_EDGELLM_BUILD_DEPS CONFIGURE_DEPENDS
        ${TRT_EDGELLM_ROOT}/cpp/*.cpp
        ${TRT_EDGELLM_ROOT}/cpp/*.cu
        ${TRT_EDGELLM_ROOT}/cpp/*.h
        ${TRT_EDGELLM_ROOT}/examples/utils/*.cpp
        ${TRT_EDGELLM_ROOT}/examples/utils/*.cu
        ${TRT_EDGELLM_ROOT}/examples/utils/*.h
        ${TRT_EDGELLM_ROOT}/CMakeLists.txt
        ${TRT_EDGELLM_ROOT}/cpp/CMakeLists.txt
        ${TRT_EDGELLM_ROOT}/examples/utils/CMakeLists.txt
    )
    add_custom_command(
        OUTPUT ${TRT_EDGELLM_CORE_LIB} ${TRT_EDGELLM_TOKENIZER_LIB} ${TRT_EDGELLM_UTILS_LIB}
        COMMAND ${CMAKE_COMMAND} --build ${TRT_EDGELLM_BUILD_DIR} --parallel --target edgellmCore edgellmTokenizer exampleUtils
        DEPENDS ${TRT_EDGELLM_BUILD_DEPS}
        COMMENT "Rebuilding TensorRT-Edge-LLM runtime libraries for edge_fm_trt"
        VERBATIM
    )
    add_custom_target(edge_fm_trt_runtime_libs
        DEPENDS ${TRT_EDGELLM_CORE_LIB} ${TRT_EDGELLM_TOKENIZER_LIB} ${TRT_EDGELLM_UTILS_LIB}
    )

    set(_edge_fm_cuda_driver_hints
        ${CUDAToolkit_LIBRARY_DIR}
        ${CUDAToolkit_LIBRARY_ROOT}
        ${CUDAToolkit_TARGET_DIR}
        /usr/local/cuda
        /usr/local/cuda-${CUDAToolkit_VERSION}
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
    )
    find_library(
        CUDA_DRIVER_LIB
        cuda
        HINTS ${_edge_fm_cuda_driver_hints}
        PATH_SUFFIXES
            lib
            lib64
            lib/stubs
            targets/aarch64-linux/lib/stubs
            targets/x86_64-linux/lib/stubs
            compat
    )
    if(NOT CUDA_DRIVER_LIB)
        set(CUDA_DRIVER_LIB "cuda")  # Let linker find it
    endif()

    pybind11_add_module(edge_fm_trt_python pybind_trt_runtime.cpp)
    add_dependencies(edge_fm_trt_python edge_fm_trt_runtime_libs)
    target_include_directories(edge_fm_trt_python PRIVATE
        ${TRT_EDGELLM_ROOT}/cpp
        ${TRT_EDGELLM_ROOT}/examples
        ${TRT_EDGELLM_ROOT}/examples/multimodal
        ${TRT_EDGELLM_ROOT}/3rdParty/nlohmannJson/include
        ${TRT_EDGELLM_ROOT}/3rdParty/stb
        ${TRT_INCLUDE_DIR}
    )
    if(CUDAToolkit_FOUND)
        target_include_directories(edge_fm_trt_python PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    target_link_libraries(edge_fm_trt_python PRIVATE
        ${TRT_EDGELLM_UTILS_LIB}
        ${TRT_EDGELLM_CORE_LIB}
        ${TRT_EDGELLM_TOKENIZER_LIB}
        ${TRT_NVINFER_LIBRARY}
        CUDA::cudart
        ${CUDA_DRIVER_LIB}
        dl
        pybind11::module
    )
    # RPATH: TensorRT + CUDA (需与 TRT 构建时的 CUDA 版本一致)
    set(_trt_rpath "${TRT_LIBRARY_DIR}")
    if(CUDAToolkit_ROOT)
        list(APPEND _trt_rpath "${CUDAToolkit_ROOT}/lib64")
    elseif(CUDAToolkit_LIBRARY_DIR)
        list(APPEND _trt_rpath "${CUDAToolkit_LIBRARY_DIR}")
    endif()
    foreach(_cuda_driver_dir
        "/usr/local/cuda-${CUDAToolkit_VERSION}/compat"
        "/usr/local/cuda/compat"
        "/usr/lib/aarch64-linux-gnu/nvidia"
        "/usr/lib/aarch64-linux-gnu/tegra"
        "/usr/lib/aarch64-linux-gnu"
        "/usr/local/cuda/targets/aarch64-linux/lib/stubs"
        "/usr/local/cuda/targets/x86_64-linux/lib/stubs")
        if(EXISTS "${_cuda_driver_dir}")
            list(APPEND _trt_rpath "${_cuda_driver_dir}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _trt_rpath)
    set_target_properties(edge_fm_trt_python PROPERTIES
        OUTPUT_NAME "edge_fm_trt"
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${_trt_rpath}"
        BUILD_WITH_INSTALL_RPATH FALSE
        BUILD_RPATH "${CMAKE_BINARY_DIR}/lib:${_trt_rpath}"
    )
    install(TARGETS edge_fm_trt_python
        LIBRARY DESTINATION python
        RUNTIME DESTINATION python
    )
    message(STATUS "edge_fm_trt Python module will be built")
endif()
