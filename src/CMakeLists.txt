# ============================================================================
# 模块化构建
# ============================================================================
add_subdirectory(utils)
add_subdirectory(operators)
add_subdirectory(backends)
add_subdirectory(engine)
if(ENABLE_CUDA)
    add_subdirectory(layers)
    add_subdirectory(models)
endif()

# ============================================================================
# 主库 edge_fm
# ============================================================================
set(EDGE_FM_SOURCES
    edge-fm.cpp
    tensor.cpp
)

add_library(edge_fm SHARED ${EDGE_FM_SOURCES})

target_include_directories(edge_fm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(edge_fm PUBLIC
    edge_fm_utils
    edge_fm_operators
    edge_fm_backends
    edge_fm_engine
)
if(ENABLE_CUDA)
    target_link_libraries(edge_fm PUBLIC
        edge_fm_layers
        edge_fm_models
    )
endif()

# 显式链接 CUDA 运行时，避免 libedge_fm.so 出现 undefined symbol: __cudaLaunchKernel
if(ENABLE_CUDA)
    target_link_libraries(edge_fm PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
endif()

if(EDGE_FM_ENABLE_TRT_PLUGIN_OPS)
    set(_edge_fm_trt_plugin_rpath "${TRT_LIBRARY_DIR}" "${TRT_EDGELLM_PLUGIN_DIR}")
    list(REMOVE_DUPLICATES _edge_fm_trt_plugin_rpath)
    set_target_properties(edge_fm PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;${_edge_fm_trt_plugin_rpath}"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${_edge_fm_trt_plugin_rpath}"
    )
endif()

# Link dlpack if available (header-only library)
if(TARGET dlpack)
    target_link_libraries(edge_fm PUBLIC dlpack)
endif()

# ============================================================================
# Python 模块
# ============================================================================
if(BUILD_PYTHON)
    if(TARGET pybind11::module)
        add_subdirectory(python)
    else()
        message(WARNING "pybind11 not available. Python module will not be built.")
        message(WARNING "Set BUILD_PYTHON=OFF to suppress this warning.")
    endif()
endif()

# ============================================================================
# 安装配置
# ============================================================================
install(TARGETS edge_fm
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ../include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 安装 dlpack 头文件
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/dlpack/include/dlpack
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 安装默认配置文件
set(EDGE_FM_INSTALL_CONFIG_FILES
    ${EDGE_FM_ACTIVE_CONFIG_DIR}/engine_default.json
)

if(NOT PLATFORM STREQUAL "j6m")
    list(APPEND EDGE_FM_INSTALL_CONFIG_FILES
        ${EDGE_FM_ACTIVE_CONFIG_DIR}/operator_impl_table.json
        ${EDGE_FM_ACTIVE_CONFIG_DIR}/operator_impl_table_llm.json
        ${EDGE_FM_ACTIVE_CONFIG_DIR}/operator_impl_table_vlm.json
    )
endif()

install(FILES ${EDGE_FM_INSTALL_CONFIG_FILES}
    DESTINATION config
)
