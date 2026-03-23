# ============================================================================
# 模块化构建
# ============================================================================
add_subdirectory(utils)
add_subdirectory(layers)
add_subdirectory(engine)
add_subdirectory(models)

# ============================================================================
# 主库 edge_fm
# ============================================================================
add_library(edge_fm SHARED edge-fm.cpp)

target_include_directories(edge_fm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(edge_fm PUBLIC
    edge_fm_utils
    edge_fm_layers
    edge_fm_engine
    edge_fm_models
)

# 显式链接 CUDA 运行时，避免 libedge_fm.so 出现 undefined symbol: __cudaLaunchKernel
if(CUDAToolkit_FOUND)
    target_link_libraries(edge_fm PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
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
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../examples/config/engine_default.json
    DESTINATION config
)