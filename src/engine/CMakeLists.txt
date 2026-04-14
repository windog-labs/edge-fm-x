# ============================================================================
# Engine 模块
# ============================================================================

file(GLOB_RECURSE ENGINE_SOURCES "*.cpp")
if(ENABLE_CUDA)
    file(GLOB_RECURSE ENGINE_CUDA_SOURCES "*.cu")
    list(APPEND ENGINE_SOURCES ${ENGINE_CUDA_SOURCES})
endif()

add_library(edge_fm_engine STATIC
    ${ENGINE_SOURCES}
)

set_target_properties(edge_fm_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(edge_fm_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(edge_fm_engine PUBLIC
    edge_fm_utils
    edge_fm_backends
    edge_fm_operators
    edge_fm_layers
)

# Define install prefix for finding config files at runtime
target_compile_definitions(edge_fm_engine PRIVATE
    EDGE_FM_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
)

if(ENABLE_CUDA)
    target_link_libraries(edge_fm_engine PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )
    target_include_directories(edge_fm_engine PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    target_link_libraries(edge_fm_engine PUBLIC ${CUDNN_LIBRARY})
    target_include_directories(edge_fm_engine PUBLIC ${CUDNN_INCLUDE_DIR})
endif()

install(TARGETS edge_fm_engine
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
)
