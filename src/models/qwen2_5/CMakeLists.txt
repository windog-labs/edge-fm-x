# ============================================================================
# Qwen2.5 模型模块
# ============================================================================

add_library(edge_fm_qwen2_5 STATIC
    qwen2_5.cpp
    trt_mlp_bridge.cpp
    trt_linear_bridge.cpp
)

set_target_properties(edge_fm_qwen2_5 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(edge_fm_qwen2_5 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(edge_fm_qwen2_5 PUBLIC
    edge_fm_utils
    edge_fm_layers
    edge_fm_engine
)

if(BUILD_TRT_MLP_BRIDGE)
    target_include_directories(edge_fm_qwen2_5 PRIVATE ${TRT_INCLUDE_DIR})
    target_link_libraries(edge_fm_qwen2_5 PUBLIC ${TRT_NVINFER_LIBRARY})
endif()

install(TARGETS edge_fm_qwen2_5
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
)
