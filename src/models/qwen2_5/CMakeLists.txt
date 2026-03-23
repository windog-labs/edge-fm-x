# ============================================================================
# Qwen2.5 模型模块
# ============================================================================

add_library(edge_fm_qwen2_5 STATIC
    qwen2_5.cpp
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

install(TARGETS edge_fm_qwen2_5
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
)

