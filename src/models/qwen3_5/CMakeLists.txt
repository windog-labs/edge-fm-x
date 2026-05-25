add_library(edge_fm_qwen3_5 STATIC
    qwen3_5.cpp
)

set_target_properties(edge_fm_qwen3_5 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(edge_fm_qwen3_5 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(edge_fm_qwen3_5 PUBLIC
    edge_fm_layers
    edge_fm_operators
    edge_fm_engine
    edge_fm_utils
)

install(TARGETS edge_fm_qwen3_5
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
)
