# ============================================================================
# Models 模块（聚合模块）
# ============================================================================

# Model 基类库
add_library(edge_fm_model STATIC
    model.cpp
    model_description.cpp
    layered_transformer.cpp
)

set_target_properties(edge_fm_model PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(edge_fm_model PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(edge_fm_model PUBLIC
    edge_fm_utils
    edge_fm_engine
)

install(TARGETS edge_fm_model
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
)

# Models 聚合接口库
add_library(edge_fm_models INTERFACE)

target_include_directories(edge_fm_models INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(edge_fm_models INTERFACE
    edge_fm_model
    edge_fm_utils
    edge_fm_layers
    edge_fm_engine
)

install(TARGETS edge_fm_models
    EXPORT edge_fm_targets
)
