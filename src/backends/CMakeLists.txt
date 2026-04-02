# ============================================================================
# Backend-specific modules
# ============================================================================

add_library(edge_fm_backends STATIC
    backend_artifact_cache.cpp
    horizon_module_emitter.cpp
)

set_target_properties(edge_fm_backends PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(edge_fm_backends PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(edge_fm_backends PUBLIC
    edge_fm_utils
    nlohmann_json::nlohmann_json
)

install(TARGETS edge_fm_backends
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
)
