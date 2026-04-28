# ============================================================================
# Backend-specific modules
# ============================================================================

add_library(edge_fm_backends STATIC
    backend_artifact_cache.cpp
    horizon_module_emitter.cpp
    runtime_backend.cpp
)

if(EDGE_FM_ENABLE_HORIZON_RUNTIME)
    target_sources(edge_fm_backends PRIVATE
        horizon_runtime_backend.cpp
    )
    target_compile_definitions(edge_fm_backends PRIVATE EDGE_FM_ENABLE_HORIZON_RUNTIME=1)
endif()

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

if(EDGE_FM_ENABLE_HORIZON_RUNTIME)
    target_include_directories(edge_fm_backends PUBLIC ${EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIRS})
    if(EDGE_FM_HORIZON_RUNTIME_LIBRARY_DIRS)
        target_link_directories(edge_fm_backends PUBLIC ${EDGE_FM_HORIZON_RUNTIME_LIBRARY_DIRS})
        foreach(_edge_fm_horizon_library_dir IN LISTS EDGE_FM_HORIZON_RUNTIME_LIBRARY_DIRS)
            target_link_options(edge_fm_backends INTERFACE "LINKER:-rpath-link,${_edge_fm_horizon_library_dir}")
        endforeach()
    endif()
    if(EDGE_FM_HORIZON_RUNTIME_LIBRARIES)
        target_link_libraries(edge_fm_backends PUBLIC ${EDGE_FM_HORIZON_RUNTIME_LIBRARIES})
    endif()
endif()

install(TARGETS edge_fm_backends
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
)
