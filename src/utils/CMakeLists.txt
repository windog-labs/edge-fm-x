# ============================================================================
# Utils 模块
# ============================================================================

set(UTILS_SOURCES
    backend_artifact_cache.cpp
    horizon_module_emitter.cpp
    logging.cpp
)

if(ENABLE_CUDA)
    list(APPEND UTILS_SOURCES device/weight_loader.cpp device/cuda_graph.cpp device/tuning_cache.cpp)
endif()

add_library(edge_fm_utils STATIC ${UTILS_SOURCES})
set_target_properties(edge_fm_utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(edge_fm_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(edge_fm_utils PUBLIC
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

if(ENABLE_CUDA)
    target_link_libraries(edge_fm_utils PUBLIC 
        CUDA::cudart
        CUDA::nvToolsExt
        safetensors_cpp
    )
endif()

install(TARGETS edge_fm_utils
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
)
