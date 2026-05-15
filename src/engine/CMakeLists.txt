# ============================================================================
# Engine 模块
# ============================================================================

set(ENGINE_SOURCES
    engine.cpp
    engine_factory.cpp
    tasks/stage_execution/mock_stage_runner.cpp
    tasks/token_generation/compact_vocab.cpp
    tasks/token_generation/kv_manager.cpp
    tasks/token_generation/scheduler.cpp
    tasks/stage_execution/stage_execution_engine.cpp
    tasks/trajectory_planning/planner_state_manager.cpp
    tasks/trajectory_planning/planner_tensor_utils.cpp
    tasks/trajectory_planning/trajectory_planner_engine.cpp
    tasks/stage_execution/horizon/horizon_engine.cpp
)

if(ENABLE_CUDA)
    list(APPEND ENGINE_SOURCES
        tasks/token_generation/cuda/scheduler.cpp
        tasks/token_generation/cuda/standard_engine.cpp
        tasks/token_generation/cuda/tuning/cuda_operator_tuner.cpp
        experimental/speculative/eagle_engine.cpp
    )
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
)
if(ENABLE_CUDA)
    target_link_libraries(edge_fm_engine PUBLIC
        edge_fm_layers
    )
endif()

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

if(ENABLE_CUDA AND CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    target_link_libraries(edge_fm_engine PUBLIC ${CUDNN_LIBRARY})
    target_include_directories(edge_fm_engine PUBLIC ${CUDNN_INCLUDE_DIR})
endif()

install(TARGETS edge_fm_engine
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
)
