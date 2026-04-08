# ============================================================================
# Operator implementations and selection tables
# ============================================================================

set(OPERATORS_SOURCES
    operator_impl_table.cpp
)

if(ENABLE_CUDA)
    list(APPEND OPERATORS_SOURCES
        linear_impl.cu
        attention_op.cu
        norm_op.cu
        activation_op.cu
        fused_gate_up_activation_op.cu
        ../../third_party/flashinfer/csrc/nv_internal/cpp/common/stringUtils.cpp
        ../../third_party/flashinfer/csrc/nv_internal/cpp/common/tllmException.cpp
        ../../third_party/flashinfer/csrc/nv_internal/cpp/common/logger.cpp
        ../../third_party/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
        kernels/int4_groupwise_gemm/int4WoqGemmCuda.cu
        kernels/int4_groupwise_gemm/int4WoQGemvCuda.cu
    )
endif()

add_library(edge_fm_operators STATIC
    ${OPERATORS_SOURCES}
)

set_target_properties(edge_fm_operators PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(edge_fm_operators PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(edge_fm_operators PUBLIC
    edge_fm_utils
    nlohmann_json::nlohmann_json
)

if(ENABLE_CUDA)
    target_compile_definitions(edge_fm_operators PRIVATE
        ENABLE_BF16
        ENABLE_FP8
    )
    target_include_directories(edge_fm_operators PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/flashinfer/csrc/nv_internal
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/flashinfer/csrc/nv_internal/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/flashinfer/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include
    )
    target_link_libraries(edge_fm_operators PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
    )
    target_include_directories(edge_fm_operators PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

install(TARGETS edge_fm_operators
    EXPORT edge_fm_targets
    ARCHIVE DESTINATION lib
)
