# ============================================================================
# Layers 模块
# ============================================================================
# layer
add_library(layer STATIC layer.cpp)
set_target_properties(layer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(layer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(layer PUBLIC
    nlohmann_json::nlohmann_json
    CUDA::cudart
    edge_fm_utils
)
target_include_directories(layer PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

# embed_head
add_library(embed_head STATIC embed_head.cu)
set_target_properties(embed_head PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(embed_head PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(embed_head PUBLIC
    nlohmann_json::nlohmann_json
    layer
    CUDA::cudart
    edge_fm_utils
)
target_include_directories(embed_head PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

# attention
add_library(attention STATIC attention.cu)
set_target_properties(attention PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(attention PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(attention PUBLIC layer)

# layernorm
add_library(layernorm STATIC layernorm.cu)
set_target_properties(layernorm PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(layernorm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(layernorm PRIVATE
    ENABLE_BF16
    ENABLE_FP8
)
target_link_libraries(layernorm PUBLIC layer)

# activation
add_library(activation STATIC activation.cu)
set_target_properties(activation PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(activation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(activation PUBLIC layer)

# sampler
add_library(sampler STATIC sampler.cu)
set_target_properties(sampler PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(sampler PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(sampler PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
if(CUDA_LIBCPP_INCLUDE_DIR)
    target_include_directories(sampler PUBLIC ${CUDA_LIBCPP_INCLUDE_DIR})
endif()
target_link_libraries(sampler PUBLIC layer)

# linear
add_library(linear STATIC linear.cu)
set_target_properties(linear PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(linear PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(linear PUBLIC
    nlohmann_json::nlohmann_json
    layer
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    edge_fm_utils
)
target_include_directories(linear PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
# Add int4GroupwiseGemmKernels source files
target_sources(linear PRIVATE
    int4GroupwiseGemmKernels/int4WoqGemmCuda.cu
    int4GroupwiseGemmKernels/int4WoQGemvCuda.cu
)

add_library(edge_fm_layers INTERFACE)
target_include_directories(edge_fm_layers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(edge_fm_layers INTERFACE
    layer
    embed_head
    attention
    layernorm
    activation
    sampler
    linear
    edge_fm_utils
)

install(TARGETS edge_fm_layers 
                layer 
                attention 
                layernorm 
                activation 
                sampler
                linear
    EXPORT edge_fm_targets
)