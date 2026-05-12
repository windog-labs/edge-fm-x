add_executable(edge_fm_runtime_backend_helpers_test
    test_runtime_backend_helpers.cpp
)

target_link_libraries(edge_fm_runtime_backend_helpers_test PRIVATE
    edge_fm_backends
)

if(EDGE_FM_ENABLE_HORIZON_RUNTIME)
    target_compile_definitions(edge_fm_runtime_backend_helpers_test PRIVATE
        EDGE_FM_ENABLE_HORIZON_RUNTIME=1
    )
endif()

add_test(
    NAME edge_fm_runtime_backend_helpers
    COMMAND edge_fm_runtime_backend_helpers_test
)

add_executable(edge_fm_tensor_cpu_test
    test_tensor_cpu.cpp
)

target_link_libraries(edge_fm_tensor_cpu_test PRIVATE
    edge_fm
)

add_test(
    NAME edge_fm_tensor_cpu
    COMMAND edge_fm_tensor_cpu_test
)

add_executable(edge_fm_compact_vocab_test
    test_compact_vocab.cpp
    compact_vocab_cuda_stubs.cpp
    ../../src/engine/compact_vocab.cpp
    ../../src/engine/engine.cpp
    ../../src/tensor.cpp
)

target_link_libraries(edge_fm_compact_vocab_test PRIVATE
    nlohmann_json::nlohmann_json
)

target_compile_definitions(edge_fm_compact_vocab_test PRIVATE
    EDGE_FM_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
)

if(ENABLE_CUDA)
    target_link_libraries(edge_fm_compact_vocab_test PRIVATE CUDA::cudart)
endif()

add_test(
    NAME edge_fm_compact_vocab
    COMMAND edge_fm_compact_vocab_test
)

add_executable(edge_fm_smolvla_horizon_smoke
    smolvla_horizon_prefill_decode_smoke.cpp
)

target_link_libraries(edge_fm_smolvla_horizon_smoke PRIVATE
    edge_fm
)

add_executable(edge_fm_smolvla_horizon_benchmark
    smolvla_horizon_benchmark.cpp
)

target_link_libraries(edge_fm_smolvla_horizon_benchmark PRIVATE
    edge_fm
)
