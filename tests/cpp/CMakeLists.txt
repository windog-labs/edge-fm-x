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
