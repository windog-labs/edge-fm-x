add_executable(vlaforge_region_c_abi_smoke region_c_abi_smoke.c)
target_link_libraries(vlaforge_region_c_abi_smoke PRIVATE vlaforge_runtime)
add_test(NAME vlaforge_region_c_abi_smoke COMMAND vlaforge_region_c_abi_smoke)

add_executable(vlaforge_region_value_c_abi_smoke region_value_c_abi_smoke.c)
target_link_libraries(vlaforge_region_value_c_abi_smoke PRIVATE
    vlaforge_runtime)
add_test(NAME vlaforge_region_value_c_abi_smoke
    COMMAND vlaforge_region_value_c_abi_smoke)

add_executable(vlaforge_region_cpp_abi_smoke region_cpp_abi_smoke.cpp)
target_link_libraries(vlaforge_region_cpp_abi_smoke PRIVATE vlaforge_runtime)
add_test(NAME vlaforge_region_cpp_abi_smoke COMMAND vlaforge_region_cpp_abi_smoke)

add_executable(vlaforge_static_arena_smoke static_arena_smoke.cpp)
target_link_libraries(vlaforge_static_arena_smoke PRIVATE vlaforge_runtime)
add_test(NAME vlaforge_static_arena_smoke COMMAND vlaforge_static_arena_smoke)

add_executable(vlaforge_runtime_state_smoke runtime_state_smoke.cpp)
target_link_libraries(vlaforge_runtime_state_smoke PRIVATE vlaforge_runtime)
add_test(NAME vlaforge_runtime_state_smoke COMMAND vlaforge_runtime_state_smoke)

add_executable(vlaforge_session_c_abi_smoke session_c_abi_smoke.c)
target_link_libraries(vlaforge_session_c_abi_smoke PRIVATE vlaforge_runtime)
add_test(NAME vlaforge_session_c_abi_smoke COMMAND vlaforge_session_c_abi_smoke)

add_executable(vlaforge_artifact_verifier_smoke artifact_verifier_smoke.cpp)
target_link_libraries(vlaforge_artifact_verifier_smoke PRIVATE
    vlaforge_runtime)
add_test(NAME vlaforge_artifact_verifier_smoke
    COMMAND vlaforge_artifact_verifier_smoke)

function(vlaforge_add_external_plugin_fixture target)
    add_library(${target} SHARED
        ../../examples/external_bev_plugin/plugin.cpp)
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
    target_compile_definitions(${target} PRIVATE
        "VLAFORGE_EXPECTED_SCHEMA_DIGEST=\"0000000000000000000000000000000000000000000000000000000000000000\""
        ${ARGN})
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()

vlaforge_add_external_plugin_fixture(
    vlaforge_external_region_test_plugin)
vlaforge_add_external_plugin_fixture(
    vlaforge_external_region_bad_abi_plugin
    VLAFORGE_PLUGIN_BAD_ABI=1)
vlaforge_add_external_plugin_fixture(
    vlaforge_external_region_missing_symbol_plugin
    VLAFORGE_PLUGIN_OMIT_ENTRYPOINT=1)

add_executable(vlaforge_external_region_plugin_smoke
    external_region_plugin_smoke.cpp)
target_link_libraries(vlaforge_external_region_plugin_smoke PRIVATE
    vlaforge_runtime)
add_test(NAME vlaforge_external_region_plugin_smoke
    COMMAND vlaforge_external_region_plugin_smoke
        $<TARGET_FILE:vlaforge_external_region_test_plugin>
        $<TARGET_FILE:vlaforge_external_region_bad_abi_plugin>
        $<TARGET_FILE:vlaforge_external_region_missing_symbol_plugin>)

if(TARGET vlaforge_aoti_backend)
    add_executable(vlaforge_cuda_state_store_smoke
        cuda_state_store_smoke.cpp)
    target_link_libraries(vlaforge_cuda_state_store_smoke PRIVATE
        vlaforge_runtime)
    add_test(NAME vlaforge_cuda_state_store_smoke
        COMMAND vlaforge_cuda_state_store_smoke)

    add_executable(vlaforge_aoti_region_smoke aoti_region_smoke.cpp)
    target_link_libraries(vlaforge_aoti_region_smoke PRIVATE
        vlaforge_aoti_backend)
    target_compile_options(vlaforge_aoti_region_smoke PRIVATE
        ${VLAFORGE_TORCH_CXX_FLAGS})
    if(MSVC)
        target_compile_options(vlaforge_aoti_region_smoke PRIVATE /W4 /WX)
    else()
        target_compile_options(vlaforge_aoti_region_smoke PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(TARGET vlaforge_torchscript_backend)
    add_executable(vlaforge_torchscript_region_smoke
        torchscript_region_smoke.cpp)
    target_link_libraries(vlaforge_torchscript_region_smoke PRIVATE
        vlaforge_torchscript_backend)
    target_compile_options(vlaforge_torchscript_region_smoke PRIVATE
        ${VLAFORGE_TORCH_CXX_FLAGS})
    if(MSVC)
        target_compile_options(vlaforge_torchscript_region_smoke PRIVATE
            /W4 /WX)
    else()
        target_compile_options(vlaforge_torchscript_region_smoke PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(TARGET vlaforge_tensorrt_backend)
    add_executable(vlaforge_tensorrt_region_contract_smoke
        tensorrt_region_contract_smoke.cpp)
    target_link_libraries(vlaforge_tensorrt_region_contract_smoke PRIVATE
        vlaforge_tensorrt_backend)
    if(MSVC)
        target_compile_options(vlaforge_tensorrt_region_contract_smoke
            PRIVATE /W4 /WX)
    else()
        target_compile_options(vlaforge_tensorrt_region_contract_smoke
            PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_executable(vlaforge_tensorrt_region_on_device_smoke
        tensorrt_region_on_device_smoke.cpp)
    target_link_libraries(vlaforge_tensorrt_region_on_device_smoke PRIVATE
        vlaforge_tensorrt_backend)
    if(MSVC)
        target_compile_options(vlaforge_tensorrt_region_on_device_smoke
            PRIVATE /W4 /WX)
    else()
        target_compile_options(vlaforge_tensorrt_region_on_device_smoke
            PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    if(NOT VLAFORGE_TENSORRT_DRIVERLESS_COMPILE_ONLY)
        add_test(NAME vlaforge_tensorrt_region_contract_smoke
            COMMAND vlaforge_tensorrt_region_contract_smoke)
        add_test(NAME vlaforge_tensorrt_region_on_device_smoke
            COMMAND vlaforge_tensorrt_region_on_device_smoke)
    endif()
endif()
