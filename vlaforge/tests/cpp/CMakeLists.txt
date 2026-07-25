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

if(TARGET vlaforge_aoti_backend)
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
