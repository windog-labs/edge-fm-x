# ============================================================================
# Platform Configuration: Horizon J6M
# ============================================================================

set(PLATFORM_NAME "Horizon J6M")
set(PLATFORM_ARCH "aarch64")
set(PLATFORM_ACCELERATOR "BPU Nash-M")

set(EDGE_FM_J6M_TOOLCHAIN_ROOT
    "/arm-gnu-toolchain-12.2.rel1-x86_64-aarch64-none-linux-gnu"
    CACHE PATH
    "Horizon J6M aarch64 GCC toolchain root")
set(EDGE_FM_J6M_TARGET_TRIPLE
    "aarch64-none-linux-gnu"
    CACHE STRING
    "Horizon J6M aarch64 toolchain target triple")
set(EDGE_FM_J6M_MARCH
    "nash-m"
    CACHE STRING
    "Horizon BPU march used by model compilation helpers")

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

if(EXISTS "${EDGE_FM_J6M_TOOLCHAIN_ROOT}")
    list(APPEND CMAKE_FIND_ROOT_PATH "${EDGE_FM_J6M_TOOLCHAIN_ROOT}/${EDGE_FM_J6M_TARGET_TRIPLE}")
endif()

set(BUILD_PYTHON OFF CACHE BOOL "Python bindings are disabled for the J6M target build" FORCE)
set(ENABLE_CUDA OFF CACHE BOOL "CUDA is not used by the Horizon J6M whole-model backend" FORCE)
set(EDGE_FM_ENABLE_TRT_PLUGIN_OPS OFF)
set(TRT_LIBRARY_DIR "")
set(TRT_EDGELLM_PLUGIN_DIR "")
set(TRT_EDGELLM_PLUGIN_LIBRARY "")
set(TRT_EDGELLM_AVAILABLE FALSE)

if(EXISTS "/usr/ucp")
    set(HORIZON_DEPS_ROOT "/usr/ucp" CACHE PATH "Horizon SDK/deps root")
    set(HORIZON_LIBRARY_DIR "/usr/ucp" CACHE PATH "Horizon runtime library directory")
endif()

add_compile_definitions(
    PLATFORM_HORIZON=1
    PLATFORM_J6M=1
    PLATFORM_AARCH64=1
)

message(STATUS "Platform: ${PLATFORM_NAME}")
message(STATUS "  Architecture: ${PLATFORM_ARCH}")
message(STATUS "  Accelerator: ${PLATFORM_ACCELERATOR}")
message(STATUS "  BPU march: ${EDGE_FM_J6M_MARCH}")
message(STATUS "  Toolchain root: ${EDGE_FM_J6M_TOOLCHAIN_ROOT}")
message(STATUS "  Target triple: ${EDGE_FM_J6M_TARGET_TRIPLE}")
message(STATUS "CUDA: DISABLED")

set(EDGE_FM_ENABLE_HORIZON_RUNTIME OFF)
set(EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIRS "")
set(EDGE_FM_HORIZON_RUNTIME_LIBRARY_DIRS "")
set(EDGE_FM_HORIZON_RUNTIME_LIBRARIES "")

if(ENABLE_HORIZON_RUNTIME)
    set(_edge_fm_horizon_include_hints "")
    set(_edge_fm_horizon_library_hints "")
    if(HORIZON_DEPS_ROOT)
        list(APPEND _edge_fm_horizon_include_hints
            "${HORIZON_DEPS_ROOT}/ucp/include"
            "${HORIZON_DEPS_ROOT}/include"
        )
        list(APPEND _edge_fm_horizon_library_hints
            "${HORIZON_DEPS_ROOT}/ucp/lib"
            "${HORIZON_DEPS_ROOT}/appsdk/appuser/lib"
        )
    endif()
    if(HORIZON_INCLUDE_DIR)
        list(APPEND _edge_fm_horizon_include_hints "${HORIZON_INCLUDE_DIR}")
    endif()
    if(HORIZON_LIBRARY_DIR)
        list(APPEND _edge_fm_horizon_library_hints "${HORIZON_LIBRARY_DIR}")
    endif()

    find_path(EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIR
        NAMES hobot/dnn/hb_dnn.h
        HINTS ${_edge_fm_horizon_include_hints}
    )

    if(NOT EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR
            "ENABLE_HORIZON_RUNTIME=ON but Horizon headers were not found. "
            "Set HORIZON_DEPS_ROOT or HORIZON_INCLUDE_DIR.")
    endif()

    list(APPEND EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIRS "${EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIR}")
    if(HORIZON_LIBRARIES)
        set(EDGE_FM_HORIZON_RUNTIME_LIBRARIES ${HORIZON_LIBRARIES})
    else()
        set(_edge_fm_horizon_required_libraries dnn hbucp)
        set(_edge_fm_horizon_optional_libraries
            hbrt4
            hbtl
            hb_arm_rpc
            hbmem
            hlog_wrapper
            bpu
            cjson
            alog
            hbipcfhal
            jsoncpp
            vdsp
            perfetto_sdk
        )
        foreach(_edge_fm_horizon_library_name IN LISTS
                _edge_fm_horizon_required_libraries
                _edge_fm_horizon_optional_libraries)
            string(TOUPPER "${_edge_fm_horizon_library_name}" _edge_fm_horizon_library_var_suffix)
            string(REPLACE "_" "-" _edge_fm_horizon_library_dash_name "${_edge_fm_horizon_library_name}")
            unset(EDGE_FM_HORIZON_${_edge_fm_horizon_library_var_suffix}_LIBRARY CACHE)
            find_library(EDGE_FM_HORIZON_${_edge_fm_horizon_library_var_suffix}_LIBRARY
                NAMES
                    ${_edge_fm_horizon_library_name}
                    ${_edge_fm_horizon_library_dash_name}
                    lib${_edge_fm_horizon_library_name}.so
                    lib${_edge_fm_horizon_library_name}.so.1
                    lib${_edge_fm_horizon_library_name}.so.2
                HINTS ${_edge_fm_horizon_library_hints}
            )
            if(EDGE_FM_HORIZON_${_edge_fm_horizon_library_var_suffix}_LIBRARY)
                list(APPEND EDGE_FM_HORIZON_RUNTIME_LIBRARIES
                    "${EDGE_FM_HORIZON_${_edge_fm_horizon_library_var_suffix}_LIBRARY}")
            else()
                list(FIND _edge_fm_horizon_required_libraries "${_edge_fm_horizon_library_name}" _edge_fm_horizon_required_index)
                if(NOT _edge_fm_horizon_required_index EQUAL -1)
                    message(FATAL_ERROR
                        "ENABLE_HORIZON_RUNTIME=ON but lib${_edge_fm_horizon_library_name}.so was not found. "
                        "Set HORIZON_DEPS_ROOT, HORIZON_LIBRARY_DIR, or HORIZON_LIBRARIES.")
                endif()
            endif()
        endforeach()
    endif()

    foreach(_edge_fm_horizon_library_dir IN LISTS _edge_fm_horizon_library_hints)
        if(EXISTS "${_edge_fm_horizon_library_dir}")
            list(APPEND EDGE_FM_HORIZON_RUNTIME_LIBRARY_DIRS "${_edge_fm_horizon_library_dir}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES EDGE_FM_HORIZON_RUNTIME_LIBRARY_DIRS)
    if(EDGE_FM_HORIZON_RUNTIME_LIBRARY_DIRS)
        link_directories(${EDGE_FM_HORIZON_RUNTIME_LIBRARY_DIRS})
    endif()

    set(EDGE_FM_ENABLE_HORIZON_RUNTIME ON)
    message(STATUS "Horizon runtime backend: ENABLED")
    message(STATUS "Horizon include: ${EDGE_FM_HORIZON_RUNTIME_INCLUDE_DIR}")
    message(STATUS "Horizon libraries: ${EDGE_FM_HORIZON_RUNTIME_LIBRARIES}")
else()
    message(STATUS "Horizon runtime backend: DISABLED")
endif()
