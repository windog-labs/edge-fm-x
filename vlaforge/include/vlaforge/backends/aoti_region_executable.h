#ifndef VLAFORGE_BACKENDS_AOTI_REGION_EXECUTABLE_H_
#define VLAFORGE_BACKENDS_AOTI_REGION_EXECUTABLE_H_

#include "vlaforge/runtime/region_executable.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Returns the CUDA AOTInductor implementation of the stable
 * VLAForgeRegionExecutable ABI. The backend is built only when
 * VLAFORGE_BUILD_AOTI_BACKEND=ON.
 */
const VLAForgeRegionExecutableApi*
vlaforge_aoti_region_executable_api(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_BACKENDS_AOTI_REGION_EXECUTABLE_H_
