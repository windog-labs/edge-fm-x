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

/*
 * Production callable ABI. Tensor values are borrowed until run returns.
 * Scalar values are rejected because an AOTI TensorRegion callable accepts
 * tensors only; host scalars must be tensorized by the generated Session or a
 * preprocessing Region.
 */
const VLAForgeRegionExecutableValueApi*
vlaforge_aoti_region_executable_value_api(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_BACKENDS_AOTI_REGION_EXECUTABLE_H_
