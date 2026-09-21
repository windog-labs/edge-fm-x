#ifndef VLAFORGE_BACKENDS_AOTI_REGION_EXECUTABLE_H_
#define VLAFORGE_BACKENDS_AOTI_REGION_EXECUTABLE_H_

#include "vlaforge/runtime/region_executable.h"
#include "vlaforge/runtime/execution_context.h"

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
 * Production callable ABI. Tensor storage is borrowed through synchronize.
 * Scalar values are rejected because an AOTI TensorRegion callable accepts
 * tensors only; host scalars must be tensorized by the generated Session or a
 * preprocessing Region.
 */
const VLAForgeRegionExecutableValueApi*
vlaforge_aoti_region_executable_value_api(void);

/* Optional shared-context support; existing Region ABI tables are unchanged. */
const VLAForgeRegionExecutionExtensionApi*
vlaforge_aoti_region_execution_extension_api(void);

/* Optional per-instance override, before load only. The caller supplies an
 * existing absolute canonical private (0700) directory. The backend owns only
 * unique children, not this root. No global environment or old ABI changes.
 * Explicit extraction requires a complete artifact SHA-256/size descriptor.
 */
VLAForgeStatus vlaforge_aoti_set_package_extraction_root(
    VLAForgeRegionExecutable* executable, const char* path, size_t path_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_BACKENDS_AOTI_REGION_EXECUTABLE_H_
