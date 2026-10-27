#ifndef VLAFORGE_BACKENDS_TENSORRT_REGION_EXECUTABLE_H_
#define VLAFORGE_BACKENDS_TENSORRT_REGION_EXECUTABLE_H_

#include "vlaforge/runtime/execution_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Returns the TensorRT implementation of the stable RegionExecutable ABI.
 * Serialized TensorRT engines are loaded without Python. Tensor bindings are
 * borrowed through synchronize() and remain owned by the caller.
 */
const VLAForgeRegionExecutableApi*
vlaforge_tensorrt_region_executable_api(void);

/*
 * Value ABI variant. Tensor values must use CONTIGUOUS layout. Scalars are
 * intentionally rejected; adapters must tensorize scalar engine inputs.
 */
const VLAForgeRegionExecutableValueApi*
vlaforge_tensorrt_region_executable_value_api(void);

/* Optional borrowed CUDA stream; synchronize before changing/detaching it.
 * Shared-context support alone does not certify an engine for graph capture. */
const VLAForgeRegionExecutionExtensionApi*
vlaforge_tensorrt_region_execution_extension_api(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_BACKENDS_TENSORRT_REGION_EXECUTABLE_H_
