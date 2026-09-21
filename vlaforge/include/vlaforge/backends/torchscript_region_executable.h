#ifndef VLAFORGE_BACKENDS_TORCHSCRIPT_REGION_EXECUTABLE_H_
#define VLAFORGE_BACKENDS_TORCHSCRIPT_REGION_EXECUTABLE_H_

#include "vlaforge/runtime/region_executable.h"
#include "vlaforge/runtime/execution_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Returns the CPU TorchScript implementation of the stable
 * VLAForgeRegionExecutable ABI. The archive is loaded and invoked entirely
 * through LibTorch; no Python runtime is embedded or started.
 */
const VLAForgeRegionExecutableApi*
vlaforge_torchscript_region_executable_api(void);

/* Tensor-only Value ABI. Uses backend_variant="torchscript-aten/1" or
 * "torchscript-aten-context/1" (CUDA only, optional shared stream sidecar).
 * Preserves archived CPU/CUDA tensor placement and scopes JIT optimization
 * off to the calling thread. CUDA support is a separate build option. The
 * base variant synchronizes each call. The context variant uses a borrowed
 * stream when bound, leaving completion to synchronize or the Session.
 */
const VLAForgeRegionExecutableValueApi*
vlaforge_torchscript_region_executable_value_api(void);

/* Context variant only. Rebinding requires completed work; caller owns the
 * stream lifetime. External capture uses the shared LibTorch graph provider.
 */
const VLAForgeRegionExecutionExtensionApi*
vlaforge_torchscript_region_execution_extension_api(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_BACKENDS_TORCHSCRIPT_REGION_EXECUTABLE_H_
