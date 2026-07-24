#ifndef VLAFORGE_BACKENDS_TORCHSCRIPT_REGION_EXECUTABLE_H_
#define VLAFORGE_BACKENDS_TORCHSCRIPT_REGION_EXECUTABLE_H_

#include "vlaforge/runtime/region_executable.h"

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_BACKENDS_TORCHSCRIPT_REGION_EXECUTABLE_H_
