#ifndef VLAFORGE_RUNTIME_EXTERNAL_REGION_PLUGIN_H_
#define VLAFORGE_RUNTIME_EXTERNAL_REGION_PLUGIN_H_

#include <stddef.h>

#include "vlaforge/runtime/region_executable.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Host-side loader for customer RegionExecutable shared libraries.
 *
 * A plugin exports VLAFORGE_REGION_EXECUTABLE_VALUE_API_SYMBOL and returns a
 * VLAForgeRegionExecutableValueApi. The plugin accepts only the stable
 * Tensor/Scalar ABI; middleware objects and sensor callbacks are intentionally
 * outside this interface.
 */
typedef struct VLAForgeExternalRegionPlugin VLAForgeExternalRegionPlugin;

VLAForgeStatus
vlaforge_external_region_plugin_open(const char *path, size_t path_size,
                                     VLAForgeExternalRegionPlugin **plugin);

const VLAForgeRegionExecutableValueApi *
vlaforge_external_region_plugin_api(const VLAForgeExternalRegionPlugin *plugin);

void vlaforge_external_region_plugin_close(
    VLAForgeExternalRegionPlugin *plugin);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VLAFORGE_RUNTIME_EXTERNAL_REGION_PLUGIN_H_
