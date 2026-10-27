#ifndef VLAFORGE_RUNTIME_CUDA_GRAPH_H_
#define VLAFORGE_RUNTIME_CUDA_GRAPH_H_

#include "vlaforge/runtime/bounded_replay.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Native CUDA capture for preallocated, externally-streamed executables.
 * No framework allocator/RNG integration is provided. Engine/effect eligibility
 * is a separate contract; failed completion quarantines resources. CPU-only
 * builds return null rather than simulating graph support. */
const VLAForgeGraphBackendApi* vlaforge_cuda_graph_backend_api(void);

#ifdef __cplusplus
}
#endif

#endif  // VLAFORGE_RUNTIME_CUDA_GRAPH_H_
