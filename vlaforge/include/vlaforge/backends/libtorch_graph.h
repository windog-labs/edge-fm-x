#ifndef VLAFORGE_BACKENDS_LIBTORCH_GRAPH_H_
#define VLAFORGE_BACKENDS_LIBTORCH_GRAPH_H_

#include "vlaforge/runtime/bounded_replay.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Shared LibTorch allocator/graph provider. Eligible Regions must use the
 * supplied stream and keep all execution and storage inside the capture.
 * Failed capture quarantines unsafe allocator state until process exit. */
const VLAForgeGraphBackendApi* vlaforge_libtorch_graph_backend_api(void);

#ifdef __cplusplus
}
#endif

#endif  // VLAFORGE_BACKENDS_LIBTORCH_GRAPH_H_
