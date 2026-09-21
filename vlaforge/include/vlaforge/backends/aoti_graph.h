#ifndef VLAFORGE_BACKENDS_AOTI_GRAPH_H_
#define VLAFORGE_BACKENDS_AOTI_GRAPH_H_

#include "vlaforge/runtime/bounded_replay.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Owns a LibTorch CUDA graph and its caching-allocator pool. An invalidated
 * capture is fatal: the provider quarantines unsafe pool state until process
 * exit rather than claiming an ordinary-path recovery. */
const VLAForgeGraphBackendApi* vlaforge_aoti_graph_backend_api(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VLAFORGE_BACKENDS_AOTI_GRAPH_H_
