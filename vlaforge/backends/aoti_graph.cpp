#include "vlaforge/backends/aoti_graph.h"
#include "vlaforge/backends/libtorch_graph.h"

extern "C" const VLAForgeGraphBackendApi* vlaforge_aoti_graph_backend_api(void) {
  return vlaforge_libtorch_graph_backend_api();
}
