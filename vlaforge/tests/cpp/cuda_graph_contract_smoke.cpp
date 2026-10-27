#include "vlaforge/runtime/cuda_graph.h"

int main() {
  const auto* api = vlaforge_cuda_graph_backend_api();
#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
  if (api == nullptr || api->struct_size != sizeof(*api) ||
      api->abi_version != VLAFORGE_GRAPH_BACKEND_ABI_VERSION) { return 1; }
  VLAForgeGraphExecutable* graph = nullptr;
  const VLAForgeExecutionContextView cpu{
      sizeof(VLAForgeExecutionContextView), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION,
      {VLAFORGE_DEVICE_CPU, 0}, nullptr};
  if (api->create(nullptr, &graph).code == VLAFORGE_STATUS_OK || graph != nullptr ||
      api->create(&cpu, &graph).code != VLAFORGE_STATUS_FAILED_PRECONDITION || graph != nullptr ||
      api->create(&cpu, nullptr).code != VLAFORGE_STATUS_INVALID_ARGUMENT ||
      api->begin(nullptr).code != VLAFORGE_STATUS_FAILED_PRECONDITION ||
      api->end(nullptr).code != VLAFORGE_STATUS_FAILED_PRECONDITION ||
      api->abort(nullptr).code != VLAFORGE_STATUS_FAILED_PRECONDITION ||
      api->launch(nullptr).code != VLAFORGE_STATUS_FAILED_PRECONDITION) { return 2; }
  api->destroy(nullptr, 0u);
#else
  if (api != nullptr) { return 3; }
#endif
  return 0;
}
