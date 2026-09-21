#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

extern "C" cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph) {
  using Function = cudaError_t (*)(cudaGraphExec_t);
  static const auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "cudaGraphExecDestroy"));
  static bool injected = false;
  const auto* mode = std::getenv("VF_FAIL_GRAPH_DESTROY");
  if (!injected && mode != nullptr && std::strcmp(mode, "exec") == 0) {
    injected = true;
    std::fprintf(stderr, "[GRAPH-FAILURE] injected exec destroy %p\n", graph);
    return cudaErrorUnknown;
  }
  return original ? original(graph) : cudaErrorUnknown;
}

extern "C" cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  using Function = cudaError_t (*)(cudaGraph_t);
  static const auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "cudaGraphDestroy"));
  static bool injected = false;
  const auto* mode = std::getenv("VF_FAIL_GRAPH_DESTROY");
  if (!injected && mode != nullptr && std::strcmp(mode, "graph") == 0) {
    injected = true;
    std::fprintf(stderr, "[GRAPH-FAILURE] injected graph destroy %p\n", graph);
    return cudaErrorUnknown;
  }
  return original ? original(graph) : cudaErrorUnknown;
}
