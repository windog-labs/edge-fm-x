#include "vlaforge/backends/libtorch_graph.h"

#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using PoolId = std::pair<unsigned long long, unsigned long long>;
const auto* api = vlaforge_libtorch_graph_backend_api();

void Check(VLAForgeStatus status) {
  if (status.code != VLAFORGE_STATUS_OK) {
    throw std::runtime_error(std::string(status.message, status.message_size));
  }
}

std::set<PoolId> PrivatePools() {
  std::set<PoolId> result;
  for (const auto& segment : c10::cuda::CUDACachingAllocator::snapshot().segments) {
    if (segment.owner_private_pool_id != PoolId{0, 0}) result.insert(segment.owner_private_pool_id);
  }
  return result;
}

struct Owner {
  VLAForgeExecutionContext* context = nullptr;
  VLAForgeExecutionContextView view{};
  VLAForgeGraphExecutable* graph = nullptr;
  at::Tensor input, output;

  explicit Owner(int scale, bool partial = false) {
    const VLAForgeExecutionContextOptions options{sizeof(options), VLAFORGE_EXECUTION_CONTEXT_ABI_VERSION,
                                                  {VLAFORGE_DEVICE_CUDA, 0}};
    Check(vlaforge_execution_context_create(&options, &context));
    view.struct_size = sizeof(view);
    Check(vlaforge_execution_context_get_view(context, &view));
    const c10::cuda::CUDAStreamGuard guard(Stream());
    input = at::arange(4096, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    output = at::empty_like(input);
    output.copy_(input.mul(scale));
    Stream().synchronize();
    Check(api->create(&view, &graph));
    Check(api->begin(graph));
    output.copy_(input.mul(scale));
    if (partial) Check(api->abort(graph));
    else Check(api->end(graph));
  }

  c10::cuda::CUDAStream Stream() const {
    return c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(view.native_stream), 0);
  }

  void Validate(const std::string& path, int scale, bool replay = true) {
    const c10::cuda::CUDAStreamGuard guard(Stream());
    if (replay) Check(api->launch(graph));
    Stream().synchronize();
    auto cpu = output.cpu();
    const auto* values = cpu.const_data_ptr<float>();
    for (int element = 0; element < 4096; ++element) {
      if (values[element] != float(element * scale)) throw std::runtime_error("full output mismatch");
    }
    std::ofstream raw(path, std::ios::binary);
    raw.write(reinterpret_cast<const char*>(values), cpu.nbytes());
    if (!raw.good()) throw std::runtime_error("full raw write failure");
  }

  void Destroy(unsigned poisoned = 0) {
    api->destroy(graph, poisoned);
    graph = nullptr;
    vlaforge_execution_context_destroy(context);
    context = nullptr;
  }
  ~Owner() { if (context != nullptr) Destroy(); }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const std::string folder(argv[1]), mode(argv[2]);
  const bool fault = mode != "success";
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  const c10::cuda::CUDAGuard device(0);
  for (int cycle = 0; cycle < (fault ? 1 : 5); ++cycle) {
    std::vector<std::unique_ptr<Owner>> owners;
    for (int index = 0; index < 2; ++index) owners.push_back(std::make_unique<Owner>(index + 2));
    const auto captured_pools = PrivatePools();
    if (captured_pools.size() != 2) return 3;
    for (int index = 1; index >= 0; --index) {
      owners[index]->Validate(folder + "/cycle-" + std::to_string(cycle) + "-graph-" + std::to_string(index) + ".f32", index + 2);
    }
    const auto before = c10::cuda::CUDACachingAllocator::getDeviceStats(0).num_device_free;
    if (fault) {
      if (mode != "poison") setenv("VF_FAIL_GRAPH_DESTROY", mode.c_str(), 1);
      owners[0]->Destroy(mode == "poison" ? 1u : 0u);
      unsetenv("VF_FAIL_GRAPH_DESTROY");
      if (c10::cuda::CUDACachingAllocator::getDeviceStats(0).num_device_free != before) return 4;
      if (PrivatePools() != captured_pools) return 5;
      // A healthy new graph must get its own pool even after one was quarantined.
      auto replacement = std::make_unique<Owner>(5);
      const auto with_replacement = PrivatePools();
      if (with_replacement.size() != 3) return 6;
      replacement->Validate(folder + "/replacement.f32", 5);
      owners[1]->Validate(folder + "/survivor.f32", 3);
      replacement->Destroy();
      owners[1]->Destroy();
      if (PrivatePools().size() != 1) return 7;
      std::printf("{\"quarantine_retained\":true,\"failed_destroy_device_frees\":0,\"new_pool_isolated\":true,\"published_outputs_exact\":true}\n");
      return 78;
    }
    for (auto& owner : owners) owner->Destroy();
    owners.clear();
    if (!PrivatePools().empty()) return 8;
    // Legal partial capture balances the allocator without ever instantiating.
    auto partial = std::make_unique<Owner>(7, true);
    partial->Validate(folder + "/cycle-" + std::to_string(cycle) + "-partial.f32", 7, false);
    partial->Destroy();
    partial.reset();
    if (!PrivatePools().empty()) return 9;
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    if (stats.num_device_free <= before) return 10;
    std::printf("{\"cycle\":%d,\"private_pool_count\":0,\"device_frees_this_destroy\":%lld}\n",
                cycle, static_cast<long long>(stats.num_device_free - before));
  }
}
