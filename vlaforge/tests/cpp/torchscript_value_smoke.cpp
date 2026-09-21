#include "vlaforge/backends/torchscript_region_executable.h"

#include <ATen/ATen.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
const auto* api = vlaforge_torchscript_region_executable_value_api();
using Handle = std::unique_ptr<VLAForgeRegionExecutable, void (*)(VLAForgeRegionExecutable*)>;

void Check(VLAForgeStatus status) {
  if (status.code != VLAFORGE_STATUS_OK) {
    throw std::runtime_error(std::string(status.message, status.message_size));
  }
}

void Reject(VLAForgeStatus status) {
  if (status.code == VLAFORGE_STATUS_OK) throw std::runtime_error("invalid call accepted");
}

Handle Create(VLAForgeDevice device) {
  VLAForgeRegionCreateOptions options{
      sizeof(options), VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION, 0u, device};
  VLAForgeRegionExecutable* result = nullptr;
  Check(api->create(&options, &result));
  return Handle(result, api->destroy);
}

VLAForgeArtifactDescriptor Artifact(const std::string& path) {
  constexpr char variant[] = "torchscript-aten/1";
  return {sizeof(VLAForgeArtifactDescriptor), VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
          path.data(), path.size(), nullptr, 0u, nullptr, 0u, nullptr, 0u,
          "torchscript-aten/1", sizeof(variant) - 1u};
}

VLAForgeValueView View(at::Tensor& tensor, VLAForgeDevice device) {
  VLAForgeValueView result{};
  result.struct_size = sizeof(result);
  result.kind = VLAFORGE_VALUE_TENSOR;
  result.value.tensor = {sizeof(VLAForgeBoundTensor),
      {tensor.data_ptr(), static_cast<std::uint64_t>(tensor.nbytes()), tensor.sizes().data(),
       static_cast<std::uint32_t>(tensor.dim()), VLAFORGE_DTYPE_BF16, device},
      VLAFORGE_LAYOUT_CONTIGUOUS, 1u};
  return result;
}

void Exact(const at::Tensor& left, const at::Tensor& right) {
  const auto a = left.cpu().contiguous(), b = right.cpu().contiguous();
  if (a.sizes() != b.sizes() || a.scalar_type() != b.scalar_type() ||
      std::memcmp(a.const_data_ptr(), b.const_data_ptr(), a.nbytes()) != 0) {
    throw std::runtime_error("full output bytes differ");
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    const bool cuda = std::string(argv[2]) == "cuda";
    const VLAForgeDevice device{cuda ? VLAFORGE_DEVICE_CUDA : VLAFORGE_DEVICE_CPU, 0};
    const auto placement = cuda ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
    const auto options = at::TensorOptions().dtype(at::kBFloat16).device(placement);
    const std::string path(argv[1]);
    const auto scalar = at::scalar_tensor(30.983866769659336, at::TensorOptions().dtype(at::kFloat));
    torch::jit::Module module("NativeConstantProbe");
    module.register_buffer("scale", scalar);
    module.define("def forward(self, x):\n  return (x * self.scale, x + 1.0)\n");
    module.save(path);
    auto handle = Create(device);
    auto artifact = Artifact(path);
    auto invalid = artifact;
    invalid.backend_variant_size = 0;
    Reject(api->load(handle.get(), &invalid));
    Check(api->load(handle.get(), &artifact));
    VLAForgeWorkspaceRequirement workspace{};
    Check(api->query_workspace(handle.get(), &workspace));
    if (workspace.size_bytes != 0 || workspace.device.kind != device.kind) return 3;
    auto input = at::arange(16, options).reshape({4, 4}).div(8.0);
    auto first = at::empty_like(input), second = at::empty_like(input);
    auto in = View(input, device), one = View(first, device), two = View(second, device);
    auto bad = in;
    bad.value.tensor.layout = VLAFORGE_LAYOUT_CUSTOM;
    Reject(api->bind_input(handle.get(), 0u, &bad));
    bad = in;
    bad.value.tensor.tensor.size_bytes -= 1u;
    Reject(api->bind_input(handle.get(), 0u, &bad));
    bad = in;
    bad.kind = VLAFORGE_VALUE_SCALAR;
    Reject(api->bind_input(handle.get(), 0u, &bad));
    bad = in;
    bad.value.tensor.tensor.dtype = VLAFORGE_DTYPE_U64;
    Reject(api->bind_input(handle.get(), 0u, &bad));
    bad = in;
    bad.value.tensor.alignment = 3u;
    Reject(api->bind_input(handle.get(), 0u, &bad));
    if (cuda) {
      auto host = input.cpu();
      auto wrong = View(host, device);
      Reject(api->bind_input(handle.get(), 0u, &wrong));
    }
    Check(api->bind_input(handle.get(), 0u, &in));
    Check(api->bind_output(handle.get(), 0u, &one));
    Check(api->bind_output(handle.get(), 1u, &two));
    auto wrong_shape = second.reshape({2, 8});
    auto wrong_output = View(wrong_shape, device);
    first.fill_(42);
    second.fill_(43);
    Check(api->bind_output(handle.get(), 1u, &wrong_output));
    Reject(api->run(handle.get()));
    Exact(first, at::full_like(input, 42));
    Exact(second, at::full_like(input, 43));
    Check(api->bind_output(handle.get(), 1u, &two));
    for (const bool optimize : {true, false, true}) {
      torch::jit::GraphOptimizerEnabledGuard caller(optimize);
      input.add_(0.125);
      Check(api->run(handle.get()));
      Check(api->synchronize(handle.get()));
      if (torch::jit::getGraphExecutorOptimize() != optimize) return 4;
      Exact(first, input * scalar);
      Exact(second, input + 1.0);
    }
    const std::string missing = path + "#missing";
    const auto bad_method = Artifact(missing);
    Reject(api->load(handle.get(), &bad_method));
    Reject(api->run(handle.get()));
    Check(api->load(handle.get(), &artifact));
    Reject(api->run(handle.get()));  // Successful reload must not retain bindings.

    torch::jit::Module counter("NativeCounterProbe");
    counter.register_buffer("counter", at::zeros({}, at::TensorOptions().device(placement)));
    counter.define("def forward(self, x):\n  self.counter.add_(1.0)\n  return x + self.counter\n");
    counter.save(path);
    auto left = Create(device), right = Create(device);
    Check(api->load(left.get(), &artifact));
    Check(api->load(right.get(), &artifact));
    for (auto* item : {left.get(), right.get()}) {
      Check(api->bind_input(item, 0u, &in));
      Check(api->bind_output(item, 0u, &one));
      Check(api->run(item));
      Exact(first, input + 1.0);
    }
    torch::jit::Module foreign("ForeignDeviceProbe");
    foreign.define("def forward(self, x):\n  return x.to(device=torch.device(\"cuda:127\"))\n");
    foreign.save(path);
    Reject(api->load(handle.get(), &artifact));
    Reject(api->run(handle.get()));
    std::printf("PASS,full-bytes,scalar-device,jit-guard,bindings,load-failure,session-isolation,foreign-device,no-partial-output\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 10;
  }
}
