"""Offline linker-wrapper instrumentation, never a deployed numerical policy."""

LINK_FLAG = "-Wl,--wrap=vlaforge_aoti_region_executable_value_api"

SOURCE = r"""
#include "vlaforge/backends/aoti_region_executable.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/autocast_mode.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" const VLAForgeRegionExecutableValueApi*
__real_vlaforge_aoti_region_executable_value_api(void);

namespace openpi_offline_trace {
struct Region {
  std::uint32_t id = 0;
  std::map<std::uint32_t, VLAForgeValueView> inputs;
  std::map<std::uint32_t, VLAForgeValueView> outputs;
};
std::map<VLAForgeRegionExecutable*, Region> regions;
std::size_t invocation = 0;
const auto* Real() { return __real_vlaforge_aoti_region_executable_value_api(); }
std::string Root() {
  const char* value = std::getenv("VLAFORGE_DIAGNOSTIC_TRACE_DIR");
  if (!value || !*value) throw std::runtime_error("missing diagnostic trace directory");
  return value;
}
void Policy(std::size_t call) {
  auto& c = at::globalContext();
  std::ofstream f(Root() + "/numerical.jsonl", std::ios::app);
  if (!f) throw std::runtime_error("cannot open numerical diagnostic");
  f << std::boolalpha << "{\"call\":" << call
    << ",\"threads\":" << at::get_num_threads()
    << ",\"interop_threads\":" << at::get_num_interop_threads()
    << ",\"float32_matmul_precision_enum\":" << static_cast<int>(c.float32MatmulPrecision())
    << ",\"cuda_matmul_allow_tf32\":" << c.allowTF32CuBLAS()
    << ",\"cudnn_allow_tf32\":" << c.allowTF32CuDNN()
    << ",\"cuda_matmul_fp16_reduction_option_enum\":" << static_cast<int>(c.allowFP16ReductionCuBLAS())
    << ",\"cuda_matmul_bf16_reduction_option_enum\":" << static_cast<int>(c.allowBF16ReductionCuBLAS())
    << ",\"autocast_cpu_enabled\":" << at::autocast::is_autocast_enabled(at::kCPU)
    << ",\"autocast_cpu_dtype\":\"" << c10::toString(at::autocast::get_autocast_dtype(at::kCPU)) << "\""
    << ",\"autocast_cuda_enabled\":" << at::autocast::is_autocast_enabled(at::kCUDA)
    << ",\"autocast_cuda_dtype\":\"" << c10::toString(at::autocast::get_autocast_dtype(at::kCUDA)) << "\""
    << ",\"autocast_cache_enabled\":" << at::autocast::is_autocast_cache_enabled()
    << ",\"sdpa_flash_enabled\":" << c.userEnabledFlashSDP()
    << ",\"sdpa_mem_efficient_enabled\":" << c.userEnabledMemEfficientSDP()
    << ",\"sdpa_math_enabled\":" << c.userEnabledMathSDP()
    << ",\"sdpa_cudnn_enabled\":" << c.userEnabledCuDNNSDP()
    << ",\"sdpa_math_allow_fp16_bf16_reduction\":" << c.allowFP16BF16ReductionMathSDP()
    << ",\"deterministic_algorithms_enabled\":" << c.deterministicAlgorithms()
    << ",\"deterministic_algorithms_warn_only\":" << c.deterministicAlgorithmsWarnOnly()
    << ",\"cudnn_enabled\":" << c.userEnabledCuDNN()
    << ",\"cudnn_benchmark\":" << c.benchmarkCuDNN()
    << ",\"cudnn_deterministic\":" << c.deterministicCuDNN()
    << "}\n";
  if (!f) throw std::runtime_error("cannot write numerical diagnostic");
}
void Dump(std::size_t call, const Region& region, const char* direction,
          const std::map<std::uint32_t, VLAForgeValueView>& bindings) {
  for (const auto& [index, value] : bindings) {
    if (value.kind != VLAFORGE_VALUE_TENSOR ||
        value.value.tensor.layout != VLAFORGE_LAYOUT_CONTIGUOUS) {
      throw std::runtime_error("trace requires explicit contiguous tensor bindings");
    }
    const auto& view = value.value.tensor.tensor;
    std::vector<char> bytes(view.size_bytes);
    if (view.device.kind != VLAFORGE_DEVICE_CUDA ||
        cudaMemcpy(bytes.data(), view.data, bytes.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("diagnostic tensor download failed");
    }
    const std::string name = "call-" + std::to_string(call) + "-" + direction + "-" + std::to_string(index) + ".bin";
    std::ofstream data(Root() + "/" + name, std::ios::binary);
    data.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!data) throw std::runtime_error("cannot write complete tensor diagnostic");
    std::ofstream meta(Root() + "/tensors.jsonl", std::ios::app);
    meta << "{\"call\":" << call << ",\"region_id\":" << region.id
         << ",\"direction\":\"" << direction << "\",\"index\":" << index
         << ",\"file\":\"" << name << "\",\"size_bytes\":" << view.size_bytes
         << ",\"dtype_enum\":" << static_cast<int>(view.dtype)
         << ",\"device_ordinal\":" << view.device.ordinal
         << ",\"layout\":\"contiguous\",\"storage_offset\":0,\"address_mod_256\":"
         << reinterpret_cast<std::uintptr_t>(view.data) % 256u << ",\"shape\":[";
    for (std::uint32_t i = 0; i < view.rank; ++i) {
      if (i) meta << ',';
      meta << view.dimensions[i];
    }
    meta << "],\"stride\":[";
    std::vector<std::int64_t> strides(view.rank, 1);
    for (std::size_t i = view.rank; i > 1u; --i) strides[i - 2u] = strides[i - 1u] * view.dimensions[i - 1u];
    for (std::size_t i = 0; i < strides.size(); ++i) { if (i) meta << ','; meta << strides[i]; }
    meta << "],\"stride_source\":\"native contiguous ABI from_blob contract\"}\n";
    if (!meta) throw std::runtime_error("cannot write tensor metadata");
  }
}
VLAForgeStatus Create(const VLAForgeRegionCreateOptions* options, VLAForgeRegionExecutable** result) {
  const auto status = Real()->create(options, result);
  if (status.code == VLAFORGE_STATUS_OK) regions[*result].id = options->region_id;
  return status;
}
VLAForgeStatus Input(VLAForgeRegionExecutable* executable, std::uint32_t index, const VLAForgeValueView* value) {
  const auto status = Real()->bind_input(executable, index, value);
  if (status.code == VLAFORGE_STATUS_OK) regions.at(executable).inputs[index] = *value;
  return status;
}
VLAForgeStatus Output(VLAForgeRegionExecutable* executable, std::uint32_t index, const VLAForgeValueView* value) {
  const auto status = Real()->bind_output(executable, index, value);
  if (status.code == VLAFORGE_STATUS_OK) regions.at(executable).outputs[index] = *value;
  return status;
}
VLAForgeStatus Run(VLAForgeRegionExecutable* executable) {
  try {
    const auto call = invocation++;
    auto& region = regions.at(executable);
    Policy(call);
    Dump(call, region, "input", region.inputs);
    auto status = Real()->run(executable);
    if (status.code != VLAFORGE_STATUS_OK) return status;
    status = Real()->synchronize(executable);
    if (status.code != VLAFORGE_STATUS_OK) return status;
    Dump(call, region, "output", region.outputs);
    return status;
  } catch (const std::exception&) {
    return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, "offline trace failed; inspect retained files");
  }
}
void Destroy(VLAForgeRegionExecutable* executable) {
  regions.erase(executable);
  Real()->destroy(executable);
}
}  // namespace openpi_offline_trace

extern "C" const VLAForgeRegionExecutableValueApi*
__wrap_vlaforge_aoti_region_executable_value_api(void) {
  static const auto api = [] {
    auto result = *openpi_offline_trace::Real();
    result.create = &openpi_offline_trace::Create;
    result.bind_input = &openpi_offline_trace::Input;
    result.bind_output = &openpi_offline_trace::Output;
    result.run = &openpi_offline_trace::Run;
    result.destroy = &openpi_offline_trace::Destroy;
    return result;
  }();
  return &api;
}
"""
