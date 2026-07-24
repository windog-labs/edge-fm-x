#ifndef VLAFORGE_RUNTIME_TENSOR_VIEW_H_
#define VLAFORGE_RUNTIME_TENSOR_VIEW_H_

#include <cstddef>
#include <cstdint>

namespace vlaforge::runtime {

enum class ScalarType : std::uint32_t {
  kBool = 0,
  kI32 = 1,
  kI64 = 2,
  kF16 = 3,
  kBF16 = 4,
  kF32 = 5,
  kF64 = 6,
  kU8 = 7,
};

enum class DeviceType : std::uint32_t {
  kCpu = 0,
  kCuda = 1,
};

struct TensorView final {
  void* data = nullptr;
  const std::int64_t* shape = nullptr;
  std::uint32_t rank = 0;
  ScalarType scalar_type = ScalarType::kF32;
  DeviceType device_type = DeviceType::kCpu;
  std::uint32_t device_index = 0;
  std::size_t bytes = 0;
};

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_TENSOR_VIEW_H_
