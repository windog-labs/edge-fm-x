#include "vlaforge/runtime/device_copy.h"

#include <cstring>

#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
#include <cuda_runtime_api.h>
#endif

namespace vlaforge::runtime {
namespace {

bool IsCpu(VLAForgeDevice device) noexcept {
  return device.kind == VLAFORGE_DEVICE_CPU && device.ordinal == 0;
}

bool IsCuda(VLAForgeDevice device) noexcept {
  return device.kind == VLAFORGE_DEVICE_CUDA && device.ordinal >= 0;
}

}  // namespace

Status CopyBytes(void* destination, VLAForgeDevice destination_device,
                 const void* source, VLAForgeDevice source_device,
                 std::size_t size_bytes,
                 std::uint32_t subject_id) noexcept {
  if ((size_bytes != 0u &&
       (destination == nullptr || source == nullptr)) ||
      (!IsCpu(destination_device) && !IsCuda(destination_device)) ||
      (!IsCpu(source_device) && !IsCuda(source_device))) {
    return Status::Error(StatusCode::kInvalidArgument, subject_id,
                         "invalid explicit device copy");
  }
  if (size_bytes == 0u || destination == source) {
    return Status::Ok();
  }
  if (IsCpu(destination_device) && IsCpu(source_device)) {
    std::memcpy(destination, source, size_bytes);
    return Status::Ok();
  }
#if defined(VLAFORGE_ENABLE_CUDA_ARENA)
  if (IsCuda(destination_device) && IsCuda(source_device) &&
      destination_device.ordinal != source_device.ordinal) {
    if (cudaMemcpyPeer(destination, destination_device.ordinal, source,
                       source_device.ordinal, size_bytes) != cudaSuccess) {
      return Status::Error(StatusCode::kInternal, subject_id,
                           "CUDA peer copy failed");
    }
    return Status::Ok();
  }
  const int ordinal = IsCuda(destination_device)
                          ? destination_device.ordinal
                          : source_device.ordinal;
  if (cudaSetDevice(ordinal) != cudaSuccess) {
    return Status::Error(StatusCode::kInternal, subject_id,
                         "CUDA device selection failed");
  }
  cudaMemcpyKind kind = cudaMemcpyDefault;
  if (IsCpu(source_device)) {
    kind = cudaMemcpyHostToDevice;
  } else if (IsCpu(destination_device)) {
    kind = cudaMemcpyDeviceToHost;
  } else {
    kind = cudaMemcpyDeviceToDevice;
  }
  if (cudaMemcpy(destination, source, size_bytes, kind) != cudaSuccess) {
    return Status::Error(StatusCode::kInternal, subject_id,
                         "CUDA device copy failed");
  }
  return Status::Ok();
#else
  return Status::Error(StatusCode::kFailedPrecondition, subject_id,
                       "CUDA copy requires CUDA-enabled runtime");
#endif
}

}  // namespace vlaforge::runtime
