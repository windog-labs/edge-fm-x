#ifndef VLAFORGE_RUNTIME_DEVICE_COPY_H_
#define VLAFORGE_RUNTIME_DEVICE_COPY_H_

#include <cstddef>
#include <cstdint>

#include "vlaforge/runtime/region_executable.h"
#include "vlaforge/runtime/status.h"

namespace vlaforge::runtime {

// Performs one explicit synchronous copy between declared CPU/CUDA devices.
// The function never infers ownership and never changes either allocation.
[[nodiscard]] Status CopyBytes(
    void* destination, VLAForgeDevice destination_device,
    const void* source, VLAForgeDevice source_device,
    std::size_t size_bytes, std::uint32_t subject_id = 0) noexcept;

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_DEVICE_COPY_H_
