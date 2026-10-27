#include "vlaforge/backends/image_preprocessing.h"

#include "pillow_image_internal.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <new>

namespace {
std::mutex imaging_mutex;

bool Bytes(std::size_t width, std::size_t height, std::size_t* result) noexcept {
  if (width == 0 || height == 0 || width > static_cast<std::size_t>(std::numeric_limits<int>::max() / 4 - 1) ||
      height > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      width > std::numeric_limits<std::size_t>::max() / 4 / height) return false;
  *result = width * height * 3;
  return true;
}
}

namespace vlaforge::backends {
runtime::Status ResizeRgbWithPillow(const RgbImageU8& image, const RgbResizeSpec& spec,
                                    std::uint8_t* output, std::size_t output_bytes) noexcept {
  using runtime::Status;
  using runtime::StatusCode;
  std::size_t input_size = 0, output_size = 0, padded_size = 0;
  const auto width = spec.pad_to_square ? std::max(image.width, image.height) : image.width;
  const auto height = spec.pad_to_square ? width : image.height;
  if (image.data == nullptr || output == nullptr || !Bytes(image.width, image.height, &input_size) ||
      !Bytes(spec.width, spec.height, &output_size) || !Bytes(width, height, &padded_size) ||
      image.bytes != input_size || output_bytes != output_size) {
    return Status::Error(StatusCode::kInvalidArgument, 0, "invalid contiguous image buffers or dimensions");
  }
  int filter = 0;
  switch (spec.filter) {
    case ImageFilter::kBilinear: filter = 2; break;
    case ImageFilter::kBicubic: filter = 3; break;
    case ImageFilter::kLanczos: filter = 1; break;
    default: return Status::Error(StatusCode::kInvalidArgument, 0, "unsupported declared image filter");
  }
  try {
    std::lock_guard<std::mutex> lock(imaging_mutex);
    VLAForgePillowResize request{image.data, output, static_cast<int>(image.width), static_cast<int>(image.height),
        static_cast<int>(spec.width), static_cast<int>(spec.height), spec.pad_to_square ? 1 : 0, filter, {0, 0, 0}};
    std::copy(spec.background.begin(), spec.background.end(), request.background);
    const char* error = nullptr;
    if (!vlaforge_pillow_resize(&request, &error)) return Status::Error(StatusCode::kResourceExhausted, 0, error);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::Error(StatusCode::kResourceExhausted, 0, "image preparation allocation failed");
  } catch (...) {
    return Status::Error(StatusCode::kInvalidArgument, 0, "native image preparation failed");
  }
}
}  // namespace vlaforge::backends
