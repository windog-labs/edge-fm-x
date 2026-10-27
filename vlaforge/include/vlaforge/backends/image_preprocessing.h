#ifndef VLAFORGE_BACKENDS_IMAGE_PREPROCESSING_H_
#define VLAFORGE_BACKENDS_IMAGE_PREPROCESSING_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "vlaforge/runtime/status.h"

namespace vlaforge::backends {

struct RgbImageU8 {
  const std::uint8_t* data;
  std::size_t width, height, bytes;
};
enum class ImageFilter : std::uint8_t { kBilinear, kBicubic, kLanczos };
struct RgbResizeSpec {
  std::size_t width, height;
  bool pad_to_square;
  std::array<std::uint8_t, 3> background;
  ImageFilter filter;
};

#if defined(_WIN32)
#if defined(VLAFORGE_PILLOW_IMAGE_BUILD)
#define VLAFORGE_IMAGE_EXPORT __declspec(dllexport)
#else
#define VLAFORGE_IMAGE_EXPORT __declspec(dllimport)
#endif
#else
#define VLAFORGE_IMAGE_EXPORT __attribute__((visibility("default")))
#endif

// Contiguous three-byte pixels; channel order is preserved, not interpreted.
// The optional pinned C library owns resize arithmetic. No interpreter is used.
VLAFORGE_IMAGE_EXPORT runtime::Status ResizeRgbWithPillow(
    const RgbImageU8& image, const RgbResizeSpec& spec,
    std::uint8_t* output, std::size_t output_bytes) noexcept;

#undef VLAFORGE_IMAGE_EXPORT
}  // namespace vlaforge::backends
#endif
