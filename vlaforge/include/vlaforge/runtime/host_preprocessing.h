#ifndef VLAFORGE_RUNTIME_HOST_PREPROCESSING_H_
#define VLAFORGE_RUNTIME_HOST_PREPROCESSING_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "vlaforge/runtime/status.h"

namespace vlaforge::runtime {

enum class SequenceSide : std::uint8_t { kLeft = 0, kRight = 1 };

// Callers provide sized host storage. Scale preserves the existing layout;
// no implicit channel permutation, image resize, clipping or normalizer lookup.
Status ScaleU8ToF32(const std::uint8_t* input, std::size_t count,
                    float divisor, float* output) noexcept;
Status StandardizeF32(const float* input, std::size_t rows, std::size_t width,
                      const float* mean, const float* stddev, float epsilon,
                      float* output) noexcept;

struct PatchifySpec {
  std::size_t frames, channels, height, width;
  std::size_t patch_height, patch_width, temporal_patch, spatial_merge;
  bool repeat_last_frame;
};

// Reorder contiguous [frame, channel, height, width] into grouped patch rows.
// Row order: temporal block, spatial group y/x, merge y/x. Within each row:
// channel, temporal offset, patch y/x. No normalization, resizing or allocation.
// An incomplete final temporal block repeats its last frame only when declared.
// Both counts must match the complete declared shapes. Finite input and disjoint
// buffers are required; validation failure leaves output unchanged.
Status PatchifyF32(const float* input, std::size_t input_count,
                   const PatchifySpec& spec, float* output,
                   std::size_t output_count) noexcept;

// Finite FLOAT preparation only. Exact in-place conversion is supported;
// partial overlap, nonfinite input and BF16 overflow fail before any write.
Status RoundF32ToBFloat16F32(const float* input, std::size_t count, float* output) noexcept;
// Scatter normalized rows into a wider zero-filled layout. Optional availability
// mask has the same shape; outputs must not overlap input or parameter storage.
Status ScatterDivideF32(const float* input, std::size_t rows, std::size_t input_width,
                        const std::size_t* indices, const float* divisors, std::size_t output_width,
                        bool round_bf16, float* output, float* availability_mask = nullptr) noexcept;

// Numeric transforms validate the complete domain before writing output.
// Tokenization itself belongs to a proven provider, not this padding utility.
Status PadTokenIds(const std::int32_t* tokens, std::size_t count,
                   std::size_t capacity, std::int32_t pad_id, bool truncate,
                   SequenceSide padding_side, SequenceSide truncation_side,
                   std::int32_t* output, std::uint8_t* attention_mask) noexcept;
Status EnsureTextSuffix(std::string_view text, std::string_view suffix,
                        std::string* output) noexcept;

}  // namespace vlaforge::runtime

#endif  // VLAFORGE_RUNTIME_HOST_PREPROCESSING_H_
