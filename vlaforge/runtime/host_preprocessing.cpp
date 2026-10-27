#include "vlaforge/runtime/host_preprocessing.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

namespace vlaforge::runtime {
namespace {

Status Invalid(const char* message) noexcept {
  return Status::Error(StatusCode::kInvalidArgument, 0u, message);
}

bool ValidSide(SequenceSide side) noexcept {
  return side == SequenceSide::kLeft || side == SequenceSide::kRight;
}

bool Overlap(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) noexcept {
  const auto left = reinterpret_cast<std::uintptr_t>(a), right = reinterpret_cast<std::uintptr_t>(b);
  return left <= right ? right - left < a_bytes : left - right < b_bytes;
}

float BFloatRound(float value) noexcept {
  static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559);
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits = (bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace

Status ScaleU8ToF32(const std::uint8_t* input, std::size_t count,
                    float divisor, float* output) noexcept {
  if (input == nullptr || output == nullptr || count == 0u ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      !std::isfinite(divisor) || divisor <= 0.0F || !std::isfinite(255.0F / divisor)) {
    return Invalid("invalid U8 scaling buffers, count or divisor");
  }
  for (std::size_t index = 0; index < count; ++index) {
    output[index] = static_cast<float>(input[index]) / divisor;
  }
  return Status::Ok();
}

Status StandardizeF32(const float* input, std::size_t rows, std::size_t width,
                      const float* mean, const float* stddev, float epsilon,
                      float* output) noexcept {
  if (input == nullptr || output == nullptr || mean == nullptr || stddev == nullptr ||
      rows == 0u || width == 0u ||
      rows > std::numeric_limits<std::size_t>::max() / sizeof(float) / width ||
      !std::isfinite(epsilon) || epsilon < 0.0F) {
    return Invalid("invalid standardization buffers, dimensions or epsilon");
  }
  for (std::size_t dimension = 0; dimension < width; ++dimension) {
    const float denominator = stddev[dimension] + epsilon;
    if (!std::isfinite(mean[dimension]) || !std::isfinite(stddev[dimension]) ||
        stddev[dimension] < 0.0F || !std::isfinite(denominator) || denominator <= 0.0F) {
      return Invalid("invalid standardization statistics");
    }
  }
  for (std::size_t index = 0; index < rows * width; ++index) {
    const auto dimension = index % width;
    const float result = (input[index] - mean[dimension]) / (stddev[dimension] + epsilon);
    if (!std::isfinite(input[index]) || !std::isfinite(result)) {
      return Invalid("nonfinite standardization input or result");
    }
  }
  for (std::size_t index = 0; index < rows * width; ++index) {
    const auto dimension = index % width;
    output[index] = (input[index] - mean[dimension]) / (stddev[dimension] + epsilon);
  }
  return Status::Ok();
}

Status PatchifyF32(const float* input, std::size_t input_count,
                   const PatchifySpec& spec, float* output,
                   std::size_t output_count) noexcept {
  if (input == nullptr || output == nullptr || spec.frames == 0 || spec.channels == 0 ||
      spec.height == 0 || spec.width == 0 || spec.patch_height == 0 || spec.patch_width == 0 ||
      spec.temporal_patch == 0 || spec.spatial_merge == 0 ||
      spec.height % spec.patch_height != 0 || spec.width % spec.patch_width != 0 ||
      (spec.frames % spec.temporal_patch != 0 && !spec.repeat_last_frame)) {
    return Invalid("invalid patchify buffers, dimensions or padding policy");
  }
  const auto grid_y = spec.height / spec.patch_height, grid_x = spec.width / spec.patch_width;
  if (grid_y % spec.spatial_merge != 0 || grid_x % spec.spatial_merge != 0) {
    return Invalid("patch grid is not divisible by spatial merge");
  }
  const auto blocks = (spec.frames - 1) / spec.temporal_patch + 1;
  constexpr auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
  const auto multiply = [](std::size_t* value, std::size_t factor) {
    if (*value > limit / factor) return false;
    *value *= factor;
    return true;
  };
  std::size_t frame_size = spec.channels, padded = blocks;
  if (frame_size > limit || !multiply(&frame_size, spec.height) || !multiply(&frame_size, spec.width) ||
      padded > limit || !multiply(&padded, spec.temporal_patch)) {
    return Invalid("patchify dimensions overflow");
  }
  std::size_t source_count = frame_size, target_count = frame_size;
  if (!multiply(&source_count, spec.frames) || !multiply(&target_count, padded) ||
      source_count != input_count || target_count != output_count ||
      Overlap(input, input_count * sizeof(float), output, output_count * sizeof(float)) ||
      Overlap(output, output_count * sizeof(float), &spec, sizeof(spec))) {
    return Invalid("patchify count mismatch or overlapping buffers");
  }
  for (std::size_t i = 0; i < input_count; ++i) {
    if (!std::isfinite(input[i])) return Invalid("nonfinite patchify input");
  }
  std::size_t destination = 0;
  for (std::size_t t = 0; t < blocks; ++t)
    for (std::size_t gy = 0; gy < grid_y / spec.spatial_merge; ++gy)
      for (std::size_t gx = 0; gx < grid_x / spec.spatial_merge; ++gx)
        for (std::size_t my = 0; my < spec.spatial_merge; ++my)
          for (std::size_t mx = 0; mx < spec.spatial_merge; ++mx)
            for (std::size_t c = 0; c < spec.channels; ++c)
              for (std::size_t dt = 0; dt < spec.temporal_patch; ++dt)
                for (std::size_t py = 0; py < spec.patch_height; ++py)
                  for (std::size_t px = 0; px < spec.patch_width; ++px) {
                    const auto frame = std::min(t * spec.temporal_patch + dt, spec.frames - 1);
                    const auto y = (gy * spec.spatial_merge + my) * spec.patch_height + py;
                    const auto x = (gx * spec.spatial_merge + mx) * spec.patch_width + px;
                    output[destination++] = input[((frame * spec.channels + c) * spec.height + y) * spec.width + x];
                  }
  return Status::Ok();
}

Status RoundF32ToBFloat16F32(const float* input, std::size_t count, float* output) noexcept {
  if (input == nullptr || output == nullptr || count == 0 ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      (input != output && Overlap(input, count * sizeof(float), output, count * sizeof(float)))) {
    return Invalid("invalid BF16 carrier conversion buffers or overlap");
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(input[index]) || !std::isfinite(BFloatRound(input[index]))) {
      return Invalid("nonfinite BF16 carrier conversion input or result");
    }
  }
  for (std::size_t index = 0; index < count; ++index) output[index] = BFloatRound(input[index]);
  return Status::Ok();
}

Status ScatterDivideF32(const float* input, std::size_t rows, std::size_t input_width,
                        const std::size_t* indices, const float* divisors, std::size_t output_width,
                        bool round_bf16, float* output, float* availability_mask) noexcept {
  if (input == nullptr || output == nullptr || indices == nullptr || divisors == nullptr ||
      rows == 0 || input_width == 0 || output_width < input_width ||
      rows > std::numeric_limits<std::size_t>::max() / sizeof(float) / input_width ||
      rows > std::numeric_limits<std::size_t>::max() / sizeof(float) / output_width ||
      input_width > std::numeric_limits<std::size_t>::max() / sizeof(std::size_t)) {
    return Invalid("invalid scatter buffers, rows or widths");
  }
  const auto input_bytes = rows * input_width * sizeof(float), output_bytes = rows * output_width * sizeof(float);
  const auto parameters_overlap = [&](const void* destination) {
    return Overlap(destination, output_bytes, input, input_bytes) ||
        Overlap(destination, output_bytes, indices, input_width * sizeof(std::size_t)) ||
        Overlap(destination, output_bytes, divisors, input_width * sizeof(float));
  };
  if (parameters_overlap(output) || (availability_mask != nullptr &&
      (parameters_overlap(availability_mask) || Overlap(output, output_bytes, availability_mask, output_bytes)))) {
    return Invalid("scatter output overlaps input, parameters or mask");
  }
  for (std::size_t dimension = 0; dimension < input_width; ++dimension) {
    if (indices[dimension] >= output_width || !std::isfinite(divisors[dimension]) || divisors[dimension] == 0) {
      return Invalid("invalid scatter index or divisor");
    }
    for (std::size_t earlier = 0; earlier < dimension; ++earlier) {
      if (indices[dimension] == indices[earlier]) return Invalid("duplicate scatter index");
    }
  }
  for (std::size_t index = 0; index < rows * input_width; ++index) {
    if (!std::isfinite(input[index])) return Invalid("nonfinite scatter input");
    const float value = input[index] / divisors[index % input_width];
    if (!std::isfinite(value) || (round_bf16 && !std::isfinite(BFloatRound(value)))) {
      return Invalid("nonfinite scatter result");
    }
  }
  std::fill(output, output + rows * output_width, 0.0F);
  if (availability_mask != nullptr) std::fill(availability_mask, availability_mask + rows * output_width, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t dimension = 0; dimension < input_width; ++dimension) {
      const float value = input[row * input_width + dimension] / divisors[dimension];
      output[row * output_width + indices[dimension]] = round_bf16 ? BFloatRound(value) : value;
      if (availability_mask != nullptr) availability_mask[row * output_width + indices[dimension]] = 1.0F;
    }
  }
  return Status::Ok();
}

Status PadTokenIds(const std::int32_t* tokens, std::size_t count,
                   std::size_t capacity, std::int32_t pad_id, bool truncate,
                   SequenceSide padding_side, SequenceSide truncation_side,
                   std::int32_t* output, std::uint8_t* attention_mask) noexcept {
  if ((count != 0u && tokens == nullptr) || capacity == 0u || output == nullptr || attention_mask == nullptr ||
      capacity > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t) ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t) || pad_id < 0 ||
      !ValidSide(padding_side) || !ValidSide(truncation_side)) {
    return Invalid("invalid token padding buffers, capacity or policy");
  }
  if (count > capacity && !truncate) {
    return Status::Error(StatusCode::kOutOfRange, 0u, "token sequence exceeds declared capacity");
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (tokens[index] < 0) { return Invalid("negative token ID"); }
  }
  const auto used = std::min(count, capacity);
  const auto source = truncation_side == SequenceSide::kLeft ? count - used : 0u;
  const auto target = padding_side == SequenceSide::kLeft ? capacity - used : 0u;
  if (used != 0u) {
    std::memmove(output + target, tokens + source, used * sizeof(std::int32_t));
  }
  std::fill(output, output + target, pad_id);
  std::fill(output + target + used, output + capacity, pad_id);
  std::fill(attention_mask, attention_mask + capacity, 0u);
  for (std::size_t index = 0; index < used; ++index) {
    attention_mask[target + index] = 1u;
  }
  return Status::Ok();
}

Status EnsureTextSuffix(std::string_view text, std::string_view suffix,
                        std::string* output) noexcept {
  if (output == nullptr) { return Invalid("text output is null"); }
  try {
    std::string value(text);
    if (text.size() < suffix.size() || text.substr(text.size() - suffix.size()) != suffix) {
      value.append(suffix);
    }
    output->swap(value);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::Error(StatusCode::kResourceExhausted, 0u, "text preparation allocation failed");
  } catch (...) {
    return Invalid("text preparation failed");
  }
}

}  // namespace vlaforge::runtime
