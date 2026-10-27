#include "vlaforge/runtime/host_preprocessing.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace {
using vlaforge::runtime::PatchifyF32;
using vlaforge::runtime::PatchifySpec;

bool TestSpatialAndTemporalOrder() {
  std::array<float, 16> input{}, output{};
  std::iota(input.begin(), input.end(), 0.0F);
  PatchifySpec spec{1, 1, 4, 4, 1, 1, 1, 2, false};
  const std::array<float, 16> expected{0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};
  if (!PatchifyF32(input.data(), input.size(), spec, output.data(), output.size()).ok() || output != expected) return false;
  spec.temporal_patch = 2;
  spec.repeat_last_frame = true;
  std::array<float, 32> repeated{};
  if (!PatchifyF32(input.data(), input.size(), spec, repeated.data(), repeated.size()).ok()) return false;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (repeated[2 * i] != expected[i] || repeated[2 * i + 1] != expected[i]) return false;
  }
  std::array<float, 32> frames{}, actual{};
  std::iota(frames.begin(), frames.end(), 0.0F);
  spec = {2, 2, 2, 4, 1, 2, 2, 1, false};
  const std::array<float, 32> rectangular{
    0, 1, 16, 17, 8, 9, 24, 25, 2, 3, 18, 19, 10, 11, 26, 27,
    4, 5, 20, 21, 12, 13, 28, 29, 6, 7, 22, 23, 14, 15, 30, 31};
  return PatchifyF32(frames.data(), frames.size(), spec, actual.data(), actual.size()).ok() && actual == rectangular;
}

bool TestValidationBeforeWrite() {
  std::array<float, 16> input{};
  std::array<float, 32> output{};
  output.fill(37);
  const auto previous = output;
  const PatchifySpec valid{1, 1, 4, 4, 1, 1, 2, 2, true};
  for (int field = 0; field < 8; ++field) {
    auto spec = valid;
    std::size_t* dimensions[]{&spec.frames, &spec.channels, &spec.height, &spec.width,
        &spec.patch_height, &spec.patch_width, &spec.temporal_patch, &spec.spatial_merge};
    *dimensions[field] = 0;
    if (PatchifyF32(input.data(), input.size(), spec, output.data(), output.size()).ok() || output != previous) return false;
  }
  for (int field = 0; field < 8; ++field) {
    auto spec = valid;
    std::size_t* dimensions[]{&spec.frames, &spec.channels, &spec.height, &spec.width,
        &spec.patch_height, &spec.patch_width, &spec.temporal_patch, &spec.spatial_merge};
    *dimensions[field] = std::numeric_limits<std::size_t>::max();
    if (PatchifyF32(input.data(), input.size(), spec, output.data(), output.size()).ok() || output != previous) return false;
  }
  auto reject = [&](const PatchifySpec& spec, std::size_t count = 16, std::size_t target = 32) {
    return !PatchifyF32(input.data(), count, spec, output.data(), target).ok() && output == previous;
  };
  auto spec = valid;
  spec.repeat_last_frame = false;
  if (!reject(spec) || !reject(valid, 15) || !reject(valid, 16, 31) || !reject(valid, 16, 33)) return false;
  spec = valid;
  spec.patch_height = 3;
  if (!reject(spec)) return false;
  spec = valid;
  spec.spatial_merge = 3;
  if (!reject(spec)) return false;
  if (PatchifyF32(nullptr, 16, valid, output.data(), 32).ok() || output != previous ||
      PatchifyF32(input.data(), 16, valid, nullptr, 32).ok()) return false;
  const auto preserved_input = input;
  if (PatchifyF32(input.data(), input.size(), valid, input.data(), 32).ok() ||
      PatchifyF32(input.data(), input.size(), valid, input.data() + 1, 32).ok() || input != preserved_input) return false;
  PatchifySpec overlapping_spec{1, 1, 1, 1, 1, 1, 1, 1, false};
  if (PatchifyF32(input.data(), 1, overlapping_spec, reinterpret_cast<float*>(&overlapping_spec), 1).ok() ||
      overlapping_spec.frames != 1) return false;
  for (float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    input.back() = bad;
    if (!reject(valid)) return false;
  }
  return true;
}

bool TestMultipleTemporalBlocks() {
  const std::array<float, 6> input{0, 1, 10, 11, 20, 21};
  std::array<float, 8> output{};
  const PatchifySpec spec{3, 1, 1, 2, 1, 1, 2, 1, true};
  const std::array<float, 8> expected{0, 10, 1, 11, 20, 20, 21, 21};
  if (!PatchifyF32(input.data(), input.size(), spec, output.data(), output.size()).ok() || output != expected) return false;
  const std::array<float, 2> bits{-0.0F, std::numeric_limits<float>::denorm_min()};
  std::array<float, 2> copied{};
  return PatchifyF32(bits.data(), bits.size(), {1, 1, 1, 2, 1, 1, 1, 1, false}, copied.data(), copied.size()).ok() &&
      std::memcmp(bits.data(), copied.data(), sizeof(bits)) == 0;
}
}

int main() {
  if (!TestSpatialAndTemporalOrder()) return 1;
  if (!TestValidationBeforeWrite()) return 2;
  if (!TestMultipleTemporalBlocks()) return 3;
  return 0;
}
