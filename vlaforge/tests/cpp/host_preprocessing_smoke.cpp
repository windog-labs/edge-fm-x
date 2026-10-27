#include "vlaforge/runtime/host_preprocessing.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace {
bool TestRoundAndScatter() {
  using namespace vlaforge::runtime;
  const std::array<std::uint32_t, 6> bits{0x3f808000u, 0x3f818000u, 0xbf808000u, 0xbf818000u, 0x80000000u, 0x00008000u};
  const std::array<std::uint32_t, 6> expected{0x3f800000u, 0x3f820000u, 0xbf800000u, 0xbf820000u, 0x80000000u, 0u};
  std::array<float, 6> source{}, rounded{};
  std::memcpy(source.data(), bits.data(), sizeof(bits));
  if (!RoundF32ToBFloat16F32(source.data(), source.size(), rounded.data()).ok() ||
      std::memcmp(rounded.data(), expected.data(), sizeof(expected)) != 0) return false;
  auto inplace = source;
  if (RoundF32ToBFloat16F32(inplace.data(), inplace.size() - 1, inplace.data() + 1).ok() || inplace != source) return false;
  if (!RoundF32ToBFloat16F32(inplace.data(), inplace.size(), inplace.data()).ok() ||
      std::memcmp(inplace.data(), expected.data(), sizeof(expected)) != 0) return false;
  const auto preserved = rounded;
  for (float invalid : {std::numeric_limits<float>::max(), std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    auto data = source; data.back() = invalid;
    if (RoundF32ToBFloat16F32(data.data(), data.size(), rounded.data()).ok() || rounded != preserved) return false;
  }
  const std::array<float, 4> values{2, 10, 6, 14};
  const std::array<std::size_t, 2> indices{3, 1}, repeated{1, 1}, outside{3, 5};
  const std::array<float, 2> divisors{2, 2}, zero_divisor{2, 0};
  std::array<float, 10> output{}, mask{};
  if (!ScatterDivideF32(values.data(), 2, 2, indices.data(), divisors.data(), 5, true, output.data(), mask.data()).ok() ||
      output != std::array<float, 10>{0, 5, 0, 1, 0, 0, 7, 0, 3, 0} ||
      mask != std::array<float, 10>{0, 1, 0, 1, 0, 0, 1, 0, 1, 0}) return false;
  const auto previous = output, previous_mask = mask;
  if (ScatterDivideF32(values.data(), 2, 2, repeated.data(), divisors.data(), 5, true, output.data(), mask.data()).ok() ||
      ScatterDivideF32(values.data(), 2, 2, outside.data(), divisors.data(), 5, true, output.data(), mask.data()).ok() ||
      ScatterDivideF32(values.data(), 2, 2, indices.data(), zero_divisor.data(), 5, true, output.data(), mask.data()).ok() ||
      ScatterDivideF32(values.data(), std::numeric_limits<std::size_t>::max(), 2,
                       indices.data(), divisors.data(), 5, true, output.data(), mask.data()).ok() ||
      output != previous || mask != previous_mask) return false;
  auto bad_values = values; bad_values.back() = std::numeric_limits<float>::quiet_NaN();
  if (ScatterDivideF32(bad_values.data(), 2, 2, indices.data(), divisors.data(), 5, false,
                       output.data(), mask.data()).ok() || output != previous || mask != previous_mask) return false;
  if (ScatterDivideF32(values.data(), 2, 2, indices.data(), divisors.data(), 5, false,
                       output.data(), output.data()).ok()) return false;
  return ScatterDivideF32(values.data(), 2, 2, indices.data(), divisors.data(), 5, false,
                           output.data(), nullptr).ok() && output == previous;
}
}

int main() {
  using namespace vlaforge::runtime;
  if (!TestRoundAndScatter()) return 14;
  const std::array<std::uint8_t, 4> pixels{0, 1, 127, 255};
  std::array<float, 4> result{};
  if (!ScaleU8ToF32(pixels.data(), pixels.size(), 255.0F, result.data()).ok() ||
      result[0] != 0.0F || result[1] != 1.0F / 255.0F || result[3] != 1.0F) { return 1; }
  const auto preserved = result;
  for (float divisor : {0.0F, -1.0F, std::numeric_limits<float>::denorm_min(),
                        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    if (ScaleU8ToF32(pixels.data(), pixels.size(), divisor, result.data()).ok() || result != preserved) { return 2; }
  }
  const std::array<float, 4> values{2, 5, 4, 9};
  const std::array<float, 2> mean{0, 1}, stddev{2, 4};
  if (!StandardizeF32(values.data(), 2, 2, mean.data(), stddev.data(), 0, result.data()).ok() ||
      result != std::array<float, 4>{1, 1, 2, 2}) { return 3; }
  const auto normalized = result;
  for (float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::max()}) {
    auto source = values;
    source[3] = bad;
    const std::array<float, 2> tiny{1.0F, 0.01F};
    if (StandardizeF32(source.data(), 2, 2, mean.data(), tiny.data(), 0, result.data()).ok() || result != normalized) { return 4; }
  }
  const std::array<float, 2> invalid_stddev{1, -1};
  if (StandardizeF32(values.data(), 2, 2, mean.data(), invalid_stddev.data(), 0, result.data()).ok() ||
      StandardizeF32(values.data(), std::numeric_limits<std::size_t>::max(), 2,
                     mean.data(), stddev.data(), 0, result.data()).ok() || result != normalized) { return 5; }
  const std::array<std::int32_t, 5> tokens{4, 2, 3, 9, 8};
  std::array<std::int32_t, 4> ids{};
  std::array<std::uint8_t, 4> mask{};
  if (!PadTokenIds(tokens.data(), 2, 4, 17, true, SequenceSide::kRight, SequenceSide::kRight,
                   ids.data(), mask.data()).ok() || ids != std::array<std::int32_t, 4>{4, 2, 17, 17} ||
      mask != std::array<std::uint8_t, 4>{1, 1, 0, 0}) { return 6; }
  if (!PadTokenIds(tokens.data(), 2, 4, 17, true, SequenceSide::kLeft, SequenceSide::kRight,
                   ids.data(), mask.data()).ok() || ids != std::array<std::int32_t, 4>{17, 17, 4, 2} ||
      mask != std::array<std::uint8_t, 4>{0, 0, 1, 1}) { return 7; }
  if (!PadTokenIds(tokens.data(), 5, 4, 17, true, SequenceSide::kRight, SequenceSide::kLeft,
                   ids.data(), mask.data()).ok() || ids != std::array<std::int32_t, 4>{2, 3, 9, 8} ||
      mask != std::array<std::uint8_t, 4>{1, 1, 1, 1}) { return 8; }
  const auto previous = ids;
  if (PadTokenIds(tokens.data(), 5, 4, 17, false, SequenceSide::kRight, SequenceSide::kRight,
                  ids.data(), mask.data()).ok() || ids != previous) { return 9; }
  const std::array<std::int32_t, 2> negative{4, -1};
  if (PadTokenIds(negative.data(), 2, 4, 17, true, SequenceSide::kRight, SequenceSide::kRight,
                  ids.data(), mask.data()).ok() || ids != previous) { return 10; }
  if (!PadTokenIds(nullptr, 0, 4, 17, true, SequenceSide::kRight, SequenceSide::kRight,
                   ids.data(), mask.data()).ok() || ids != std::array<std::int32_t, 4>{17, 17, 17, 17} ||
      mask != std::array<std::uint8_t, 4>{0, 0, 0, 0}) { return 11; }
  std::string text;
  std::array<std::int32_t, 4> inplace{4, 2, 0, 0};
  if (!PadTokenIds(inplace.data(), 2, 4, 17, true, SequenceSide::kLeft, SequenceSide::kRight,
                   inplace.data(), mask.data()).ok() || inplace != std::array<std::int32_t, 4>{17, 17, 4, 2}) { return 13; }
  if (!EnsureTextSuffix("alpha", "\n", &text).ok() || text != "alpha\n" ||
      !EnsureTextSuffix(text, "\n", &text).ok() || text != "alpha\n" ||
      !EnsureTextSuffix("", "\n", &text).ok() || text != "\n" ||
      EnsureTextSuffix("alpha", "\n", nullptr).ok()) { return 12; }
  return 0;
}
