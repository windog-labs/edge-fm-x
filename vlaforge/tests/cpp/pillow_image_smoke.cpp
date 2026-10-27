#include "vlaforge/backends/image_preprocessing.h"

#include <array>
#include <atomic>
#include <thread>
#include <vector>

int main() {
  using namespace vlaforge::backends;
  const std::array<std::uint8_t, 6> input{10, 20, 30, 40, 50, 60};
  const RgbImageU8 image{input.data(), 2, 1, input.size()};
  const RgbResizeSpec spec{2, 2, true, {70, 80, 90}, ImageFilter::kBicubic};
  std::array<std::uint8_t, 12> output{};
  const std::array<std::uint8_t, 12> expected{10, 20, 30, 40, 50, 60, 70, 80, 90, 70, 80, 90};
  if (!ResizeRgbWithPillow(image, spec, output.data(), output.size()).ok() || output != expected) return 1;
  const auto saved = output;
  auto invalid = spec; invalid.filter = static_cast<ImageFilter>(255);
  if (ResizeRgbWithPillow(image, invalid, output.data(), output.size()).ok() || output != saved ||
      ResizeRgbWithPillow(image, spec, output.data(), output.size() - 1).ok() || output != saved) return 2;
  auto truncated = image; truncated.bytes -= 1;
  if (ResizeRgbWithPillow(truncated, spec, output.data(), output.size()).ok() || output != saved) return 3;
  std::atomic<bool> passed{true};
  std::vector<std::thread> workers;
  for (int thread = 0; thread < 4; ++thread) {
    workers.emplace_back([&] {
      const std::array<std::uint8_t, 3> pixel{23, 117, 241};
      std::array<std::uint8_t, 27> expanded{};
      for (auto filter : {ImageFilter::kBilinear, ImageFilter::kBicubic, ImageFilter::kLanczos}) {
        for (int run = 0; run < 10; ++run) {
          const RgbResizeSpec target{3, 3, false, {0, 0, 0}, filter};
          if (!ResizeRgbWithPillow({pixel.data(), 1, 1, pixel.size()}, target, expanded.data(), expanded.size()).ok()) passed = false;
          for (std::size_t index = 0; index < expanded.size(); ++index) if (expanded[index] != pixel[index % 3]) passed = false;
        }
      }
    });
  }
  for (auto& worker : workers) worker.join();
  return passed ? 0 : 4;
}
