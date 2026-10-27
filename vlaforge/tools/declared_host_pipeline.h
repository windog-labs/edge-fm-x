#ifndef VLAFORGE_TOOLS_DECLARED_HOST_PIPELINE_H_
#define VLAFORGE_TOOLS_DECLARED_HOST_PIPELINE_H_

#include "vlaforge/runtime/host_preprocessing.h"
#include "vlaforge/runtime/region_executable.h"
#include "vlaforge/backends/image_preprocessing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <utility>
#include <vector>

namespace vlaforge::tools {

struct HostTensorSpec {
  VLAForgeDType dtype;
  std::vector<std::int64_t> shape;
};

class HostTensorBuffer final {
 public:
  explicit HostTensorBuffer(const HostTensorSpec& spec) : dtype(spec.dtype) {
    count = 1;
    for (auto dimension : spec.shape) {
      if (dimension <= 0 || count > std::numeric_limits<std::size_t>::max() / static_cast<std::uint64_t>(dimension)) {
        throw std::invalid_argument("host tensor requires bounded positive dimensions");
      }
      count *= static_cast<std::size_t>(dimension);
    }
    std::size_t item_size;
    switch (dtype) {
      case VLAFORGE_DTYPE_F64: item_size = sizeof(double); break;
      case VLAFORGE_DTYPE_F32: item_size = sizeof(float); break;
      case VLAFORGE_DTYPE_I64: item_size = sizeof(std::int64_t); break;
      case VLAFORGE_DTYPE_I32: item_size = sizeof(std::int32_t); break;
      case VLAFORGE_DTYPE_U8:
      case VLAFORGE_DTYPE_BOOL: item_size = sizeof(std::uint8_t); break;
      default: throw std::invalid_argument("unsupported native host tensor dtype");
    }
    if (count > std::numeric_limits<std::size_t>::max() / item_size) {
      throw std::invalid_argument("host tensor byte count overflows");
    }
    if (dtype == VLAFORGE_DTYPE_F32) { storage = std::vector<float>(count); }
    else if (dtype == VLAFORGE_DTYPE_F64) { storage = std::vector<double>(count); }
    else if (dtype == VLAFORGE_DTYPE_I32) { storage = std::vector<std::int32_t>(count); }
    else if (dtype == VLAFORGE_DTYPE_I64) { storage = std::vector<std::int64_t>(count); }
    else { storage = std::vector<std::uint8_t>(count); }
  }

  void* data() { return std::visit([](auto& values) -> void* { return values.data(); }, storage); }
  const void* data() const { return std::visit([](const auto& values) -> const void* { return values.data(); }, storage); }
  std::size_t bytes() const {
    return std::visit([](const auto& values) { return values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type); }, storage);
  }
  bool finite() const {
    return std::visit([this](const auto& values) {
      using Value = typename std::decay_t<decltype(values)>::value_type;
      if constexpr (std::is_floating_point_v<Value>) {
        return std::all_of(values.begin(), values.end(), [](Value value) { return std::isfinite(value); });
      } else if constexpr (std::is_same_v<Value, std::uint8_t>) {
        return dtype != VLAFORGE_DTYPE_BOOL ||
            std::all_of(values.begin(), values.end(), [](Value value) { return value <= 1u; });
      } else {
        return true;
      }
    }, storage);
  }
  VLAForgeDType dtype;
  std::size_t count = 0;

 private:
  std::variant<std::vector<std::uint8_t>, std::vector<std::int32_t>, std::vector<std::int64_t>,
               std::vector<float>, std::vector<double>> storage;
};

struct ScaleU8 { std::size_t source, target; float divisor; };
struct StandardizeF32 {
  std::size_t source, target;
  std::vector<float> mean, stddev;
  float epsilon;
};
struct FillBool { std::size_t target; bool value; };
struct CopyTensor { std::size_t source, target; };
struct ScatterDivide {
  std::size_t source, target, width;
  std::optional<std::size_t> mask;
  std::vector<std::size_t> indices;
  std::vector<float> divisors;
  bool round_bf16;
};
struct PrepareImages {
  std::vector<std::size_t> sources;
  std::size_t target;
  std::vector<std::array<std::size_t, 2>> source_sizes;
  backends::RgbResizeSpec resize;
  double rescale;
  std::array<float, 3> mean, stddev;
  bool round_bf16;
};
struct TokenizeText {
  std::size_t source, target, mask;
  std::int32_t pad_id;
  bool add_special_tokens, truncate;
  runtime::SequenceSide padding_side, truncation_side;
  std::string suffix;
};
using HostOperation = std::variant<ScaleU8, StandardizeF32, FillBool, CopyTensor, TokenizeText, ScatterDivide, PrepareImages>;
using TokenizeFn = std::function<runtime::Status(std::string_view, bool, std::vector<std::int32_t>*)>;
using ImageResizeFn = std::function<runtime::Status(const backends::RgbImageU8&, const backends::RgbResizeSpec&,
                                                   std::uint8_t*, std::size_t)>;

// The driver loads raw decoded observations outside timing. Every Prepare
// executes the declarations again; no processed-input or instruction cache.
class DeclaredHostPipeline final {
 public:
  DeclaredHostPipeline(std::vector<HostOperation> operations, TokenizeFn tokenizer, ImageResizeFn image_resize = {})
      : operations_(std::move(operations)), tokenizer_(std::move(tokenizer)), image_resize_(std::move(image_resize)) {}

  runtime::Status Prepare(const std::vector<HostTensorBuffer>& raw,
                          const std::vector<std::string>& text,
                          std::vector<HostTensorBuffer>* outputs) {
    using runtime::Status;
    using runtime::StatusCode;
    if (outputs == nullptr) { return Status::Error(StatusCode::kInvalidArgument, 0u, "host outputs are null"); }
    for (const auto& operation : operations_) {
      const auto status = std::visit([&](const auto& op) -> Status {
        using T = std::decay_t<decltype(op)>;
        if (op.target >= outputs->size()) { return Invalid(); }
        auto& destination = outputs->at(op.target);
        if constexpr (std::is_same_v<T, TokenizeText>) {
          if (op.source >= text.size() || op.mask >= outputs->size() || op.mask == op.target || !tokenizer_ ||
              destination.dtype != VLAFORGE_DTYPE_I32 ||
              (outputs->at(op.mask).dtype != VLAFORGE_DTYPE_BOOL && outputs->at(op.mask).dtype != VLAFORGE_DTYPE_I32) ||
              destination.count != outputs->at(op.mask).count) { return Invalid(); }
          std::string prompt;
          auto result = runtime::EnsureTextSuffix(text[op.source], op.suffix, &prompt);
          if (!result.ok()) { return result; }
          result = tokenizer_(prompt, op.add_special_tokens, &tokens_);
          if (!result.ok()) { return result; }
          const bool int_mask = outputs->at(op.mask).dtype == VLAFORGE_DTYPE_I32;
          if (int_mask) {
            try { mask_scratch_.resize(destination.count); }
            catch (const std::bad_alloc&) { return Status::Error(StatusCode::kResourceExhausted, 0, "token mask allocation failed"); }
          }
          auto* mask = int_mask ? mask_scratch_.data() : static_cast<std::uint8_t*>(outputs->at(op.mask).data());
          result = runtime::PadTokenIds(tokens_.data(), tokens_.size(), destination.count, op.pad_id, op.truncate,
              op.padding_side, op.truncation_side, static_cast<std::int32_t*>(destination.data()),
              mask);
          if (result.ok() && int_mask) {
            auto* output = static_cast<std::int32_t*>(outputs->at(op.mask).data());
            std::copy(mask_scratch_.begin(), mask_scratch_.end(), output);
          }
          return result;
        } else if constexpr (std::is_same_v<T, ScatterDivide>) {
          if (op.source >= raw.size() || raw[op.source].dtype != VLAFORGE_DTYPE_F32 ||
              destination.dtype != VLAFORGE_DTYPE_F32 || op.indices.empty() ||
              op.indices.size() != op.divisors.size() || op.width == 0 || destination.count % op.width != 0 ||
              raw[op.source].count % op.indices.size() != 0 ||
              raw[op.source].count / op.indices.size() != destination.count / op.width) return Invalid();
          float* mask = nullptr;
          if (op.mask) {
            if (*op.mask >= outputs->size() || *op.mask == op.target || outputs->at(*op.mask).dtype != VLAFORGE_DTYPE_F32 ||
                outputs->at(*op.mask).count != destination.count) return Invalid();
            mask = static_cast<float*>(outputs->at(*op.mask).data());
          }
          return runtime::ScatterDivideF32(static_cast<const float*>(raw[op.source].data()), destination.count / op.width,
              op.indices.size(), op.indices.data(), op.divisors.data(), op.width, op.round_bf16,
              static_cast<float*>(destination.data()), mask);
        } else if constexpr (std::is_same_v<T, PrepareImages>) {
          if (!image_resize_ || destination.dtype != VLAFORGE_DTYPE_F32 || op.sources.empty() ||
              op.source_sizes.size() != op.sources.size() || op.resize.width == 0 || op.resize.height == 0 ||
              op.resize.width > std::numeric_limits<std::size_t>::max() / 3 / op.resize.height ||
              !std::isfinite(op.rescale) || op.rescale <= 0) return Invalid();
          const auto plane = op.resize.width * op.resize.height, count = plane * 3;
          if (destination.count % count != 0 || destination.count / count != op.sources.size()) return Invalid();
          for (std::size_t channel = 0; channel < 3; ++channel) {
            if (!std::isfinite(op.mean[channel]) || !std::isfinite(op.stddev[channel]) || op.stddev[channel] <= 0) return Invalid();
            for (double endpoint : {0.0, 255.0}) {
              float value = (static_cast<float>(endpoint * op.rescale) - op.mean[channel]) / op.stddev[channel];
              if (!std::isfinite(value) || (op.round_bf16 && !runtime::RoundF32ToBFloat16F32(&value, 1, &value).ok())) return Invalid();
            }
          }
          for (std::size_t image = 0; image < op.sources.size(); ++image) {
            const auto source = op.sources[image], width = op.source_sizes[image][0], height = op.source_sizes[image][1];
            if (source >= raw.size() || raw[source].dtype != VLAFORGE_DTYPE_U8 || width == 0 || height == 0 ||
                width > std::numeric_limits<std::size_t>::max() / 3 / height || raw[source].count != width * height * 3) return Invalid();
          }
          try { image_scratch_.resize(count); }
          catch (const std::bad_alloc&) { return Status::Error(StatusCode::kResourceExhausted, 0, "image scratch allocation failed"); }
          auto* data = static_cast<float*>(destination.data());
          for (std::size_t image = 0; image < op.sources.size(); ++image) {
            const auto& source = raw[op.sources[image]];
            const backends::RgbImageU8 view{static_cast<const std::uint8_t*>(source.data()),
                op.source_sizes[image][0], op.source_sizes[image][1], source.bytes()};
            auto result = image_resize_(view, op.resize, image_scratch_.data(), count);
            if (!result.ok()) return result;
            for (std::size_t pixel = 0; pixel < plane; ++pixel) {
              for (std::size_t channel = 0; channel < 3; ++channel) {
                const float scaled = static_cast<float>(static_cast<double>(image_scratch_[pixel * 3 + channel]) * op.rescale);
                data[image * count + channel * plane + pixel] = (scaled - op.mean[channel]) / op.stddev[channel];
              }
            }
            if (op.round_bf16) {
              result = runtime::RoundF32ToBFloat16F32(data + image * count, count, data + image * count);
              if (!result.ok()) return result;
            }
          }
          return Status::Ok();
        } else if constexpr (std::is_same_v<T, FillBool>) {
          if (destination.dtype != VLAFORGE_DTYPE_BOOL) { return Invalid(); }
          std::memset(destination.data(), op.value ? 1 : 0, destination.bytes());
          return Status::Ok();
        } else {
          if (op.source >= raw.size() || raw[op.source].count != destination.count) { return Invalid(); }
          const auto& source = raw[op.source];
          if constexpr (std::is_same_v<T, ScaleU8>) {
            if (source.dtype != VLAFORGE_DTYPE_U8 || destination.dtype != VLAFORGE_DTYPE_F32) { return Invalid(); }
            return runtime::ScaleU8ToF32(static_cast<const std::uint8_t*>(source.data()), source.count,
                                         op.divisor, static_cast<float*>(destination.data()));
          } else if constexpr (std::is_same_v<T, StandardizeF32>) {
            if (source.dtype != VLAFORGE_DTYPE_F32 || destination.dtype != VLAFORGE_DTYPE_F32 ||
                op.mean.empty() || op.stddev.size() != op.mean.size() || source.count % op.mean.size() != 0) { return Invalid(); }
            return runtime::StandardizeF32(static_cast<const float*>(source.data()), source.count / op.mean.size(),
                op.mean.size(), op.mean.data(), op.stddev.data(), op.epsilon, static_cast<float*>(destination.data()));
          } else {
            if (source.dtype != destination.dtype || source.bytes() != destination.bytes()) { return Invalid(); }
            std::memcpy(destination.data(), source.data(), source.bytes());
            return Status::Ok();
          }
        }
      }, operation);
      if (!status.ok()) { return status; }
    }
    return Status::Ok();
  }

 private:
  static runtime::Status Invalid() {
    return runtime::Status::Error(runtime::StatusCode::kInvalidArgument, 0u, "native host operation contract mismatch");
  }
  std::vector<HostOperation> operations_;
  TokenizeFn tokenizer_;
  ImageResizeFn image_resize_;
  std::vector<std::int32_t> tokens_;
  std::vector<std::uint8_t> mask_scratch_, image_scratch_;
};

}  // namespace vlaforge::tools

#endif  // VLAFORGE_TOOLS_DECLARED_HOST_PIPELINE_H_
