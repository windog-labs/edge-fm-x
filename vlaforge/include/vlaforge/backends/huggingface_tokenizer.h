#ifndef VLAFORGE_BACKENDS_HUGGINGFACE_TOKENIZER_H_
#define VLAFORGE_BACKENDS_HUGGINGFACE_TOKENIZER_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "vlaforge/runtime/status.h"

namespace vlaforge::backends {

// Optional native Rust provider. Caller serializes use; no Python, model-name
// dispatch, prompt cache or implicit padding/truncation policy.
class HuggingFaceTokenizer final {
 public:
  ~HuggingFaceTokenizer();
  HuggingFaceTokenizer(const HuggingFaceTokenizer&) = delete;
  HuggingFaceTokenizer& operator=(const HuggingFaceTokenizer&) = delete;
  static runtime::Status Create(std::string_view json,
                                std::unique_ptr<HuggingFaceTokenizer>* output) noexcept;
  runtime::Status Encode(std::string_view text, bool add_special_tokens,
                          std::vector<std::int32_t>* output) const noexcept;

 private:
  explicit HuggingFaceTokenizer(void* handle) noexcept : handle_(handle) {}
  void* handle_ = nullptr;
};

}  // namespace vlaforge::backends

#endif  // VLAFORGE_BACKENDS_HUGGINGFACE_TOKENIZER_H_
