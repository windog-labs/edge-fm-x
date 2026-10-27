#include "vlaforge/backends/huggingface_tokenizer.h"

#include <cstddef>
#include <limits>
#include <new>

namespace {
struct TokenIds { std::uint32_t* data; std::size_t size; };
extern "C" std::uint32_t vlaforge_hf_tokenizer_new(const void*, std::size_t, void**);
extern "C" std::uint32_t vlaforge_hf_tokenizer_encode(void*, const void*, std::size_t, std::uint8_t, TokenIds*);
extern "C" void vlaforge_hf_tokenizer_free_ids(TokenIds);
extern "C" void vlaforge_hf_tokenizer_free(void*);

struct OwnedIds final {
  TokenIds value{};
  ~OwnedIds() { vlaforge_hf_tokenizer_free_ids(value); }
};
}  // namespace

namespace vlaforge::backends {
using runtime::Status;
using runtime::StatusCode;

HuggingFaceTokenizer::~HuggingFaceTokenizer() { vlaforge_hf_tokenizer_free(handle_); }

Status HuggingFaceTokenizer::Create(std::string_view json,
                                    std::unique_ptr<HuggingFaceTokenizer>* output) noexcept {
  if (output == nullptr) { return Status::Error(StatusCode::kInvalidArgument, 0u, "null tokenizer output"); }
  void* handle = nullptr;
  if (vlaforge_hf_tokenizer_new(json.data(), json.size(), &handle) != 0u || handle == nullptr) {
    return Status::Error(StatusCode::kInvalidArgument, 0u, "invalid Hugging Face tokenizer JSON");
  }
  auto* provider = new (std::nothrow) HuggingFaceTokenizer(handle);
  if (provider == nullptr) {
    vlaforge_hf_tokenizer_free(handle);
    return Status::Error(StatusCode::kResourceExhausted, 0u, "tokenizer allocation failed");
  }
  output->reset(provider);
  return Status::Ok();
}

Status HuggingFaceTokenizer::Encode(std::string_view text, bool add_special_tokens,
                                    std::vector<std::int32_t>* output) const noexcept {
  if (output == nullptr) { return Status::Error(StatusCode::kInvalidArgument, 0u, "null token output"); }
  OwnedIds ids;
  if (vlaforge_hf_tokenizer_encode(handle_, text.data(), text.size(), add_special_tokens ? 1u : 0u, &ids.value) != 0u) {
    return Status::Error(StatusCode::kInvalidArgument, 0u, "tokenization failed: invalid text, size or provider");
  }
  try {
    std::vector<std::int32_t> result;
    result.reserve(ids.value.size);
    for (std::size_t index = 0; index < ids.value.size; ++index) {
      if (ids.value.data[index] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        return Status::Error(StatusCode::kOutOfRange, 0u, "token ID exceeds INT32 range");
      }
      result.push_back(static_cast<std::int32_t>(ids.value.data[index]));
    }
    output->swap(result);
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::Error(StatusCode::kResourceExhausted, 0u, "token buffer allocation failed");
  } catch (...) {
    return Status::Error(StatusCode::kInternal, 0u, "token output construction failed");
  }
}

}  // namespace vlaforge::backends
