#include "vlaforge/backends/huggingface_tokenizer.h"

#include <memory>
#include <string>
#include <vector>

int main() {
  using vlaforge::backends::HuggingFaceTokenizer;
  const std::string json = R"({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"alpha":1,"beta":2},"unk_token":"[UNK]"}})";
  std::unique_ptr<HuggingFaceTokenizer> tokenizer;
  if (HuggingFaceTokenizer::Create(json, nullptr).ok() ||
      HuggingFaceTokenizer::Create("not JSON", &tokenizer).ok() || tokenizer != nullptr ||
      !HuggingFaceTokenizer::Create(json, &tokenizer).ok() || tokenizer == nullptr) { return 1; }
  std::vector<std::int32_t> ids;
  if (!tokenizer->Encode("alpha beta", true, &ids).ok() || ids != std::vector<std::int32_t>{1, 2}) { return 2; }
  const auto original = ids;
  const std::string invalid_utf8(1, static_cast<char>(0xff));
  if (tokenizer->Encode(invalid_utf8, true, &ids).ok() || ids != original ||
      tokenizer->Encode("alpha", true, nullptr).ok() ||
      HuggingFaceTokenizer::Create("{}", &tokenizer).ok() || tokenizer == nullptr) { return 3; }
  if (!tokenizer->Encode("gamma", false, &ids).ok() || ids != std::vector<std::int32_t>{0} ||
      !tokenizer->Encode("", true, &ids).ok() || !ids.empty()) { return 4; }
  const std::string oversized((1u << 20u) + 1u, 'a');
  if (tokenizer->Encode(oversized, true, &ids).ok() || !ids.empty()) { return 5; }
  return 0;
}
