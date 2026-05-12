#pragma once

#include "engine/engine.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace edge_fm {

class CompactVocab {
public:
    CompactVocab() = default;
    explicit CompactVocab(const EngineConfig& config);

    bool enabled() const noexcept { return enabled_; }
    int32_t original_vocab_size() const noexcept { return original_vocab_size_; }
    int32_t compact_vocab_size() const noexcept { return compact_vocab_size_; }
    const std::filesystem::path& mapping_path() const noexcept { return mapping_path_; }

    std::vector<int32_t> remap_token_ids(
        const std::vector<int32_t>& ids,
        const std::string& context) const;
    std::vector<int32_t> remap_required_token_ids(
        const std::vector<int32_t>& ids,
        const std::string& context) const;
    std::vector<int32_t> remap_optional_token_ids(
        const std::vector<int32_t>& ids,
        const std::string& context) const;
    std::vector<int32_t> restore_token_ids(
        const std::vector<int32_t>& ids,
        const std::string& context) const;

    Request remap_request(
        const Request& request,
        Device device,
        int32_t device_id,
        MemoryOwnership ownership = MemoryOwnership::OwnCudaMalloc,
        void* stream_handle = nullptr) const;

private:
    int32_t remap_input_token_id(int32_t id, const std::string& context) const;
    int32_t remap_required_token_id(int32_t id, const std::string& context) const;
    int32_t restore_token_id(int32_t id, const std::string& context) const;
    int32_t remap_embed_token_id(const Request& request) const;

    bool enabled_ = false;
    bool reject_unknown_input_ids_ = true;
    int32_t original_vocab_size_ = 0;
    int32_t compact_vocab_size_ = 0;
    std::filesystem::path mapping_path_;
    std::vector<int32_t> old_to_new_;
    std::vector<int32_t> new_to_old_;
};

} // namespace edge_fm
