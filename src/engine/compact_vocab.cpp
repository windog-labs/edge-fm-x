#include "engine/compact_vocab.h"

#include "utils/check.h"

#include <fstream>
#include <sstream>
#include <tuple>

namespace edge_fm {

namespace {

constexpr const char* kCompactVocabFormat = "edgefm.compact_vocab.v1";

nlohmann::json load_json_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw ConfigurationError("Cannot open compact_vocab mapping file: " + path.string());
    }

    nlohmann::json payload;
    try {
        file >> payload;
    } catch (const nlohmann::json::parse_error& exc) {
        throw ConfigurationError(
            "Failed to parse compact_vocab mapping file " + path.string() + ": " + exc.what());
    }
    return payload;
}

int32_t required_int(const nlohmann::json& payload, const std::string& key) {
    check<ConfigurationError>(
        payload.contains(key) && payload[key].is_number_integer(),
        "compact_vocab mapping requires integer field: " + key);
    return payload[key].get<int32_t>();
}

std::vector<int32_t> required_int_array(const nlohmann::json& payload, const std::string& key) {
    check<ConfigurationError>(
        payload.contains(key) && payload[key].is_array(),
        "compact_vocab mapping requires integer array field: " + key);

    std::vector<int32_t> out;
    out.reserve(payload[key].size());
    for (const auto& item : payload[key]) {
        check<ConfigurationError>(
            item.is_number_integer(),
            "compact_vocab mapping field " + key + " must contain only integers");
        out.push_back(item.get<int32_t>());
    }
    return out;
}

std::filesystem::path resolve_mapping_path(const EngineConfig& config, const std::string& raw_path) {
    check<ConfigurationError>(
        !raw_path.empty(),
        "compact_vocab.enabled=true requires compact_vocab.mapping_path");
    std::filesystem::path path(raw_path);
    if (path.is_absolute()) {
        return path.lexically_normal();
    }
    return (config.config_dir() / path).lexically_normal();
}

void validate_model_vocab_size(const EngineConfig& config, int32_t compact_vocab_size) {
    const auto validate_one = [&](const nlohmann::json& model_config, const std::string& name) {
        check<ConfigurationError>(
            model_config.contains("vocab_size") && model_config["vocab_size"].is_number_integer(),
            "compact_vocab requires " + name + ".config.json to define integer vocab_size");
        const int32_t model_vocab_size = model_config["vocab_size"].get<int32_t>();
        check<ConfigurationError>(
            model_vocab_size == compact_vocab_size,
            "compact_vocab v1 expects pre-pruned checkpoint vocab_size == compact_vocab_size. " +
                name + " vocab_size=" + std::to_string(model_vocab_size) +
                ", compact_vocab_size=" + std::to_string(compact_vocab_size));
    };

    validate_one(config.prefill_model_config(), "prefill_model");
    validate_one(config.decode_model_config(), "decode_model");
}

std::string ids_context(const std::string& context, int32_t id) {
    return context + " token id " + std::to_string(id);
}

} // namespace

CompactVocab::CompactVocab(const EngineConfig& config) {
    const nlohmann::json compact_config = config.compact_vocab();
    enabled_ = compact_config.value("enabled", false);
    if (!enabled_) {
        return;
    }

    reject_unknown_input_ids_ = compact_config.value("reject_unknown_input_ids", true);
    check<ConfigurationError>(
        reject_unknown_input_ids_,
        "compact_vocab v1 requires reject_unknown_input_ids=true because no unknown-token remap is defined");

    mapping_path_ = resolve_mapping_path(config, compact_config.value("mapping_path", std::string("")));
    nlohmann::json mapping = load_json_file(mapping_path_);

    check<ConfigurationError>(
        mapping.value("format", std::string("")) == kCompactVocabFormat,
        "compact_vocab mapping format must be " + std::string(kCompactVocabFormat));

    original_vocab_size_ = required_int(mapping, "original_vocab_size");
    compact_vocab_size_ = required_int(mapping, "compact_vocab_size");
    check<ConfigurationError>(original_vocab_size_ > 0, "original_vocab_size must be positive");
    check<ConfigurationError>(compact_vocab_size_ > 0, "compact_vocab_size must be positive");
    check<ConfigurationError>(
        compact_vocab_size_ <= original_vocab_size_,
        "compact_vocab_size must be <= original_vocab_size");

    old_to_new_ = required_int_array(mapping, "old_to_new");
    new_to_old_ = required_int_array(mapping, "new_to_old");
    check<ConfigurationError>(
        static_cast<int32_t>(old_to_new_.size()) == original_vocab_size_,
        "old_to_new length must equal original_vocab_size");
    check<ConfigurationError>(
        static_cast<int32_t>(new_to_old_.size()) == compact_vocab_size_,
        "new_to_old length must equal compact_vocab_size");

    for (int32_t old_id = 0; old_id < original_vocab_size_; ++old_id) {
        int32_t new_id = old_to_new_[old_id];
        check<ConfigurationError>(
            new_id >= -1 && new_id < compact_vocab_size_,
            "old_to_new contains out-of-range compact id at original id " + std::to_string(old_id));
        if (new_id >= 0) {
            check<ConfigurationError>(
                new_to_old_[new_id] == old_id,
                "old_to_new/new_to_old mismatch at original id " + std::to_string(old_id));
        }
    }
    for (int32_t new_id = 0; new_id < compact_vocab_size_; ++new_id) {
        int32_t old_id = new_to_old_[new_id];
        check<ConfigurationError>(
            old_id >= 0 && old_id < original_vocab_size_,
            "new_to_old contains out-of-range original id at compact id " + std::to_string(new_id));
        check<ConfigurationError>(
            old_to_new_[old_id] == new_id,
            "new_to_old/old_to_new mismatch at compact id " + std::to_string(new_id));
    }

    validate_model_vocab_size(config, compact_vocab_size_);

    std::vector<int32_t> special_token_ids = required_int_array(mapping, "special_token_ids");
    (void)remap_required_token_ids(special_token_ids, "compact_vocab.special_token_ids");
    (void)remap_required_token_ids(config.eos_token_ids(), "model eos_token_ids");
    (void)remap_required_token_ids(config.stop_token_ids(), "sampling.stop_token_ids");
}

int32_t CompactVocab::remap_input_token_id(int32_t id, const std::string& context) const {
    if (!enabled_) {
        return id;
    }
    if (id < 0 || id >= original_vocab_size_ || old_to_new_[id] < 0) {
        throw InvalidRequestError(
            "compact_vocab cannot remap " + ids_context(context, id) +
            "; the token is absent from old_to_new");
    }
    return old_to_new_[id];
}

int32_t CompactVocab::remap_required_token_id(int32_t id, const std::string& context) const {
    if (!enabled_) {
        return id;
    }
    check<ConfigurationError>(
        id >= 0 && id < original_vocab_size_ && old_to_new_[id] >= 0,
        "compact_vocab requires " + ids_context(context, id) + " to be present in old_to_new");
    return old_to_new_[id];
}

int32_t CompactVocab::restore_token_id(int32_t id, const std::string& context) const {
    if (!enabled_) {
        return id;
    }
    if (id < 0 || id >= compact_vocab_size_) {
        throw InternalError(
            "compact_vocab cannot restore " + ids_context(context, id) +
            "; compact id is out of range");
    }
    return new_to_old_[id];
}

std::vector<int32_t> CompactVocab::remap_token_ids(
    const std::vector<int32_t>& ids,
    const std::string& context) const
{
    if (!enabled_) {
        return ids;
    }

    std::vector<int32_t> out;
    out.reserve(ids.size());
    for (int32_t id : ids) {
        out.push_back(remap_input_token_id(id, context));
    }
    return out;
}

std::vector<int32_t> CompactVocab::remap_required_token_ids(
    const std::vector<int32_t>& ids,
    const std::string& context) const
{
    if (!enabled_) {
        return ids;
    }

    std::vector<int32_t> out;
    out.reserve(ids.size());
    for (int32_t id : ids) {
        out.push_back(remap_required_token_id(id, context));
    }
    return out;
}

std::vector<int32_t> CompactVocab::remap_optional_token_ids(
    const std::vector<int32_t>& ids,
    const std::string& context) const
{
    return remap_required_token_ids(ids, context);
}

std::vector<int32_t> CompactVocab::restore_token_ids(
    const std::vector<int32_t>& ids,
    const std::string& context) const
{
    if (!enabled_) {
        return ids;
    }

    std::vector<int32_t> out;
    out.reserve(ids.size());
    for (int32_t id : ids) {
        out.push_back(restore_token_id(id, context));
    }
    return out;
}

int32_t CompactVocab::remap_embed_token_id(const Request& request) const {
    if (!request.has_embedding() || request.embed_token_id() < 0) {
        return request.embed_token_id();
    }

    const auto& shape = request.embedding().shape();
    check<InvalidRequestError>(
        !shape.empty() && shape[0] >= 0,
        "compact_vocab requires embeddings to expose num_custom_embeddings in shape[0]");
    const int32_t num_custom_embeddings = static_cast<int32_t>(shape[0]);
    const int32_t old_base = request.embed_token_id();
    const int32_t new_base = remap_input_token_id(old_base, "request embed_token_id");
    for (int32_t i = 0; i < num_custom_embeddings; ++i) {
        const int32_t old_id = old_base + i;
        const int32_t new_id = remap_input_token_id(old_id, "request embedding placeholder");
        if (new_id != new_base + i) {
            throw InvalidRequestError(
                "compact_vocab requires custom embedding placeholder ids to remap to a contiguous range. "
                "old_base=" + std::to_string(old_base) +
                ", compact_base=" + std::to_string(new_base) +
                ", offset=" + std::to_string(i) +
                ", compact_id=" + std::to_string(new_id));
        }
    }
    return new_base;
}

Request CompactVocab::remap_request(
    const Request& request,
    Device device,
    int32_t device_id,
    MemoryOwnership ownership,
    void* stream_handle) const
{
    check<InvalidRequestError>(enabled_, "remap_request requires compact_vocab to be enabled");

    std::vector<int32_t> remapped_token_ids = remap_token_ids(request.token_ids(), "request token_ids");
    const int32_t remapped_embed_token_id = remap_embed_token_id(request);

    Request remapped = [&]() -> Request {
        if (request.has_position_ids()) {
            check<InvalidRequestError>(
                request.has_embedding(),
                "compact_vocab VLM/custom position_ids requests require embeddings");
            if (request.has_mrope_last_pos()) {
                return Request(
                    request.request_id(),
                    remapped_token_ids,
                    request.embedding(),
                    remapped_embed_token_id,
                    request.position_ids(),
                    request.mrope_last_pos(),
                    device,
                    device_id,
                    ownership,
                    stream_handle);
            }
            return Request(
                request.request_id(),
                remapped_token_ids,
                request.embedding(),
                remapped_embed_token_id,
                request.position_ids(),
                device,
                device_id,
                ownership,
                stream_handle);
        }
        if (request.has_embedding()) {
            return Request(
                request.request_id(),
                remapped_token_ids,
                request.embedding(),
                remapped_embed_token_id,
                device,
                device_id,
                ownership,
                stream_handle);
        }
        return Request(request.request_id(), remapped_token_ids);
    }();

    remapped.set_stop_token_ids(remap_token_ids(request.stop_token_ids(), "request stop_token_ids"));
    remapped.set_ignore_stop_tokens(request.ignore_stop_tokens());
    return remapped;
}

} // namespace edge_fm
