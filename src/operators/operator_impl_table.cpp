#include "operators/operator_impl_table.h"

#include <edge-fm/core.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace edge_fm {

namespace {

std::string normalize_identifier(const std::string& raw) {
    std::string normalized;
    normalized.reserve(raw.size());
    for (unsigned char ch : raw) {
        if (std::isalnum(ch)) {
            normalized.push_back(static_cast<char>(std::tolower(ch)));
        } else if (ch == '_' || ch == '.' || ch == '-' || ch == ' ' || ch == '/' || ch == ':') {
            normalized.push_back('_');
        }
    }
    return normalized;
}

std::string normalize_stage_key(const std::string& raw) {
    const std::string normalized = normalize_identifier(raw);
    if (normalized == "prefill") {
        return "prefill";
    }
    if (normalized == "decode") {
        return "decode";
    }
    return normalized;
}

std::string normalize_model_name(const std::string& raw) {
    const std::string normalized = normalize_identifier(raw);
    if (normalized == "qwen2_5" || normalized == "qwen25" || normalized == "qwen2") {
        return "qwen2_5";
    }
    if (normalized == "qwen2_5_vl" || normalized == "qwen25_vl" || normalized == "qwen2_vl" ||
        normalized == "qwen25vl" || normalized == "qwen2_5vl" || normalized == "qwen2vl") {
        return "qwen2_5_vl";
    }
    return normalized;
}

std::string stage_key_from_json(const nlohmann::json& json) {
    return normalize_stage_key(json.value("stage", std::string("")));
}

std::vector<OperatorImplRecord> builtin_records() {
    std::vector<OperatorImplRecord> records;

    auto add_record = [&](const std::string& model_name, const std::string& op_kind, const std::string& impl_id) {
        OperatorImplRecord record;
        record.model_name = model_name;
        record.hw_profile = "cuda";
        record.op_kind = op_kind;
        record.impl_id = impl_id;
        record.impl_params = nlohmann::json::object();
        records.push_back(std::move(record));
    };

    for (const std::string& model_name : {std::string("qwen2_5"), std::string("qwen2_5_vl")}) {
        add_record(model_name, "linear", "cublasLt");
        add_record(model_name, "attention", "flashinfer_attention");
        add_record(model_name, "norm", "flashinfer_norm");
        add_record(model_name, "activation", "flashinfer_silu_and_mul");
    }

    return records;
}

int match_exact_or_wildcard(
    const std::string& pattern,
    const std::string& query,
    int exact_score)
{
    if (pattern.empty() || pattern == "*") {
        return 0;
    }
    if (pattern == query) {
        return exact_score;
    }
    return -1;
}

int match_hw_profile(const std::string& pattern, const std::string& query) {
    if (pattern.empty() || pattern == "*") {
        return 0;
    }
    if (pattern == query) {
        return 200;
    }
    if (query.rfind(pattern + "_", 0) == 0) {
        return 120;
    }
    return -1;
}

std::string table_cache_key(const std::string& table_path) {
    if (table_path.empty()) {
        return "<builtin>";
    }
    return std::filesystem::absolute(std::filesystem::path(table_path)).lexically_normal().string();
}

bool matches_table_scope(
    const OperatorImplRecord& record,
    const std::string& normalized_model_name,
    const std::string& normalized_hw_profile,
    const std::string& normalized_op_kind)
{
    const int model_score = match_exact_or_wildcard(record.model_name, normalized_model_name, 1);
    if (model_score < 0) {
        return false;
    }
    const int hw_score = match_hw_profile(record.hw_profile, normalized_hw_profile);
    if (hw_score < 0) {
        return false;
    }
    if (normalized_op_kind.empty()) {
        return true;
    }
    const int op_kind_score = match_exact_or_wildcard(record.op_kind, normalized_op_kind, 1);
    return op_kind_score >= 0;
}

} // namespace

nlohmann::json OperatorImplRecord::to_json() const {
    return nlohmann::json{
        {"model_name", model_name},
        {"hw_profile", hw_profile},
        {"op_kind", op_kind},
        {"layer_role", layer_role},
        {"op_name", op_name},
        {"stage", stage},
        {"shape_sig", shape_sig},
        {"impl_id", impl_id},
        {"impl_params", impl_params},
    };
}

OperatorImplRecord OperatorImplRecord::from_json(const nlohmann::json& json) {
    OperatorImplRecord record;
    record.model_name = normalize_model_name(json.value("model_name", std::string("")));
    record.hw_profile = normalize_identifier(json.value("hw_profile", std::string("")));
    record.op_kind = normalize_identifier(json.value("op_kind", std::string("")));
    record.layer_role = normalize_identifier(json.value("layer_role", std::string("")));
    record.op_name = normalize_identifier(json.value("op_name", std::string("")));
    record.stage = stage_key_from_json(json);
    record.shape_sig = json.value("shape_sig", std::string(""));
    record.impl_id = json.value("impl_id", std::string(""));
    record.impl_params = json.contains("impl_params") && json["impl_params"].is_object()
        ? json["impl_params"]
        : nlohmann::json::object();
    return record;
}

OperatorImplTable& OperatorImplTable::instance() {
    static OperatorImplTable table;
    return table;
}

std::optional<OperatorImplRecord> OperatorImplTable::resolve(
    const std::string& model_name,
    const std::string& hw_profile,
    const std::string& table_path,
    const OperatorQuery& query)
{
    const std::string normalized_model_name = normalize_model_name(model_name);
    const std::string normalized_hw_profile = normalize_identifier(hw_profile);
    const std::string normalized_op_kind = normalize_identifier(query.op_kind);
    const std::string normalized_layer_role = normalize_identifier(query.layer_role);
    const std::string normalized_op_name = normalize_identifier(query.op_name);
    const std::string normalized_stage = normalize_stage_key(query.stage);

    std::lock_guard<std::mutex> lock(mutex_);
    const LoadedTable& table = load_table_locked(table_path);

    std::optional<OperatorImplRecord> best_match;
    int best_score = -1;
    for (const auto& record : table.records) {
        if (!matches_table_scope(record, normalized_model_name, normalized_hw_profile, normalized_op_kind)) {
            continue;
        }

        const int op_name_score = match_exact_or_wildcard(record.op_name, normalized_op_name, 1000000);
        if (op_name_score < 0) {
            continue;
        }
        const int layer_role_score = match_exact_or_wildcard(record.layer_role, normalized_layer_role, 100000);
        if (layer_role_score < 0) {
            continue;
        }
        const int shape_score = match_exact_or_wildcard(record.shape_sig, query.shape_sig, 10000);
        if (shape_score < 0) {
            continue;
        }
        const int stage_score = match_exact_or_wildcard(record.stage, normalized_stage, 1000);
        if (stage_score < 0) {
            continue;
        }
        const int hw_score = match_hw_profile(record.hw_profile, normalized_hw_profile);
        const int model_score = match_exact_or_wildcard(record.model_name, normalized_model_name, 100);
        const int op_kind_score = match_exact_or_wildcard(record.op_kind, normalized_op_kind, 10);

        const int total_score = op_name_score + layer_role_score + shape_score + stage_score +
            hw_score + model_score + op_kind_score;

        if (total_score >= best_score) {
            best_match = record;
            best_score = total_score;
        }
    }

    return best_match;
}

std::vector<OperatorImplRecord> OperatorImplTable::records_for_model(
    const std::string& model_name,
    const std::string& hw_profile,
    const std::string& table_path,
    const std::string& op_kind)
{
    const std::string normalized_model_name = normalize_model_name(model_name);
    const std::string normalized_hw_profile = normalize_identifier(hw_profile);
    const std::string normalized_op_kind = normalize_identifier(op_kind);

    std::vector<OperatorImplRecord> matched;
    std::lock_guard<std::mutex> lock(mutex_);
    const LoadedTable& table = load_table_locked(table_path);
    for (const auto& record : table.records) {
        if (matches_table_scope(record, normalized_model_name, normalized_hw_profile, normalized_op_kind)) {
            matched.push_back(record);
        }
    }
    return matched;
}

const OperatorImplTable::LoadedTable& OperatorImplTable::load_table_locked(const std::string& table_path) {
    const std::string key = table_cache_key(table_path);
    auto it = tables_.find(key);
    if (it != tables_.end()) {
        return it->second;
    }

    LoadedTable loaded;
    loaded.records = builtin_records();

    if (!table_path.empty()) {
        std::ifstream input(table_path);
        if (!input.is_open()) {
            throw ConfigurationError("Cannot open operator_impl_table file: " + table_path);
        }

        nlohmann::json json;
        try {
            input >> json;
        } catch (const nlohmann::json::parse_error& e) {
            throw ConfigurationError(
                "Failed to parse operator_impl_table file: " + table_path + ", " + std::string(e.what()));
        }

        if (!json.contains("records") || !json["records"].is_array()) {
            throw ConfigurationError("operator_impl_table must contain a records array: " + table_path);
        }
        for (const auto& item : json["records"]) {
            loaded.records.push_back(OperatorImplRecord::from_json(item));
        }
    }

    auto [inserted_it, inserted] = tables_.emplace(key, std::move(loaded));
    (void)inserted;
    return inserted_it->second;
}

} // namespace edge_fm
