#pragma once

#include <nlohmann/json.hpp>

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace edge_fm {

struct OperatorImplRecord {
    std::string model_name;
    std::string hw_profile;
    std::string op_kind;
    std::string layer_role;
    std::string op_name;
    std::string stage;
    std::string shape_sig;
    std::string impl_id;
    nlohmann::json impl_params = nlohmann::json::object();

    nlohmann::json to_json() const;
    static OperatorImplRecord from_json(const nlohmann::json& json);
};

struct OperatorQuery {
    std::string op_kind;
    std::string layer_role;
    std::string op_name;
    std::string stage;
    std::string shape_sig;
};

class OperatorImplTable {
public:
    static OperatorImplTable& instance();

    std::optional<OperatorImplRecord> resolve(
        const std::string& model_name,
        const std::string& hw_profile,
        const std::string& table_path,
        const OperatorQuery& query);

    std::vector<OperatorImplRecord> records_for_model(
        const std::string& model_name,
        const std::string& hw_profile,
        const std::string& table_path,
        const std::string& op_kind = std::string());

    std::vector<OperatorImplRecord> all_records(const std::string& table_path);
    void invalidate(const std::string& table_path);

private:
    OperatorImplTable() = default;

    struct LoadedTable {
        std::vector<OperatorImplRecord> records;
    };

    const LoadedTable& load_table_locked(const std::string& table_path);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, LoadedTable> tables_;
};

} // namespace edge_fm
