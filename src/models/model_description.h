#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace edge_fm {

struct CompiledOp {
    std::string name;
    std::string op_type;
    std::string backend;
    nlohmann::json attrs = nlohmann::json::object();

    nlohmann::json to_json() const;
    static CompiledOp from_json(const nlohmann::json& json);
};

struct ExecutionPlan {
    std::string model_description_hash;
    bool uses_inject_embedding = false;
    bool uses_fused_qkv = true;
    bool uses_fused_gate_up = true;
    bool uses_mrope = false;
    std::vector<CompiledOp> prefill_ops;
    std::vector<CompiledOp> decode_ops;

    nlohmann::json to_json() const;
    static ExecutionPlan from_json(const nlohmann::json& json);
};

std::string hash_model_description(const nlohmann::json& model_description);
ExecutionPlan compile_model_description(
    const nlohmann::json& model_description,
    const nlohmann::json& model_config);

} // namespace edge_fm
