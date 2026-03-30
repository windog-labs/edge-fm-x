#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>

namespace edge_fm {

struct HorizonModuleExport {
    std::string module_path;
    std::string module_name;
    std::string model_class;
    std::string factory_function;
    nlohmann::json lowering = nlohmann::json::object();

    nlohmann::json to_json() const;
};

HorizonModuleExport emit_horizon_module(
    const nlohmann::json& model_description,
    const nlohmann::json& model_config,
    const std::string& prefill_model_path,
    const std::string& decode_model_path,
    const std::filesystem::path& output_dir);

} // namespace edge_fm
