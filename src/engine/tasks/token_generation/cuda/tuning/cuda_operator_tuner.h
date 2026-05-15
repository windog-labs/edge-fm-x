#pragma once

#include <filesystem>

namespace edge_fm {

class EngineConfig;
class Model;

struct CudaOperatorTuningResult {
    std::filesystem::path tuning_dir;
    std::filesystem::path operator_table_path;
    std::filesystem::path report_path;
    bool cache_hit = false;
};

CudaOperatorTuningResult tune_cuda_operator_table(EngineConfig& config, Model& model);

} // namespace edge_fm
