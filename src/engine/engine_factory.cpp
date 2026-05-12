#include "engine/engine_factory.h"

#include "engine/horizon/engine.h"

#if defined(EDGE_FM_ENABLE_CUDA)
#include "engine/cuda/standard_engine.h"
#include "utils/device/weight_loader.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <vector>
#endif

#include <stdexcept>
#include <string>

namespace edge_fm {

#if defined(EDGE_FM_ENABLE_CUDA)
namespace {

std::vector<std::string> collect_safetensors_files(const std::string& model_dir) {
    std::filesystem::path dir(model_dir);
    std::vector<std::string> out;
    const std::string single = model_dir + "/model.safetensors";
    if (std::filesystem::exists(single) && std::filesystem::is_regular_file(single)) {
        out.push_back(single);
        return out;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string name = entry.path().filename().string();
        if (name.size() > 18 && name.compare(0, 6, "model-") == 0 &&
            name.find("-of-") != std::string::npos &&
            name.size() >= 12 && name.compare(name.size() - 12, 12, ".safetensors") == 0) {
            out.push_back(entry.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

void load_cuda_weights(const EngineConfig& config) {
    WeightLoader& loader = WeightLoader::instance();
    loader.clear_stage(ModelStage::Prefill);
    loader.clear_stage(ModelStage::Decode);

    const std::string prefill_path = config.prefill_model_path();
    const int32_t device_id = config.runtime_device_id();
    const bool is_vlm = config.resolved_model_name() == "qwen2_5_vl";

    std::vector<std::string> prefill_files = collect_safetensors_files(prefill_path);
    if (prefill_files.empty()) {
        throw ConfigurationError("No model.safetensors or model-*-of-*.safetensors found in: " + prefill_path);
    }

    const std::string decode_path = config.decode_model_path();
    const bool share_prefill_decode_weights = decode_path.empty() || decode_path == prefill_path;
    std::vector<std::string> decode_files = share_prefill_decode_weights
        ? std::vector<std::string>{}
        : collect_safetensors_files(decode_path);
    if (!share_prefill_decode_weights && decode_files.empty()) {
        throw ConfigurationError("No model.safetensors or model-*-of-*.safetensors found in decode path");
    }

    if (is_vlm) {
        auto vlm_filter = [](const std::string& name) {
            return name.rfind("model.", 0) == 0 ||
                   name.rfind("language_model.", 0) == 0 ||
                   name.rfind("lm_head.", 0) == 0;
        };
        auto vlm_key_mapper = [](const std::string& name) {
            if (name.rfind("model.model.", 0) == 0) {
                return name.substr(6);
            }
            if (name.rfind("language_model.", 0) == 0) {
                return name.substr(std::string("language_model.").size());
            }
            return name;
        };
        for (const auto& file : prefill_files) {
            loader.load_weights_from_file(
                ModelStage::Prefill, file, Device::GPU, device_id, true, vlm_filter, vlm_key_mapper);
        }
        if (!share_prefill_decode_weights) {
            for (const auto& file : decode_files) {
                loader.load_weights_from_file(
                    ModelStage::Decode, file, Device::GPU, device_id, true, vlm_filter, vlm_key_mapper);
            }
        }
        return;
    }

    for (const auto& file : prefill_files) {
        loader.load_weights_from_file(ModelStage::Prefill, file, Device::GPU, device_id, true);
    }
    if (!share_prefill_decode_weights) {
        for (const auto& file : decode_files) {
            loader.load_weights_from_file(ModelStage::Decode, file, Device::GPU, device_id, true);
        }
    }
}

} // namespace
#endif

std::unique_ptr<Engine> create_horizon_engine(const EngineConfig& config) {
    return std::make_unique<HorizonEngine>(config);
}

#if defined(EDGE_FM_ENABLE_CUDA)
std::unique_ptr<Engine> create_cuda_engine(const EngineConfig& config) {
    load_cuda_weights(config);
    auto engine = std::make_unique<StandardEngine>(config);
    if (config.tuning_enabled()) {
        engine->tune();
    }
    return engine;
}

std::string cuda_hardware_fingerprint(int32_t device_id) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        return "unknown-device-" + std::to_string(device_id);
    }

    std::ostringstream oss;
    oss << prop.name << "|cc=" << prop.major << "." << prop.minor;
    return oss.str();
}

std::string cuda_hw_profile(int32_t device_id) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        return "cuda_sm" + std::to_string(prop.major) + std::to_string(prop.minor);
    }
    return "cuda";
}
#else
std::unique_ptr<Engine> create_cuda_engine(const EngineConfig&) {
    throw ConfigurationError("CUDA backend is not compiled");
}

std::string cuda_hardware_fingerprint(int32_t device_id) {
    return "cuda-not-compiled-device-" + std::to_string(device_id);
}

std::string cuda_hw_profile(int32_t) {
    return "cuda";
}
#endif

std::unique_ptr<Engine> create_engine(const EngineConfig& config) {
    const bool speculative_enabled = config.speculative().value("enabled", false);
    if (speculative_enabled) {
        throw std::runtime_error("Speculative decoding (EagleEngine) not yet supported in EdgeFM facade");
    }

    const BackendTarget backend_target = config.backend_target_kind();
    if (config.tuning_enabled() && backend_target != BackendTarget::Cuda) {
        throw ConfigurationError(
            "Config-driven tuning.enabled currently supports CUDA only. "
            "Horizon continues to use explicit engine.tune().");
    }

    switch (backend_target) {
        case BackendTarget::Cuda:
            return create_cuda_engine(config);
        case BackendTarget::Horizon:
            return create_horizon_engine(config);
    }
    throw ConfigurationError("Unsupported backend_target: " + config.backend_target());
}

} // namespace edge_fm
