#include "utils/device/weight_loader.h"
#include "utils/logging.h"
#include <fstream>
#include <filesystem>
#include <algorithm>

#define SAFETENSORS_CPP_IMPLEMENTATION
#include <safetensors.hh>

namespace edge_fm {

WeightLoader& WeightLoader::instance() {
    static WeightLoader loader;
    return loader;
}

void WeightLoader::clear_stage(ModelStage cache_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.erase(cache_key);
    stage_stfiles_.erase(cache_key);
}

const std::unordered_map<std::string, Tensor>& WeightLoader::get(ModelStage cache_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
        return it->second;
    }
    throw ConfigurationError("Weights not found for cache_key");
}

DType safetensors_dtype_to_edge_fm_dtype(safetensors::dtype dtype) {
    switch (dtype) {
        case safetensors::dtype::kFLOAT32:
            return DType::Float32;
        case safetensors::dtype::kFLOAT16:
            return DType::Float16;
        case safetensors::dtype::kBFLOAT16:
            return DType::BFloat16;
        case safetensors::dtype::kINT32:
            return DType::Int32;
        case safetensors::dtype::kINT64:
            return DType::Int64;
        case safetensors::dtype::kUINT8:
            return DType::UInt8;
        case safetensors::dtype::kINT8:
            return DType::Int8;
        default:
            throw ConfigurationError("Unsupported safetensors dtype: " + 
                                    std::to_string(static_cast<int>(dtype)));
    }
}

void WeightLoader::load_weights_from_file(
    ModelStage cache_key,
    const std::string& safetensors_file,
    Device device,
    int32_t device_id,
    bool overwrite_if_exists,
    const std::optional<std::function<bool(const std::string&)>>& weight_filter,
    const std::optional<std::function<std::string(const std::string&)>>& key_mapper)
{
    // 如果safetensors_file已经加载过，则直接返回
    auto it = stage_stfiles_.find(cache_key);
    if (it != stage_stfiles_.end() && 
        std::find(it->second.begin(), it->second.end(), safetensors_file) != it->second.end()) {
        Logging::instance().log_info("Weights already loaded from file: {}.", safetensors_file);
        return;
    }
    
    safetensors::safetensors_t st;
    std::string warn, err;
    bool ret = safetensors::load_from_file(safetensors_file, &st, &warn, &err);
    
    if (!ret) {
        throw ConfigurationError("Failed to load safetensors file: " + safetensors_file + " - " + err);
    }
    if (!safetensors::validate_data_offsets(st, err)) {
        throw ConfigurationError("Invalid data_offsets in safetensors file: " + safetensors_file + " - " + err);
    }
    const uint8_t* databuffer = nullptr;
    if (st.mmaped) {
        databuffer = st.databuffer_addr;
    } else {
        databuffer = st.storage.data();
    }
    
    // 临时存储加载的权重
    std::unordered_map<std::string, Tensor> loaded_weights;
    for (size_t i = 0; i < st.tensors.size(); i++) {
        std::string tensor_name = st.tensors.keys()[i];
        
        // filter weights by weight_filter
        if (weight_filter.has_value() && !weight_filter.value()(tensor_name)) {
            continue;
        }
        std::string cache_name = key_mapper.has_value() ? key_mapper.value()(tensor_name) : tensor_name;

        safetensors::tensor_t tensor_info;
        if (!st.tensors.at(i, &tensor_info)) {
            throw ConfigurationError("Failed to get tensor at index " + std::to_string(i));
        }
        std::vector<int64_t> shape;
        for (size_t j = 0; j < tensor_info.shape.size(); j++) {
            shape.push_back(static_cast<int64_t>(tensor_info.shape[j]));
        }
        DType dtype = safetensors_dtype_to_edge_fm_dtype(tensor_info.dtype);
        const void* tensor_data = databuffer + tensor_info.data_offsets[0];
        
        Tensor tensor;
        if (device == Device::GPU) {
            // Clone directly from CPU to GPU (explicit src/dst devices)
            tensor = Tensor::clone_from(tensor_data, shape, dtype,
                                        Device::CPU, 0,
                                        Device::GPU, device_id,
                                        MemoryOwnership::OwnCudaMalloc, nullptr);
        } else {
            tensor = Tensor::clone_from(tensor_data, shape, dtype,
                                        Device::CPU, 0,
                                        Device::CPU, 0,
                                        MemoryOwnership::OwnCpuMalloc, nullptr);
        }
        loaded_weights[cache_name] = std::move(tensor);
    }
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& cached_weights = cache_[cache_key];
        for (auto& [name, tensor] : loaded_weights) {
            if (cached_weights.find(name) != cached_weights.end()) {
                Logging::instance().log_info(
                    "Duplicate tensor: {} found in cache_key: {}. Overwriting if overwrite_if_exists={}.", 
                    name, 
                    model_stage_to_string(cache_key), 
                    overwrite_if_exists);
                if (overwrite_if_exists) {
                    cached_weights[name] = std::move(tensor);
                }
            } else {
                cached_weights[name] = std::move(tensor);
            }
        }
        stage_stfiles_[cache_key].push_back(safetensors_file);
    }
}

} // namespace edge_fm