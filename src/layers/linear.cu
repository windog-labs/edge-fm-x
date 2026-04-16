#include "layers/linear.h"
#include "operators/fused_gate_up_activation_op.h"
#include "operators/operator_impl_table.h"
#include "utils/device/nvtx.h"
#include <vector>
#include <nlohmann/json.hpp>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <regex>
#include "utils/device/cuda_utils.h"
#include "utils/check.h"
#include "utils/device/memory.h"
#include "utils/device/weight_loader.h"
#include "operators/kernels/int4_groupwise_gemm/int4GroupwiseGemm.h"

#include <cstdlib>
#include <algorithm>
#include <array>
#include <limits>
#include <set>

using namespace trt_edgellm::kernel;

namespace edge_fm {

std::string LinearLayer::LinearShapeSignature::to_string() const {
    return "m=" + std::to_string(m) +
        "|input=" + std::to_string(static_cast<int>(input_dtype)) +
        "|weight=" + std::to_string(static_cast<int>(weight_dtype)) +
        "|output=" + std::to_string(static_cast<int>(output_dtype)) +
        "|in_features=" + std::to_string(in_features) +
        "|out_features=" + std::to_string(out_features);
}

namespace {

std::string infer_layer_role(const std::string& layer_prefix) {
    if (layer_prefix == "lm_head" || layer_prefix.find("lm_head") != std::string::npos) {
        return "lm_head";
    }
    if (layer_prefix.find("qkv_fused") != std::string::npos) {
        return "fused_qkv";
    }
    if (layer_prefix.find("gate_up_fused") != std::string::npos) {
        return "fused_gate_up";
    }
    if (layer_prefix.find(".o_proj") != std::string::npos) {
        return "attention_output";
    }
    if (layer_prefix.find(".down_proj") != std::string::npos) {
        return "mlp_down";
    }
    return "linear";
}

std::string stage_key(ModelStage stage) {
    return stage == ModelStage::Decode ? "decode" : "prefill";
}

std::string model_name_for_operator_resolution(const EngineConfig& engine_config) {
    try {
        return engine_config.resolved_model_name();
    } catch (const ConfigurationError&) {
        return std::string();
    }
}

int cublaslt_max_algo_candidates() {
    static int value = []() {
        constexpr int kDefault = 16;
        const char* raw = std::getenv("EDGE_FM_CUBLASLT_MAX_ALGO_CANDIDATES");
        if (raw == nullptr || *raw == '\0') {
            return kDefault;
        }
        char* end = nullptr;
        long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
            return kDefault;
        }
        return static_cast<int>(parsed);
    }();
    return value;
}

size_t cublaslt_max_workspace_bytes() {
    static size_t value = []() {
        constexpr size_t kDefaultBytes = 64ULL * 1024ULL * 1024ULL;
        const char* raw = std::getenv("EDGE_FM_CUBLASLT_MAX_WORKSPACE_MB");
        if (raw == nullptr || *raw == '\0') {
            return kDefaultBytes;
        }
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed == 0) {
            return kDefaultBytes;
        }
        return static_cast<size_t>(parsed) * static_cast<size_t>(1024ULL) * static_cast<size_t>(1024ULL);
    }();
    return value;
}

template <typename T>
bool get_cublaslt_algo_config_attr(
    const cublasLtMatmulAlgo_t& algo,
    cublasLtMatmulAlgoConfigAttributes_t attr,
    T* value)
{
    size_t written = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(
        &algo, attr, value, sizeof(T), &written);
    return status == CUBLAS_STATUS_SUCCESS && written == sizeof(T);
}

template <typename T>
bool set_cublaslt_algo_config_attr(
    cublasLtMatmulAlgo_t* algo,
    cublasLtMatmulAlgoConfigAttributes_t attr,
    T value)
{
    return cublasLtMatmulAlgoConfigSetAttribute(algo, attr, &value, sizeof(T)) == CUBLAS_STATUS_SUCCESS;
}

template <typename T>
bool get_cublaslt_algo_cap_attr(
    const cublasLtMatmulAlgo_t& algo,
    cublasLtMatmulAlgoCapAttributes_t attr,
    T* value)
{
    size_t written = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoCapGetAttribute(
        &algo, attr, value, sizeof(T), &written);
    return status == CUBLAS_STATUS_SUCCESS && written == sizeof(T);
}

template <typename T>
std::vector<T> get_cublaslt_algo_cap_vector(
    const cublasLtMatmulAlgo_t& algo,
    cublasLtMatmulAlgoCapAttributes_t attr)
{
    size_t bytes = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&algo, attr, nullptr, 0, &bytes) != CUBLAS_STATUS_SUCCESS || bytes == 0) {
        return {};
    }
    std::vector<T> values(bytes / sizeof(T));
    size_t written = 0;
    if (values.empty() || cublasLtMatmulAlgoCapGetAttribute(&algo, attr, values.data(), bytes, &written) != CUBLAS_STATUS_SUCCESS) {
        return {};
    }
    values.resize(written / sizeof(T));
    return values;
}

void maybe_add_algo_config_attr(
    nlohmann::json& out,
    const cublasLtMatmulAlgo_t& algo,
    cublasLtMatmulAlgoConfigAttributes_t attr,
    const char* key)
{
    switch (attr) {
        case CUBLASLT_ALGO_CONFIG_ID:
        case CUBLASLT_ALGO_CONFIG_SPLITK_NUM: {
            int32_t value = 0;
            if (get_cublaslt_algo_config_attr(algo, attr, &value)) {
                out[key] = value;
            }
            return;
        }
        case CUBLASLT_ALGO_CONFIG_TILE_ID:
        case CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME:
        case CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING:
        case CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION:
        case CUBLASLT_ALGO_CONFIG_STAGES_ID: {
            uint32_t value = 0;
            if (get_cublaslt_algo_config_attr(algo, attr, &value)) {
                out[key] = value;
            }
            return;
        }
        case CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID:
        case CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID: {
            uint16_t value = 0;
            if (get_cublaslt_algo_config_attr(algo, attr, &value)) {
                out[key] = value;
            }
            return;
        }
        default:
            return;
    }
}

nlohmann::json cublaslt_algo_to_json(const cublasLtMatmulAlgo_t& algo) {
    nlohmann::json out = nlohmann::json::object();
    maybe_add_algo_config_attr(out, algo, CUBLASLT_ALGO_CONFIG_ID, "algo_id");
    maybe_add_algo_config_attr(out, algo, CUBLASLT_ALGO_CONFIG_TILE_ID, "tile_id");
    maybe_add_algo_config_attr(out, algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, "splitk_num");
    maybe_add_algo_config_attr(out, algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, "reduction_scheme");
    maybe_add_algo_config_attr(out, algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, "cta_swizzling");
    maybe_add_algo_config_attr(out, algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, "custom_option");
    maybe_add_algo_config_attr(out, algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, "stages_id");
    maybe_add_algo_config_attr(out, algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, "inner_shape_id");
    maybe_add_algo_config_attr(out, algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, "cluster_shape_id");
    return out;
}

int64_t json_i64_or(const nlohmann::json& value, const char* key, int64_t default_value) {
    if (!value.is_object()) {
        return default_value;
    }
    const auto it = value.find(key);
    if (it == value.end() || it->is_null()) {
        return default_value;
    }
    return it->get<int64_t>();
}

bool json_has_explicit_cublaslt_key(const nlohmann::json& value) {
    if (!value.is_object()) {
        return false;
    }
    static const std::array<const char*, 9> kKeys = {{
        "algo_id",
        "tile_id",
        "splitk_num",
        "reduction_scheme",
        "cta_swizzling",
        "custom_option",
        "stages_id",
        "inner_shape_id",
        "cluster_shape_id",
    }};
    for (const char* key : kKeys) {
        if (value.contains(key)) {
            return true;
        }
    }
    return false;
}

} // namespace

LinearLayer::LinearLayer(const std::string& layer_prefix,
                         const EngineConfig& engine_config,
                         uint32_t in_features,
                         uint32_t out_features,
                         std::string layer_name) : 
    Layer(engine_config, std::move(layer_name)),
    cublaslt_handle_(nullptr),
    in_features_(in_features),
    out_features_(out_features),
    layer_prefix_(layer_prefix),
    layer_role_(infer_layer_role(layer_prefix))
{
    // Check if we need CUBLASLt handle based on model dtype
    auto check_dtype_needs_cublaslt = [](const nlohmann::json& config) -> bool {
        if (config.contains("torch_dtype")) {
            std::string dtype_str = config["torch_dtype"].get<std::string>();
            return dtype_str == "float16" || dtype_str == "fp16" || 
                   dtype_str == "bfloat16" || dtype_str == "bf16";
        }
        return false;
    };
    
    bool need_cublaslt = check_dtype_needs_cublaslt(engine_config_.prefill_model_config()) ||
                         check_dtype_needs_cublaslt(engine_config_.decode_model_config());
    
    // Create CUBLASLt handle if needed
    if (need_cublaslt) {
        // Set CUDA device before creating cublasLt handle
        CUDA_CHECK_THROW(cudaSetDevice(device_id_), "Failed to set CUDA device");
        cublasStatus_t status = cublasLtCreate(&cublaslt_handle_);
        check<DeviceError>(status == CUBLAS_STATUS_SUCCESS,
                          "Failed to create CUBLASLt handle: " + 
                          std::to_string(static_cast<int>(status)));
    }

}

LinearLayer::~LinearLayer()
{
    for (auto& [m, cached] : prefill_descriptors_map_) {
        (void)m;
        cleanup_cached_descriptors(cached);
    }
    prefill_descriptors_map_.clear();
    cleanup_cached_descriptors(decode_descriptors_);

    if (cublaslt_handle_ != nullptr) {
        cublasLtDestroy(cublaslt_handle_);
        cublaslt_handle_ = nullptr;
    }
}

namespace {

// Find weight name by prefix matching (e.g., "test" matches "test.weight", "test.qweight")
std::string find_weight_name_by_prefix(
    const std::string& layer_prefix,
    const std::unordered_map<std::string, Tensor>& weights)
{
    auto starts_with = [](const std::string& name, const std::string& prefix) {
        return name.size() >= prefix.size() && name.compare(0, prefix.size(), prefix) == 0;
    };
    
    // First try to find .weight suffix
    for (const auto& [name, tensor] : weights) {
        if (starts_with(name, layer_prefix) && name.substr(layer_prefix.size()) == ".weight") {
            return name;
        }
    }

    // Then try to find .qweight suffix
    for (const auto& [name, tensor] : weights) {
        if (starts_with(name, layer_prefix) && name.substr(layer_prefix.size()) == ".qweight") {
            return name;
        }
    }
    
    // Finally try to find any weight name starting with layer_prefix
    for (const auto& [name, tensor] : weights) {
        if (starts_with(name, layer_prefix) && name.find("weight") != std::string::npos) {
            return name;
        }
    }
    
    throw ConfigurationError("LinearLayer: No weight found matching layer prefix: " + layer_prefix);
}

}  // anonymous namespace

void LinearLayer::load_weight_set(
    const std::unordered_map<std::string, Tensor>& weights,
    const std::string& weight_name_base,
    WeightSet& weight_set)
{
    WeightSet loaded_weight_set;
    
    std::string weight_name = weight_name_base + ".weight";
    auto weight_it = weights.find(weight_name);
    
    // If not found, try alternative naming (e.g., for INT4: weight_name_base + ".qweight")
    if (weight_it == weights.end()) {
        weight_name = weight_name_base + ".qweight";
        weight_it = weights.find(weight_name);
    }
    
    check<ConfigurationError>(weight_it != weights.end(),
                              "LinearLayer: missing weight '" + weight_name_base + ".weight' or '" + 
                              weight_name_base + ".qweight' in weights");
    
    const Tensor& weight_tensor = weight_it->second;
    const auto& weight_shape = weight_tensor.shape();
    DType weight_dtype = weight_tensor.dtype();
    
    // Check if this is INT4 quantized (has qweight and scaling_factors)
    std::string scaling_name = weight_name_base + ".scaling_factors";
    auto scaling_it = weights.find(scaling_name);
    
    if (scaling_it != weights.end() && weight_dtype == DType::Int8) { // INT4 group-wise quantized
        loaded_weight_set.quant_type_ = QuantType::INT4_GROUPWISE;
        loaded_weight_set.weight_ = &weight_tensor;
        loaded_weight_set.scaling_factors_ = &scaling_it->second;
        
        check<ConfigurationError>(weight_shape.size() == 2,
                                  "LinearLayer: INT4 weight must be 2D [out_features/2, in_features]. "
                                  "Got " + std::to_string(weight_shape.size()) + "D");
        const auto& scaling_shape = scaling_it->second.shape();
        check<ConfigurationError>(scaling_shape.size() == 2,
                                  "LinearLayer: scaling_factors must be 2D [in_features/group_size, out_features]. "
                                  "Got " + std::to_string(scaling_shape.size()) + "D");
        
        // Verify dimensions match constructor parameters
        uint32_t inferred_out_features = static_cast<uint32_t>(scaling_shape[1]);
        uint32_t inferred_in_features = static_cast<uint32_t>(weight_shape[1]);
        
        check<ConfigurationError>(in_features_ == inferred_in_features && out_features_ == inferred_out_features,
                                  "LinearLayer: dimension mismatch. Expected [out_features, in_features] = [" +
                                  std::to_string(out_features_) + ", " + std::to_string(in_features_) + "], "
                                  "got [" + std::to_string(inferred_out_features) + ", " + std::to_string(inferred_in_features) + "]");
        
        uint32_t expected_scaling_dim0 = (in_features_ + loaded_weight_set.group_size_ - 1) / loaded_weight_set.group_size_;
        if (scaling_shape[0] != static_cast<int64_t>(expected_scaling_dim0)) {
            loaded_weight_set.group_size_ = static_cast<uint32_t>(in_features_ / scaling_shape[0]);
        }
        check<ConfigurationError>(weight_shape[0] == static_cast<int64_t>(out_features_ / 2),
                                  "LinearLayer: INT4 weight shape mismatch. Expected [out_features/2, in_features] = [" +
                                  std::to_string(out_features_ / 2) + ", " + std::to_string(in_features_) + "], "
                                  "got [" + std::to_string(weight_shape[0]) + ", " + std::to_string(weight_shape[1]) + "]");
    } else { // FP16/BF16 quantized
        loaded_weight_set.quant_type_ = QuantType::FP16_BF16;
        loaded_weight_set.weight_ = &weight_tensor;
        
        check<ConfigurationError>(weight_shape.size() == 2,
                                  "LinearLayer: weight must be 2D [out_features, in_features]. "
                                  "Got " + std::to_string(weight_shape.size()) + "D");
        check<ConfigurationError>(weight_dtype == DType::Float16 || weight_dtype == DType::BFloat16,
                                  "LinearLayer: weight dtype must be Float16 or BFloat16 for FP16/BF16. "
                                  "Got dtype: " + std::to_string(static_cast<int>(weight_dtype)));
        
        // Verify dimensions match constructor parameters
        uint32_t inferred_out_features = static_cast<uint32_t>(weight_shape[0]);
        uint32_t inferred_in_features = static_cast<uint32_t>(weight_shape[1]);
        
        check<ConfigurationError>(in_features_ == inferred_in_features && out_features_ == inferred_out_features,
                                  "LinearLayer: dimension mismatch. Expected [out_features, in_features] = [" +
                                  std::to_string(out_features_) + ", " + std::to_string(in_features_) + "], "
                                  "got [" + std::to_string(inferred_out_features) + ", " + std::to_string(inferred_in_features) + "]");
        
        std::string bias_name = weight_name_base + ".bias";
        auto bias_it = weights.find(bias_name);
        if (bias_it != weights.end()) {
            loaded_weight_set.bias_ = &bias_it->second;
            const auto& bias_shape = bias_it->second.shape();
            check<ConfigurationError>(bias_shape.size() == 1 && bias_shape[0] == static_cast<int64_t>(out_features_),
                                      "LinearLayer: bias shape mismatch. Expected [" + 
                                      std::to_string(out_features_) + "], got [" + 
                                      std::to_string(bias_shape[0]) + "]");
        }
    }
    
    std::swap(weight_set, loaded_weight_set);
}

void LinearLayer::load_weights(
    const std::unordered_map<std::string, Tensor>& prefill_weights,
    const std::unordered_map<std::string, Tensor>& decode_weights)
{
    // Reset weight sets to ensure clean state
    prefill_weights_ = WeightSet();
    decode_weights_ = WeightSet();
    
    // Verify that weights matching layer_prefix_ exist
    find_weight_name_by_prefix(layer_prefix_, prefill_weights);
    
    // Use layer_prefix_ directly as weight_name_base (it's already the base name without suffix)
    load_weight_set(prefill_weights, layer_prefix_, prefill_weights_);
    
    if (!decode_weights.empty()) {
        find_weight_name_by_prefix(layer_prefix_, decode_weights);
        load_weight_set(decode_weights, layer_prefix_, decode_weights_);
    } else {
        decode_weights_ = prefill_weights_;
    }
    
    weights_loaded_ = true;
}

void LinearLayer::cleanup_cached_descriptors(CachedDescriptors& cached)
{
    if (cached.matmul_desc_ != nullptr) {
        cublasLtMatmulDescDestroy(cached.matmul_desc_);
        cached.matmul_desc_ = nullptr;
    }
    
    auto destroy_layout = [](cublasLtMatrixLayout_t& desc) {
        if (desc != nullptr) {
            cublasLtMatrixLayoutDestroy(desc);
            desc = nullptr;
        }
    };
    
    destroy_layout(cached.Bdesc_);
    destroy_layout(cached.Adesc_);
    destroy_layout(cached.Cdesc_);
    destroy_layout(cached.Ddesc_);
    cached.cached_m_ = -1;
    cached.has_algo_ = false;
    cached.heuristic_candidate_count_ = 0;
    cached.best_algo_index_ = -1;
    cached.heuristic_candidates_.clear();
    cached.selected_impl_id_.clear();
    cached.selected_impl_params_ = nlohmann::json::object();
}

void LinearLayer::get_or_create_descriptors(
    int m,
    cudaDataType_t input_type,
    cudaDataType_t weight_type,
    cudaDataType_t output_type,
    const void* bias_ptr,
    CachedDescriptors& cached,
    cublasLtMatmulDesc_t& matmul_desc,
    cublasLtMatrixLayout_t& Adesc,
    cublasLtMatrixLayout_t& Bdesc,
    cublasLtMatrixLayout_t& Cdesc,
    cublasLtMatrixLayout_t& Ddesc)
{
    int n = static_cast<int>(out_features_);
    int k = static_cast<int>(in_features_);
    bool has_bias = (bias_ptr != nullptr);
    
    bool need_recreate_base = cached.matmul_desc_ == nullptr ||
                               cached.Bdesc_ == nullptr ||
                               cached.cached_weight_type_ != weight_type ||
                               cached.has_bias_ != has_bias;
    
    if (need_recreate_base) {
        if (cached.matmul_desc_ != nullptr) {
            cublasLtMatmulDescDestroy(cached.matmul_desc_);
        }
        if (cached.Bdesc_ != nullptr) {
            cublasLtMatrixLayoutDestroy(cached.Bdesc_);
        }
        // CUBLAS_COMPUTE_32F allows TF32 on Ampere+, matching PyTorch's default (allow_tf32=True)
        cublasStatus_t status = cublasLtMatmulDescCreate(&cached.matmul_desc_, 
                                                         CUBLAS_COMPUTE_32F, 
                                                         CUDA_R_32F);
        check<DeviceError>(status == CUBLAS_STATUS_SUCCESS, 
                           "LinearLayer: Failed to create matmul descriptor");
        
        // Set transposition
        // For row-major C = A @ B^T, in cuBLASLt column-major view: C^T = B @ A^T
        // So we need to swap A and B, and set transA = T, transB = N
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        status = cublasLtMatmulDescSetAttribute(cached.matmul_desc_, 
                                                CUBLASLT_MATMUL_DESC_TRANSA,
                                                &transa, sizeof(transa));
        check<DeviceError>(status == CUBLAS_STATUS_SUCCESS,
                          "LinearLayer: Failed to set TRANSA");
        
        status = cublasLtMatmulDescSetAttribute(cached.matmul_desc_, 
                                                CUBLASLT_MATMUL_DESC_TRANSB,
                                                &transb, sizeof(transb));
        check<DeviceError>(status == CUBLAS_STATUS_SUCCESS,
                          "LinearLayer: Failed to set TRANSB");
        
        // Set epilogue
        cublasLtEpilogue_t epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
        status = cublasLtMatmulDescSetAttribute(cached.matmul_desc_, 
                                                CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue));
        check<DeviceError>(status == CUBLAS_STATUS_SUCCESS,
                          "LinearLayer: Failed to set epilogue");
        
        // Create Bdesc (weight layout, independent of batch_size)
        status = cublasLtMatrixLayoutCreate(&cached.Bdesc_, weight_type, k, n, k);
        check<DeviceError>(status == CUBLAS_STATUS_SUCCESS,
                          "LinearLayer: Failed to create B layout");
        
        cached.cached_weight_type_ = weight_type;
        cached.has_bias_ = has_bias;
    }
    
    // Update bias pointer (may change even if has_bias is same)
    if (has_bias) {
        cublasStatus_t status = cublasLtMatmulDescSetAttribute(cached.matmul_desc_, 
                                                                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                                &bias_ptr, sizeof(bias_ptr));
        check<DeviceError>(status == CUBLAS_STATUS_SUCCESS, "LinearLayer: Failed to set bias pointer");
    }
    
    // Check if we need to recreate Adesc, Cdesc, Ddesc (depend on batch_size and input/output types)
    bool need_recreate_layouts = cached.Adesc_ == nullptr ||
                                  cached.Cdesc_ == nullptr ||
                                  cached.Ddesc_ == nullptr ||
                                  cached.cached_m_ != m ||
                                  cached.cached_input_type_ != input_type ||
                                  cached.cached_output_type_ != output_type;
    
    if (need_recreate_layouts) {
        // Cleanup old layouts
        auto destroy_layout = [](cublasLtMatrixLayout_t& desc) {
            if (desc != nullptr) {
                cublasLtMatrixLayoutDestroy(desc);
                desc = nullptr;
            }
        };
        
        destroy_layout(cached.Adesc_);
        destroy_layout(cached.Cdesc_);
        destroy_layout(cached.Ddesc_);
        
        // Create new layouts
        // For row-major matrices, cuBLASLt uses column-major view: swap rows and cols
        // A: [m, k] row-major -> (rows=K, cols=M, ld=K)
        cublasStatus_t status = cublasLtMatrixLayoutCreate(&cached.Adesc_, input_type, k, m, k);
        check<DeviceError>(status == CUBLAS_STATUS_SUCCESS,
                          "LinearLayer: Failed to create A layout");
        // C/D: [m, n] row-major -> (rows=N, cols=M, ld=N)
        status = cublasLtMatrixLayoutCreate(&cached.Cdesc_, output_type, n, m, n);
        check<DeviceError>(status == CUBLAS_STATUS_SUCCESS,
                          "LinearLayer: Failed to create C layout");
        
        status = cublasLtMatrixLayoutCreate(&cached.Ddesc_, output_type, n, m, n);
        check<DeviceError>(status == CUBLAS_STATUS_SUCCESS,
                          "LinearLayer: Failed to create D layout");
        
        cached.cached_m_ = m;
        cached.cached_input_type_ = input_type;
        cached.cached_output_type_ = output_type;
        cached.heuristic_candidates_.clear();
        cached.heuristic_candidate_count_ = 0;
        cached.best_algo_index_ = -1;

        // Query top-K algorithms via heuristic; operator_impl_table may optionally pin algo_index.
        const int max_algo_candidates = cublaslt_max_algo_candidates();
        std::vector<cublasLtMatmulHeuristicResult_t> results(static_cast<size_t>(max_algo_candidates));
        cublasLtMatmulPreference_t pref = nullptr;
        status = cublasLtMatmulPreferenceCreate(&pref);
        if (status == CUBLAS_STATUS_SUCCESS) {
            const size_t max_workspace_bytes = cublaslt_max_workspace_bytes();
            status = cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &max_workspace_bytes, sizeof(max_workspace_bytes));
            int returned = 0;
            if (status == CUBLAS_STATUS_SUCCESS) {
                status = cublasLtMatmulAlgoGetHeuristic(
                    cublaslt_handle_, cached.matmul_desc_,
                    cached.Bdesc_, cached.Adesc_, cached.Cdesc_, cached.Ddesc_,
                    pref, max_algo_candidates, results.data(), &returned);
            }
            cublasLtMatmulPreferenceDestroy(pref);
            if (status == CUBLAS_STATUS_SUCCESS && returned > 0) {
                cached.heuristic_candidates_.assign(results.begin(), results.begin() + returned);
                cached.heuristic_candidate_count_ = returned;
                cached.heuristic_ = cached.heuristic_candidates_[0];
                cached.has_algo_ = true;
                cached.best_algo_index_ = (returned == 1) ? 0 : -1;  // single candidate: use directly
            }
        }
    }
    
    // Return cached descriptors
    matmul_desc = cached.matmul_desc_;
    Adesc = cached.Adesc_;
    Bdesc = cached.Bdesc_;
    Cdesc = cached.Cdesc_;
    Ddesc = cached.Ddesc_;
}

bool LinearLayer::has_explicit_cublaslt_algo_config(const nlohmann::json& impl_params) const
{
    return json_has_explicit_cublaslt_key(impl_params);
}

nlohmann::json LinearLayer::describe_cublaslt_algo_config(const cublasLtMatmulAlgo_t& algo) const
{
    return cublaslt_algo_to_json(algo);
}

bool LinearLayer::get_best_cublaslt_heuristic_for_algo_id(
    const CachedDescriptors& cached,
    int32_t algo_id,
    cublasLtMatmulHeuristicResult_t* result,
    std::string* error_message) const
{
    check<InvalidRequestError>(result != nullptr, "LinearLayer: result must not be null");

    if (cublaslt_handle_ == nullptr || cached.matmul_desc_ == nullptr || cached.Adesc_ == nullptr ||
        cached.Bdesc_ == nullptr || cached.Cdesc_ == nullptr || cached.Ddesc_ == nullptr)
    {
        if (error_message != nullptr) {
            *error_message = "cublasLt limited-by-algo-id heuristic requires initialized descriptors";
        }
        return false;
    }

    if (algo_id < 0) {
        if (error_message != nullptr) {
            *error_message = "algo_id must be non-negative";
        }
        return false;
    }

    *result = {};
    cublasStatus_t status = cublasLtMatmulAlgoInit(
        cublaslt_handle_,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F,
        cached.cached_weight_type_,
        cached.cached_input_type_,
        cached.cached_output_type_,
        cached.cached_output_type_,
        algo_id,
        &result->algo);
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (error_message != nullptr) {
            *error_message =
                "cublasLtMatmulAlgoInit failed for algo_id=" + std::to_string(algo_id) +
                " status=" + std::to_string(static_cast<int>(status));
        }
        return false;
    }

    cublasLtMatmulPreference_t pref = nullptr;
    status = cublasLtMatmulPreferenceCreate(&pref);
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (error_message != nullptr) {
            *error_message =
                "cublasLtMatmulPreferenceCreate failed with status=" + std::to_string(static_cast<int>(status));
        }
        return false;
    }

    const auto cleanup_pref = [&]() {
        if (pref != nullptr) {
            (void)cublasLtMatmulPreferenceDestroy(pref);
            pref = nullptr;
        }
    };

    const uint32_t search_mode = static_cast<uint32_t>(CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID);
    status = cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_SEARCH_MODE,
        &search_mode,
        sizeof(search_mode));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup_pref();
        if (error_message != nullptr) {
            *error_message =
                "failed to set CUBLASLT_MATMUL_PREF_SEARCH_MODE, status=" +
                std::to_string(static_cast<int>(status));
        }
        return false;
    }

    const uint64_t max_workspace_bytes = static_cast<uint64_t>(cublaslt_max_workspace_bytes());
    status = cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace_bytes,
        sizeof(max_workspace_bytes));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup_pref();
        if (error_message != nullptr) {
            *error_message =
                "failed to set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, status=" +
                std::to_string(static_cast<int>(status));
        }
        return false;
    }

    int returned = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        cublaslt_handle_,
        cached.matmul_desc_,
        cached.Bdesc_,
        cached.Adesc_,
        cached.Cdesc_,
        cached.Ddesc_,
        pref,
        1,
        result,
        &returned);
    cleanup_pref();

    if (status != CUBLAS_STATUS_SUCCESS || returned <= 0 || result->state != CUBLAS_STATUS_SUCCESS) {
        if (error_message != nullptr) {
            *error_message =
                "cublasLtMatmulAlgoGetHeuristic limited-by-algo-id failed for algo_id=" +
                std::to_string(algo_id) +
                " status=" + std::to_string(static_cast<int>(status)) +
                " returned=" + std::to_string(returned) +
                " result_state=" + std::to_string(static_cast<int>(result->state));
        }
        return false;
    }

    if (result->workspaceSize > static_cast<size_t>(max_workspace_bytes)) {
        if (error_message != nullptr) {
            *error_message =
                "limited-by-algo-id heuristic returned workspace=" + std::to_string(result->workspaceSize) +
                " bytes, exceeds limit=" + std::to_string(max_workspace_bytes);
        }
        return false;
    }

    return true;
}

bool LinearLayer::try_select_explicit_cublaslt_algo(
    const LinearOpContext& ctx,
    CachedDescriptors& cached,
    std::string* error_message)
{
    if (!has_explicit_cublaslt_algo_config(cached.selected_impl_params_)) {
        return false;
    }
    if (cublaslt_handle_ == nullptr || cached.matmul_desc_ == nullptr || cached.Adesc_ == nullptr ||
        cached.Bdesc_ == nullptr || cached.Cdesc_ == nullptr || cached.Ddesc_ == nullptr)
    {
        if (error_message != nullptr) {
            *error_message = "cublasLt explicit algo selection requires initialized descriptors";
        }
        return false;
    }

    const int64_t algo_id = json_i64_or(cached.selected_impl_params_, "algo_id", -1);
    if (algo_id < 0 || algo_id > std::numeric_limits<int32_t>::max()) {
        if (error_message != nullptr) {
            *error_message =
                "cublasLt explicit algo selection requires a non-negative algo_id, got impl_params=" +
                cached.selected_impl_params_.dump();
        }
        return false;
    }

    cublasLtMatmulAlgo_t algo = {};
    cublasStatus_t status = cublasLtMatmulAlgoInit(
        cublaslt_handle_,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F,
        cached.cached_weight_type_,
        cached.cached_input_type_,
        cached.cached_output_type_,
        cached.cached_output_type_,
        static_cast<int32_t>(algo_id),
        &algo);
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (error_message != nullptr) {
            *error_message =
                "cublasLtMatmulAlgoInit failed for explicit algo_id=" + std::to_string(algo_id) +
                " status=" + std::to_string(static_cast<int>(status));
        }
        return false;
    }

    const auto try_set_attr_i32 = [&](const char* key, cublasLtMatmulAlgoConfigAttributes_t attr) {
        const auto it = cached.selected_impl_params_.find(key);
        if (it == cached.selected_impl_params_.end()) {
            return true;
        }
        const int64_t raw = it->get<int64_t>();
        if (raw < std::numeric_limits<int32_t>::min() || raw > std::numeric_limits<int32_t>::max()) {
            if (error_message != nullptr) {
                *error_message = std::string("cublasLt explicit attr out of range for ") + key;
            }
            return false;
        }
        return set_cublaslt_algo_config_attr(&algo, attr, static_cast<int32_t>(raw));
    };
    const auto try_set_attr_u32 = [&](const char* key, cublasLtMatmulAlgoConfigAttributes_t attr) {
        const auto it = cached.selected_impl_params_.find(key);
        if (it == cached.selected_impl_params_.end()) {
            return true;
        }
        const uint64_t raw = it->get<uint64_t>();
        if (raw > std::numeric_limits<uint32_t>::max()) {
            if (error_message != nullptr) {
                *error_message = std::string("cublasLt explicit attr out of range for ") + key;
            }
            return false;
        }
        return set_cublaslt_algo_config_attr(&algo, attr, static_cast<uint32_t>(raw));
    };
    const auto try_set_attr_u16 = [&](const char* key, cublasLtMatmulAlgoConfigAttributes_t attr) {
        const auto it = cached.selected_impl_params_.find(key);
        if (it == cached.selected_impl_params_.end()) {
            return true;
        }
        const uint64_t raw = it->get<uint64_t>();
        if (raw > std::numeric_limits<uint16_t>::max()) {
            if (error_message != nullptr) {
                *error_message = std::string("cublasLt explicit attr out of range for ") + key;
            }
            return false;
        }
        return set_cublaslt_algo_config_attr(&algo, attr, static_cast<uint16_t>(raw));
    };

    if (!try_set_attr_u32("tile_id", CUBLASLT_ALGO_CONFIG_TILE_ID) ||
        !try_set_attr_i32("splitk_num", CUBLASLT_ALGO_CONFIG_SPLITK_NUM) ||
        !try_set_attr_u32("reduction_scheme", CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) ||
        !try_set_attr_u32("cta_swizzling", CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING) ||
        !try_set_attr_u32("custom_option", CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION) ||
        !try_set_attr_u32("stages_id", CUBLASLT_ALGO_CONFIG_STAGES_ID) ||
        !try_set_attr_u16("inner_shape_id", CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID) ||
        !try_set_attr_u16("cluster_shape_id", CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID))
    {
        if (error_message != nullptr && error_message->empty()) {
            *error_message =
                "cublasLtMatmulAlgoConfigSetAttribute failed for impl_params=" + cached.selected_impl_params_.dump();
        }
        return false;
    }

    cublasLtMatmulHeuristicResult_t explicit_result = {};
    status = cublasLtMatmulAlgoCheck(
        cublaslt_handle_,
        cached.matmul_desc_,
        cached.Bdesc_,
        cached.Adesc_,
        cached.Cdesc_,
        cached.Ddesc_,
        &algo,
        &explicit_result);
    if (status != CUBLAS_STATUS_SUCCESS || explicit_result.state != CUBLAS_STATUS_SUCCESS) {
        if (error_message != nullptr) {
            *error_message =
                "cublasLtMatmulAlgoCheck failed for layer=" + layer_prefix_ +
                " stage=" + stage_key(ctx.stage) +
                " impl_params=" + cached.selected_impl_params_.dump() +
                " status=" + std::to_string(static_cast<int>(status)) +
                " result_state=" + std::to_string(static_cast<int>(explicit_result.state));
        }
        return false;
    }

    const size_t max_workspace_bytes = cublaslt_max_workspace_bytes();
    if (explicit_result.workspaceSize > max_workspace_bytes) {
        if (error_message != nullptr) {
            *error_message =
                "cublasLt explicit algo requires workspace=" + std::to_string(explicit_result.workspaceSize) +
                " bytes, exceeds limit=" + std::to_string(max_workspace_bytes);
        }
        return false;
    }

    cached.heuristic_ = explicit_result;
    cached.has_algo_ = true;
    cached.best_algo_index_ = -1;
    return true;
}

nlohmann::json LinearLayer::debug_enumerate_cublaslt_candidates(
    const Tensor& input,
    Tensor& output,
    ModelStage stage,
    int32_t max_algo_ids,
    int32_t top_k)
{
    check<InvalidRequestError>(cublaslt_handle_ != nullptr, "LinearLayer: CUBLASLt handle is null");

    const WeightSet& weight_set = (stage == ModelStage::Prefill) ? prefill_weights_ : decode_weights_;
    check<InvalidRequestError>(
        weight_set.quant_type_ == QuantType::FP16_BF16 && weight_set.weight_ != nullptr,
        "LinearLayer: debug_enumerate_cublaslt_candidates requires FP16/BF16 weights");

    const auto& input_shape = input.shape();
    const auto& output_shape = output.shape();
    check<InvalidRequestError>(
        input_shape.size() == 2 && output_shape.size() == 2,
        "LinearLayer: debug_enumerate_cublaslt_candidates expects 2D input/output");
    check<InvalidRequestError>(
        input_shape[0] == output_shape[0] &&
            input_shape[1] == static_cast<int64_t>(in_features_) &&
            output_shape[1] == static_cast<int64_t>(out_features_),
        "LinearLayer: debug_enumerate_cublaslt_candidates shape mismatch");

    const DType input_dtype = input.dtype();
    const DType weight_dtype = weight_set.weight_->dtype();
    const DType output_dtype = output.dtype();
    check<InvalidRequestError>(
        (input_dtype == DType::Float16 || input_dtype == DType::BFloat16) &&
            (weight_dtype == DType::Float16 || weight_dtype == DType::BFloat16) &&
            (output_dtype == DType::Float16 || output_dtype == DType::BFloat16 || output_dtype == DType::Float32),
        "LinearLayer: debug_enumerate_cublaslt_candidates only supports FP16/BF16 paths");

    const int m = static_cast<int>(input_shape[0]);
    const void* bias_ptr = weight_set.bias_ ? weight_set.bias_->data_ptr() : nullptr;
    CachedDescriptors& cached = (stage == ModelStage::Prefill)
        ? prefill_descriptors_map_[m]
        : decode_descriptors_;

    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr;
    cublasLtMatrixLayout_t Bdesc = nullptr;
    cublasLtMatrixLayout_t Cdesc = nullptr;
    cublasLtMatrixLayout_t Ddesc = nullptr;
    get_or_create_descriptors(
        m,
        input_dtype == DType::BFloat16 ? CUDA_R_16BF : CUDA_R_16F,
        weight_dtype == DType::BFloat16 ? CUDA_R_16BF : CUDA_R_16F,
        output_dtype == DType::BFloat16 ? CUDA_R_16BF : (output_dtype == DType::Float32 ? CUDA_R_32F : CUDA_R_16F),
        bias_ptr,
        cached,
        matmul_desc,
        Adesc,
        Bdesc,
        Cdesc,
        Ddesc);
    (void)matmul_desc;
    (void)Adesc;
    (void)Bdesc;
    (void)Cdesc;
    (void)Ddesc;

    const int requested_algo_ids = std::max(1, max_algo_ids);
    std::vector<int> algo_ids(static_cast<size_t>(requested_algo_ids));
    int returned_algo_ids = 0;
    const cublasStatus_t get_ids_status = cublasLtMatmulAlgoGetIds(
        cublaslt_handle_,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F,
        cached.cached_weight_type_,
        cached.cached_input_type_,
        cached.cached_output_type_,
        cached.cached_output_type_,
        requested_algo_ids,
        algo_ids.data(),
        &returned_algo_ids);
    check<DeviceError>(
        get_ids_status == CUBLAS_STATUS_SUCCESS,
        "LinearLayer: cublasLtMatmulAlgoGetIds failed with status " +
            std::to_string(static_cast<int>(get_ids_status)));

    const size_t max_workspace_bytes = cublaslt_max_workspace_bytes();
    std::set<std::string> seen_configs;
    std::vector<nlohmann::json> candidates;

    const auto maybe_push_candidate = [&](const cublasLtMatmulHeuristicResult_t& result, const char* source) {
        nlohmann::json config = cublaslt_algo_to_json(result.algo);
        const std::string key = config.dump();
        if (!seen_configs.insert(key).second) {
            return;
        }
        candidates.push_back({
            {"source", source},
            {"config", config},
            {"workspace_bytes", result.workspaceSize},
            {"waves_count", result.wavesCount},
        });
    };

    for (int idx = 0; idx < returned_algo_ids; ++idx) {
        cublasLtMatmulHeuristicResult_t result = {};
        std::string error_message;
        if (!get_best_cublaslt_heuristic_for_algo_id(cached, algo_ids[idx], &result, &error_message)) {
            continue;
        }
        maybe_push_candidate(result, "limited_by_algo_id");
    }

    if (top_k > 0 && static_cast<int32_t>(candidates.size()) > top_k) {
        candidates.resize(static_cast<size_t>(top_k));
    }

    return {
        {"layer_prefix", layer_prefix_},
        {"layer_role", layer_role_},
        {"stage", stage == ModelStage::Decode ? "decode" : "prefill"},
        {"m", m},
        {"max_workspace_bytes", max_workspace_bytes},
        {"requested_algo_ids", requested_algo_ids},
        {"returned_algo_ids", returned_algo_ids},
        {"search_mode", "limited_by_algo_id"},
        {"candidates", candidates},
    };
}

nlohmann::json LinearLayer::debug_describe_cublaslt_algo(
    const Tensor& input,
    Tensor& output,
    int32_t algo_id,
    ModelStage stage)
{
    check<InvalidRequestError>(algo_id >= 0, "LinearLayer: algo_id must be non-negative");
    check<InvalidRequestError>(cublaslt_handle_ != nullptr, "LinearLayer: CUBLASLt handle is null");

    const WeightSet& weight_set = (stage == ModelStage::Prefill) ? prefill_weights_ : decode_weights_;
    check<InvalidRequestError>(
        weight_set.quant_type_ == QuantType::FP16_BF16 && weight_set.weight_ != nullptr,
        "LinearLayer: debug_describe_cublaslt_algo requires FP16/BF16 weights");

    const auto& input_shape = input.shape();
    const auto& output_shape = output.shape();
    check<InvalidRequestError>(
        input_shape.size() == 2 && output_shape.size() == 2,
        "LinearLayer: debug_describe_cublaslt_algo expects 2D input/output");
    check<InvalidRequestError>(
        input_shape[0] == output_shape[0] &&
            input_shape[1] == static_cast<int64_t>(in_features_) &&
            output_shape[1] == static_cast<int64_t>(out_features_),
        "LinearLayer: debug_describe_cublaslt_algo shape mismatch");

    const DType input_dtype = input.dtype();
    const DType weight_dtype = weight_set.weight_->dtype();
    const DType output_dtype = output.dtype();
    check<InvalidRequestError>(
        (input_dtype == DType::Float16 || input_dtype == DType::BFloat16) &&
            (weight_dtype == DType::Float16 || weight_dtype == DType::BFloat16) &&
            (output_dtype == DType::Float16 || output_dtype == DType::BFloat16 || output_dtype == DType::Float32),
        "LinearLayer: debug_describe_cublaslt_algo only supports FP16/BF16 paths");

    const int m = static_cast<int>(input_shape[0]);
    const void* bias_ptr = weight_set.bias_ ? weight_set.bias_->data_ptr() : nullptr;
    CachedDescriptors& cached = (stage == ModelStage::Prefill)
        ? prefill_descriptors_map_[m]
        : decode_descriptors_;

    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr;
    cublasLtMatrixLayout_t Bdesc = nullptr;
    cublasLtMatrixLayout_t Cdesc = nullptr;
    cublasLtMatrixLayout_t Ddesc = nullptr;
    get_or_create_descriptors(
        m,
        input_dtype == DType::BFloat16 ? CUDA_R_16BF : CUDA_R_16F,
        weight_dtype == DType::BFloat16 ? CUDA_R_16BF : CUDA_R_16F,
        output_dtype == DType::BFloat16 ? CUDA_R_16BF : (output_dtype == DType::Float32 ? CUDA_R_32F : CUDA_R_16F),
        bias_ptr,
        cached,
        matmul_desc,
        Adesc,
        Bdesc,
        Cdesc,
        Ddesc);
    (void)matmul_desc;
    (void)Adesc;
    (void)Bdesc;
    (void)Cdesc;
    (void)Ddesc;

    cublasLtMatmulAlgo_t algo = {};
    const cublasStatus_t init_status = cublasLtMatmulAlgoInit(
        cublaslt_handle_,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F,
        cached.cached_weight_type_,
        cached.cached_input_type_,
        cached.cached_output_type_,
        cached.cached_output_type_,
        algo_id,
        &algo);
    check<DeviceError>(
        init_status == CUBLAS_STATUS_SUCCESS,
        "LinearLayer: cublasLtMatmulAlgoInit failed for algo_id=" + std::to_string(algo_id) +
            " status=" + std::to_string(static_cast<int>(init_status)));

    cublasLtMatmulHeuristicResult_t limited_result = {};
    std::string limited_error;
    const bool has_limited_result = get_best_cublaslt_heuristic_for_algo_id(
        cached, algo_id, &limited_result, &limited_error);

    int32_t splitk_support = 0;
    uint32_t reduction_mask = 0;
    uint32_t cta_swizzling_support = 0;
    int32_t custom_option_max = 0;
    (void)get_cublaslt_algo_cap_attr(algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_support);
    (void)get_cublaslt_algo_cap_attr(algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &reduction_mask);
    (void)get_cublaslt_algo_cap_attr(algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &cta_swizzling_support);
    (void)get_cublaslt_algo_cap_attr(algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custom_option_max);

    nlohmann::json out = {
        {"layer_prefix", layer_prefix_},
        {"layer_role", layer_role_},
        {"stage", stage == ModelStage::Decode ? "decode" : "prefill"},
        {"m", m},
        {"algo_id", algo_id},
        {"default_config", cublaslt_algo_to_json(algo)},
        {"tile_ids", get_cublaslt_algo_cap_vector<uint32_t>(algo, CUBLASLT_ALGO_CAP_TILE_IDS)},
        {"stage_ids", get_cublaslt_algo_cap_vector<uint32_t>(algo, CUBLASLT_ALGO_CAP_STAGES_IDS)},
        {"splitk_support", splitk_support},
        {"reduction_scheme_mask", reduction_mask},
        {"cta_swizzling_support", cta_swizzling_support},
        {"custom_option_max", custom_option_max},
        {"has_limited_result", has_limited_result},
    };
    if (has_limited_result) {
        out["limited_config"] = cublaslt_algo_to_json(limited_result.algo);
        out["limited_workspace_bytes"] = limited_result.workspaceSize;
        out["limited_waves_count"] = limited_result.wavesCount;
    } else {
        out["limited_error"] = limited_error;
    }
    return out;
}

void LinearLayer::forward_fp16_bf16(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream,
    ModelStage stage)
{
    NVTX::Range r(layer_name_);
    check<InvalidRequestError>(cublaslt_handle_ != nullptr,
                              "LinearLayer: CUBLASLt handle is null. Cannot perform FP16/BF16 forward.");
    
    // Select weight set based on stage
    const WeightSet& weight_set = (stage == ModelStage::Prefill) ? prefill_weights_ : decode_weights_;
    std::string stage_str = (stage == ModelStage::Prefill) ? "Prefill" : "Decode";
    check<InvalidRequestError>(weight_set.weight_ != nullptr,
                              "LinearLayer: weight is null for stage " + stage_str);
    
    const auto& input_shape = input.shape();
    check<InvalidRequestError>(input_shape.size() == 2,
                              "LinearLayer: input must be 2D [batch_size, in_features]. "
                              "Got " + std::to_string(input_shape.size()) + "D");
    
    const auto& output_shape = output.shape();
    check<InvalidRequestError>(output_shape.size() == 2,
                              "LinearLayer: output must be 2D [batch_size, out_features]. "
                              "Got " + std::to_string(output_shape.size()) + "D");
    
    int64_t batch_size = input_shape[0];
    int64_t input_in_features = input_shape[1];
    int64_t output_out_features = output_shape[1];
    
    check<InvalidRequestError>(input_in_features == static_cast<int64_t>(in_features_),
                              "LinearLayer: input in_features mismatch. Expected " + 
                              std::to_string(in_features_) + ", got " + std::to_string(input_in_features));
    check<InvalidRequestError>(output_out_features == static_cast<int64_t>(out_features_),
                              "LinearLayer: output out_features mismatch. Expected " + 
                              std::to_string(out_features_) + ", got " + std::to_string(output_out_features));
    check<InvalidRequestError>(input_shape[0] == output_shape[0],
                              "LinearLayer: batch_size mismatch between input and output");
    
    DType input_dtype = input.dtype();
    DType weight_dtype = weight_set.weight_->dtype();
    DType output_dtype = output.dtype();
    
    check<InvalidRequestError>(input_dtype == DType::Float16 || input_dtype == DType::BFloat16,
                              "LinearLayer: input dtype must be Float16 or BFloat16");
    check<InvalidRequestError>(weight_dtype == DType::Float16 || weight_dtype == DType::BFloat16,
                              "LinearLayer: weight dtype must be Float16 or BFloat16");
    check<InvalidRequestError>(output_dtype == DType::Float16 || output_dtype == DType::BFloat16 || output_dtype == DType::Float32,
                              "LinearLayer: output dtype must be Float16, BFloat16, or Float32");
    
    const void* bias_ptr = weight_set.bias_ ? weight_set.bias_->data_ptr() : nullptr;
    
    int m = static_cast<int>(batch_size);
    
    CachedDescriptors& cached = (stage == ModelStage::Prefill)
        ? prefill_descriptors_map_[m]
        : decode_descriptors_;

    LinearOpContext ctx;
    ctx.layer_prefix = layer_prefix_;
    ctx.layer_role = layer_role_;
    ctx.stage = stage;
    ctx.shape = LinearShapeSignature{
        m,
        input_dtype,
        weight_dtype,
        output_dtype,
        in_features_,
        out_features_,
    };
    ctx.has_bias = (bias_ptr != nullptr);

    LinearImpl* impl = resolve_impl(ctx, weight_set, cached);
    impl->forward(*this, ctx, weight_set, input, output, stream, cached);
}

void LinearLayer::forward_int4_groupwise(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream,
    ModelStage stage)
{
    NVTX::Range r(layer_name_);
    // Select weight set based on stage
    const WeightSet& weight_set = (stage == ModelStage::Prefill) ? prefill_weights_ : decode_weights_;
    std::string stage_str = (stage == ModelStage::Prefill) ? "Prefill" : "Decode";
    check<InvalidRequestError>(weight_set.weight_ != nullptr,
                              "LinearLayer: weight is null for stage " + stage_str);
    check<InvalidRequestError>(weight_set.scaling_factors_ != nullptr,
                              "LinearLayer: scaling_factors is null for stage " + stage_str);
    
    const auto& input_shape = input.shape();
    check<InvalidRequestError>(input_shape.size() == 2,
                              "LinearLayer: input must be 2D [batch_size, in_features]. "
                              "Got " + std::to_string(input_shape.size()) + "D");
    
    const auto& output_shape = output.shape();
    check<InvalidRequestError>(output_shape.size() == 2,
                              "LinearLayer: output must be 2D [batch_size, out_features]. "
                              "Got " + std::to_string(output_shape.size()) + "D");
    
    int64_t batch_size = input_shape[0];
    int64_t input_in_features = input_shape[1];
    int64_t output_out_features = output_shape[1];
    
    check<InvalidRequestError>(input_in_features == static_cast<int64_t>(in_features_),
                              "LinearLayer: input in_features mismatch. Expected " + 
                              std::to_string(in_features_) + ", got " + std::to_string(input_in_features));
    check<InvalidRequestError>(output_out_features == static_cast<int64_t>(out_features_),
                              "LinearLayer: output out_features mismatch. Expected " + 
                              std::to_string(out_features_) + ", got " + std::to_string(output_out_features));
    check<InvalidRequestError>(input_shape[0] == output_shape[0],
                              "LinearLayer: batch_size mismatch between input and output");
    
    DType input_dtype = input.dtype();
    DType output_dtype = output.dtype();
    
    check<InvalidRequestError>(input_dtype == DType::Float16,
                              "LinearLayer: INT4 groupwise requires Float16 input. "
                              "Got dtype: " + std::to_string(static_cast<int>(input_dtype)));
    check<InvalidRequestError>(output_dtype == DType::Float16,
                              "LinearLayer: INT4 groupwise requires Float16 output. "
                              "Got dtype: " + std::to_string(static_cast<int>(output_dtype)));
    
    // Get data pointers
    half* input_ptr = static_cast<half*>(input.data_ptr());
    int8_t* weight_ptr = static_cast<int8_t*>(const_cast<void*>(weight_set.weight_->data_ptr()));
    half* scaling_factors_ptr = static_cast<half*>(const_cast<void*>(weight_set.scaling_factors_->data_ptr()));
    half* output_ptr = static_cast<half*>(output.data_ptr());
    
    int m = static_cast<int>(batch_size);
    int n = static_cast<int>(out_features_);
    int k = static_cast<int>(in_features_);
    int group_size = static_cast<int>(weight_set.group_size_);
    
    if (m >= 1 && m <= 4) { // Use GEMV for small batch sizes (optimized for M=1~4)
        gemv_forward_cuda_new(
            input_ptr,
            weight_ptr,
            scaling_factors_ptr,
            output_ptr,
            m,
            n,
            k,
            group_size,
            stream
        );
    } else { // Use GEMM for larger batch sizes
        gemm_forward_cuda_new(
            input_ptr,
            weight_ptr,
            scaling_factors_ptr,
            output_ptr,
            m,
            n,
            k,
            group_size,
            stream
        );
    }
    
    CUDA_CHECK_THROW(cudaGetLastError(), "LinearLayer: INT4 groupwise GEMM/GEMV kernel launch failed");
}

void LinearLayer::forward(
    const std::unordered_map<std::string, Tensor>& inputs,
    std::unordered_map<std::string, Tensor>& outputs,
    cudaStream_t stream,
    ModelStage stage)
{
    check<InvalidRequestError>(is_initialized(), "LinearLayer is not initialized");
    
    const auto& input = inputs.at("input");
    auto& output = outputs.at("output");
    
    // Select weight set based on stage
    const WeightSet& weight_set = (stage == ModelStage::Prefill) ? prefill_weights_ : decode_weights_;
    
    // Dispatch to appropriate forward function based on quantization type
    if (weight_set.quant_type_ == QuantType::INT4_GROUPWISE) {
        forward_int4_groupwise(input, output, stream, stage);
    } else if (weight_set.quant_type_ == QuantType::FP16_BF16) {
        forward_fp16_bf16(input, output, stream, stage);
    } else {
        throw InternalError("LinearLayer: Unsupported quantization type: " + 
                           std::to_string(static_cast<int>(weight_set.quant_type_)));
    }
}

nlohmann::json LinearLayer::debug_cached_impl_info(ModelStage stage, int32_t m) const
{
    const CachedDescriptors* cached = nullptr;
    if (stage == ModelStage::Decode) {
        cached = &decode_descriptors_;
    } else {
        auto it = prefill_descriptors_map_.find(m);
        if (it != prefill_descriptors_map_.end()) {
            cached = &it->second;
        }
    }

    nlohmann::json info = {
        {"layer_prefix", layer_prefix_},
        {"layer_role", layer_role_},
        {"stage", stage == ModelStage::Decode ? "decode" : "prefill"},
        {"requested_m", m},
        {"cached", cached != nullptr},
    };
    if (cached == nullptr) {
        return info;
    }

    info["cached_m"] = cached->cached_m_;
    info["selected_impl_id"] = cached->selected_impl_id_;
    info["selected_impl_params"] = cached->selected_impl_params_;
    info["has_algo"] = cached->has_algo_;
    info["heuristic_candidate_count"] = cached->heuristic_candidate_count_;
    info["best_algo_index"] = cached->best_algo_index_;
    info["workspace_bytes"] = cached->has_algo_ ? cached->heuristic_.workspaceSize : 0;
    info["waves_count"] = cached->has_algo_ ? cached->heuristic_.wavesCount : 0.0;
    info["selected_algo_config"] = cached->has_algo_
        ? describe_cublaslt_algo_config(cached->heuristic_.algo)
        : nlohmann::json::object();
    if (!cached->heuristic_candidates_.empty()) {
        nlohmann::json candidates = nlohmann::json::array();
        for (size_t idx = 0; idx < cached->heuristic_candidates_.size(); ++idx) {
            const auto& candidate = cached->heuristic_candidates_[idx];
            candidates.push_back({
                {"algo_index", idx},
                {"workspace_bytes", candidate.workspaceSize},
                {"waves_count", candidate.wavesCount},
                {"config", describe_cublaslt_algo_config(candidate.algo)},
            });
        }
        info["heuristic_candidates"] = std::move(candidates);
    }
    return info;
}

// ============================================================================
// FusedQKVLinearLayer implementation
// ============================================================================
void FusedQKVLinearLayer::merge_weights(
    const std::unordered_map<std::string, Tensor>& weights,
    Tensor& fused_weight,
    Tensor& fused_bias,
    cudaStream_t stream)
{
    // Find Q, K, V weight tensors
    std::string q_weight_name = layer_prefix_base_ + ".q_proj.weight";
    std::string k_weight_name = layer_prefix_base_ + ".k_proj.weight";
    std::string v_weight_name = layer_prefix_base_ + ".v_proj.weight";
    
    auto q_it = weights.find(q_weight_name);
    auto k_it = weights.find(k_weight_name);
    auto v_it = weights.find(v_weight_name);
    
    check<ConfigurationError>(q_it != weights.end(), "FusedQKVLinearLayer: missing Q weight '" + q_weight_name + "'");
    check<ConfigurationError>(k_it != weights.end(), "FusedQKVLinearLayer: missing K weight '" + k_weight_name + "'");
    check<ConfigurationError>(v_it != weights.end(), "FusedQKVLinearLayer: missing V weight '" + v_weight_name + "'");
    
    const Tensor& q_weight = q_it->second;
    const Tensor& k_weight = k_it->second;
    const Tensor& v_weight = v_it->second;
    
    // Verify weight shapes and dtypes
    const auto& q_shape = q_weight.shape();
    const auto& k_shape = k_weight.shape();
    const auto& v_shape = v_weight.shape();
    
    check<ConfigurationError>(q_shape.size() == 2 && q_shape[0] == static_cast<int64_t>(q_out_features_) && 
                              q_shape[1] == static_cast<int64_t>(in_features_),
                              "FusedQKVLinearLayer: Q weight shape mismatch");
    check<ConfigurationError>(k_shape.size() == 2 && k_shape[0] == static_cast<int64_t>(k_out_features_) && 
                              k_shape[1] == static_cast<int64_t>(in_features_),
                              "FusedQKVLinearLayer: K weight shape mismatch");
    check<ConfigurationError>(v_shape.size() == 2 && v_shape[0] == static_cast<int64_t>(v_out_features_) && 
                              v_shape[1] == static_cast<int64_t>(in_features_),
                              "FusedQKVLinearLayer: V weight shape mismatch");
    
    DType q_dtype = q_weight.dtype();
    DType k_dtype = k_weight.dtype();
    DType v_dtype = v_weight.dtype();
    check<ConfigurationError>(q_dtype == k_dtype && k_dtype == v_dtype,
                              "FusedQKVLinearLayer: Q, K, V weights must have the same dtype");
    check<ConfigurationError>(q_dtype == DType::Float16 || q_dtype == DType::BFloat16,
                              "FusedQKVLinearLayer: weights must be Float16 or BFloat16");
    
    // Allocate fused weight tensor: [q_out + k_out + v_out, in_features]
    uint32_t fused_out_features = q_out_features_ + k_out_features_ + v_out_features_;
    size_t fused_weight_size = fused_out_features * in_features_;
    size_t dtype_size = (q_dtype == DType::Float16) ? sizeof(half) : sizeof(__nv_bfloat16);
    size_t fused_weight_bytes = fused_weight_size * dtype_size;
    
    std::string buffer_name = layer_prefix_base_ + ".qkv_fused.weight";
    void* fused_weight_ptr = StaticBufferManager::get_cache_buf(buffer_name, fused_weight_bytes, device_id_);
    fused_weight = Tensor::view(
        fused_weight_ptr,
        {fused_out_features, in_features_},
        q_dtype,
        Device::GPU,
        device_id_
    );
    
    // Copy Q, K, V weights to fused weight tensor (concatenate along output dimension)
    size_t q_weight_bytes = q_out_features_ * in_features_ * dtype_size;
    size_t k_weight_bytes = k_out_features_ * in_features_ * dtype_size;
    size_t v_weight_bytes = v_out_features_ * in_features_ * dtype_size;
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        fused_weight_ptr,
        q_weight.data_ptr(),
        q_weight_bytes,
        cudaMemcpyDeviceToDevice,
        stream
    ), "FusedQKVLinearLayer: failed to copy Q weight");
    void* k_dst = static_cast<char*>(fused_weight_ptr) + q_weight_bytes;
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        k_dst,
        k_weight.data_ptr(),
        k_weight_bytes,
        cudaMemcpyDeviceToDevice,
        stream
    ), "FusedQKVLinearLayer: failed to copy K weight");
    void* v_dst = static_cast<char*>(fused_weight_ptr) + q_weight_bytes + k_weight_bytes;
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        v_dst,
        v_weight.data_ptr(),
        v_weight_bytes,
        cudaMemcpyDeviceToDevice,
        stream
    ), "FusedQKVLinearLayer: failed to copy V weight");
    
    // Handle bias (if any of Q, K, V has bias, merge them)
    std::string q_bias_name = layer_prefix_base_ + ".q_proj.bias";
    std::string k_bias_name = layer_prefix_base_ + ".k_proj.bias";
    std::string v_bias_name = layer_prefix_base_ + ".v_proj.bias";
    
    auto q_bias_it = weights.find(q_bias_name);
    auto k_bias_it = weights.find(k_bias_name);
    auto v_bias_it = weights.find(v_bias_name);
    
    bool has_q_bias = (q_bias_it != weights.end());
    bool has_k_bias = (k_bias_it != weights.end());
    bool has_v_bias = (v_bias_it != weights.end());
    
    // If any has bias, we need to create fused bias (missing ones are zero)
    if (has_q_bias || has_k_bias || has_v_bias) {
        // Allocate fused bias tensor using StaticBufferManager
        size_t fused_bias_bytes = fused_out_features * dtype_size;
        std::string bias_buffer_name = layer_prefix_base_ + ".qkv_fused.bias";
        void* fused_bias_ptr = StaticBufferManager::get_cache_buf(bias_buffer_name, fused_bias_bytes, device_id_);
        fused_bias = Tensor::view(
            fused_bias_ptr,
            {fused_out_features},
            q_dtype,
            Device::GPU,
            device_id_
        );
        // Initialize fused bias to zero
        CUDA_CHECK_THROW(cudaMemsetAsync(fused_bias_ptr, 0, fused_bias_bytes, stream),
                         "FusedQKVLinearLayer: failed to initialize fused bias");
        // Copy Q bias if exists
        if (has_q_bias) {
            size_t q_bias_bytes = q_out_features_ * dtype_size;
            CUDA_CHECK_THROW(cudaMemcpyAsync(
                fused_bias_ptr,
                q_bias_it->second.data_ptr(),
                q_bias_bytes,
                cudaMemcpyDeviceToDevice,
                stream
            ), "FusedQKVLinearLayer: failed to copy Q bias");
        }
        // Copy K bias if exists
        if (has_k_bias) {
            void* k_bias_dst = static_cast<char*>(fused_bias_ptr) + q_out_features_ * dtype_size;
            size_t k_bias_bytes = k_out_features_ * dtype_size;
            CUDA_CHECK_THROW(cudaMemcpyAsync(
                k_bias_dst,
                k_bias_it->second.data_ptr(),
                k_bias_bytes,
                cudaMemcpyDeviceToDevice,
                stream
            ), "FusedQKVLinearLayer: failed to copy K bias");
        }
        // Copy V bias if exists
        if (has_v_bias) {
            void* v_bias_dst = static_cast<char*>(fused_bias_ptr) + 
                              (q_out_features_ + k_out_features_) * dtype_size;
            size_t v_bias_bytes = v_out_features_ * dtype_size;
            CUDA_CHECK_THROW(cudaMemcpyAsync(
                v_bias_dst,
                v_bias_it->second.data_ptr(),
                v_bias_bytes,
                cudaMemcpyDeviceToDevice,
                stream
            ), "FusedQKVLinearLayer: failed to copy V bias");
        }
    }
}

void FusedQKVLinearLayer::load_weights(
    const std::unordered_map<std::string, Tensor>& prefill_weights,
    const std::unordered_map<std::string, Tensor>& decode_weights)
{
    // Local tensors for merged weights
    Tensor prefill_fused_weight, prefill_fused_bias;
    Tensor decode_fused_weight, decode_fused_bias;
    
    // Merge prefill weights
    merge_weights(prefill_weights, prefill_fused_weight, prefill_fused_bias, nullptr);
    if (!decode_weights.empty() && &decode_weights != &prefill_weights) {
        merge_weights(decode_weights, decode_fused_weight, decode_fused_bias, nullptr);
    }
    
    // Use WeightLoader's mutex to protect weight map modifications
    std::lock_guard<std::mutex> lock(WeightLoader::instance().get_modification_mutex());
    auto& mutable_prefill = const_cast<std::unordered_map<std::string, Tensor>&>(prefill_weights);
    auto& mutable_decode = const_cast<std::unordered_map<std::string, Tensor>&>(decode_weights);
    // Add fused weights to the weight map (use emplace to avoid copy assignment)
    mutable_prefill.erase(layer_prefix_base_ + ".qkv_fused.weight");
    mutable_prefill.emplace(layer_prefix_base_ + ".qkv_fused.weight", std::move(prefill_fused_weight));
    if (prefill_fused_bias.data_ptr() != nullptr) {
        mutable_prefill.erase(layer_prefix_base_ + ".qkv_fused.bias");
        mutable_prefill.emplace(layer_prefix_base_ + ".qkv_fused.bias", std::move(prefill_fused_bias));
    }
    // Remove original Q, K, V weights to save memory
    mutable_prefill.erase(layer_prefix_base_ + ".q_proj.weight");
    mutable_prefill.erase(layer_prefix_base_ + ".k_proj.weight");
    mutable_prefill.erase(layer_prefix_base_ + ".v_proj.weight");
    mutable_prefill.erase(layer_prefix_base_ + ".q_proj.bias");
    mutable_prefill.erase(layer_prefix_base_ + ".k_proj.bias");
    mutable_prefill.erase(layer_prefix_base_ + ".v_proj.bias");
    // Handle decode weights (if different from prefill)
    if (!decode_weights.empty() && &decode_weights != &prefill_weights) {
        mutable_decode.erase(layer_prefix_base_ + ".qkv_fused.weight");
        mutable_decode.emplace(layer_prefix_base_ + ".qkv_fused.weight", std::move(decode_fused_weight));
        if (decode_fused_bias.data_ptr() != nullptr) {
            mutable_decode.erase(layer_prefix_base_ + ".qkv_fused.bias");
            mutable_decode.emplace(layer_prefix_base_ + ".qkv_fused.bias", std::move(decode_fused_bias));
        }
        
        mutable_decode.erase(layer_prefix_base_ + ".q_proj.weight");
        mutable_decode.erase(layer_prefix_base_ + ".k_proj.weight");
        mutable_decode.erase(layer_prefix_base_ + ".v_proj.weight");
        mutable_decode.erase(layer_prefix_base_ + ".q_proj.bias");
        mutable_decode.erase(layer_prefix_base_ + ".k_proj.bias");
        mutable_decode.erase(layer_prefix_base_ + ".v_proj.bias");
    }
    // Call base class load_weights with modified weights (now fused weights are in the map)
    LinearLayer::load_weights(mutable_prefill, mutable_decode);
    
    weights_loaded_ = true;
}

// ============================================================================
// FusedGateUpLinearLayer implementation
// ============================================================================
FusedGateUpLinearLayer::FusedGateUpLinearLayer(
    const std::string& layer_prefix_base,
    const EngineConfig& engine_config,
    uint32_t in_features,
    uint32_t gate_out_features,
    uint32_t up_out_features,
    std::string layer_name)
    : LinearLayer(layer_prefix_base + ".gate_up_fused",
                  engine_config,
                  in_features,
                  gate_out_features + up_out_features,
                  std::move(layer_name)),
      in_features_(in_features),
      gate_out_features_(gate_out_features),
      up_out_features_(up_out_features),
      layer_prefix_base_(layer_prefix_base) {}

FusedGateUpLinearLayer::~FusedGateUpLinearLayer() = default;

void FusedGateUpLinearLayer::merge_weights(
    const std::unordered_map<std::string, Tensor>& weights,
    Tensor& fused_weight,
    Tensor& fused_bias,
    cudaStream_t stream)
{
    // Find gate, up weight tensors
    std::string gate_weight_name = layer_prefix_base_ + ".gate_proj.weight";
    std::string up_weight_name = layer_prefix_base_ + ".up_proj.weight";
    
    auto gate_it = weights.find(gate_weight_name);
    auto up_it = weights.find(up_weight_name);
    
    check<ConfigurationError>(gate_it != weights.end(),
                              "FusedGateUpLinearLayer: missing gate weight '" + gate_weight_name + "'");
    check<ConfigurationError>(up_it != weights.end(),
                              "FusedGateUpLinearLayer: missing up weight '" + up_weight_name + "'");
    
    const Tensor& gate_weight = gate_it->second;
    const Tensor& up_weight = up_it->second;
    
    // Verify weight shapes
    const auto& gate_shape = gate_weight.shape();
    const auto& up_shape = up_weight.shape();
    
    check<ConfigurationError>(gate_shape.size() == 2 && gate_shape[0] == gate_out_features_ && gate_shape[1] == in_features_,
                              "FusedGateUpLinearLayer: gate weight shape mismatch");
    check<ConfigurationError>(up_shape.size() == 2 && up_shape[0] == up_out_features_ && up_shape[1] == in_features_,
                              "FusedGateUpLinearLayer: up weight shape mismatch");
    
    // Check dtypes
    DType gate_dtype = gate_weight.dtype();
    DType up_dtype = up_weight.dtype();
    
    check<ConfigurationError>(gate_dtype == up_dtype,
                              "FusedGateUpLinearLayer: gate, up weights must have the same dtype");
    check<ConfigurationError>(gate_dtype == DType::Float16 || gate_dtype == DType::BFloat16,
                              "FusedGateUpLinearLayer: weights must be Float16 or BFloat16");
    
    // Allocate fused weight tensor: [gate_out + up_out, in_features]
    // Use StaticBufferManager for persistent weight storage (not MemoryPool for temporary activations)
    uint32_t fused_out_features = gate_out_features_ + up_out_features_;
    size_t fused_weight_size = fused_out_features * in_features_;
    size_t dtype_size = (gate_dtype == DType::Float16) ? sizeof(half) : sizeof(__nv_bfloat16);
    size_t fused_weight_bytes = fused_weight_size * dtype_size;
    
    std::string buffer_name = layer_prefix_base_ + ".gate_up_fused.weight";
    void* fused_weight_ptr = StaticBufferManager::get_cache_buf(buffer_name, fused_weight_bytes, device_id_);
    fused_weight = Tensor::view(
        fused_weight_ptr,
        {fused_out_features, in_features_},
        gate_dtype,
        Device::GPU,
        device_id_
    );
    
    size_t up_weight_bytes = up_out_features_ * in_features_ * dtype_size;
    size_t gate_weight_bytes = gate_out_features_ * in_features_ * dtype_size;
    
    // Internal layout is [up, gate] so decode can directly feed TRT-LLM's fused SwiGLU kernel
    // without keeping another reordered copy of the weight tensor.
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        fused_weight_ptr,
        up_weight.data_ptr(),
        up_weight_bytes,
        cudaMemcpyDeviceToDevice,
        stream
    ), "FusedGateUpLinearLayer: failed to copy up weight");
    
    void* gate_dst = static_cast<char*>(fused_weight_ptr) + up_weight_bytes;
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        gate_dst,
        gate_weight.data_ptr(),
        gate_weight_bytes,
        cudaMemcpyDeviceToDevice,
        stream
    ), "FusedGateUpLinearLayer: failed to copy gate weight");
    
    // Handle bias (if exists)
    std::string gate_bias_name = layer_prefix_base_ + ".gate_proj.bias";
    std::string up_bias_name = layer_prefix_base_ + ".up_proj.bias";
    
    auto gate_bias_it = weights.find(gate_bias_name);
    auto up_bias_it = weights.find(up_bias_name);
    
    bool has_gate_bias = (gate_bias_it != weights.end());
    bool has_up_bias = (up_bias_it != weights.end());
    
    if (has_gate_bias || has_up_bias) {
        // Allocate fused bias tensor using StaticBufferManager
        size_t fused_bias_bytes = fused_out_features * dtype_size;
        std::string bias_buffer_name = layer_prefix_base_ + ".gate_up_fused.bias";
        void* fused_bias_ptr = StaticBufferManager::get_cache_buf(bias_buffer_name, fused_bias_bytes, device_id_);
        fused_bias = Tensor::view(
            fused_bias_ptr,
            {fused_out_features},
            gate_dtype,
            Device::GPU,
            device_id_
        );
        
        // Initialize fused bias to zero
        CUDA_CHECK_THROW(cudaMemsetAsync(fused_bias_ptr, 0, fused_bias_bytes, stream),
                         "FusedGateUpLinearLayer: failed to initialize fused bias");
        
        // Internal bias layout matches the internal weight layout: [up, gate].
        if (has_up_bias) {
            size_t up_bias_bytes = up_out_features_ * dtype_size;
            CUDA_CHECK_THROW(cudaMemcpyAsync(
                fused_bias_ptr,
                up_bias_it->second.data_ptr(),
                up_bias_bytes,
                cudaMemcpyDeviceToDevice,
                stream
            ), "FusedGateUpLinearLayer: failed to copy up bias");
        }
        
        if (has_gate_bias) {
            void* gate_bias_dst = static_cast<char*>(fused_bias_ptr) + up_out_features_ * dtype_size;
            size_t gate_bias_bytes = gate_out_features_ * dtype_size;
            CUDA_CHECK_THROW(cudaMemcpyAsync(
                gate_bias_dst,
                gate_bias_it->second.data_ptr(),
                gate_bias_bytes,
                cudaMemcpyDeviceToDevice,
                stream
            ), "FusedGateUpLinearLayer: failed to copy gate bias");
        }
    }
}

void FusedGateUpLinearLayer::load_weights(
    const std::unordered_map<std::string, Tensor>& prefill_weights,
    const std::unordered_map<std::string, Tensor>& decode_weights)
{
    swiglu_fast_path_cache_.clear();

    // Local tensors for merged weights
    Tensor prefill_fused_weight, prefill_fused_bias;
    Tensor decode_fused_weight, decode_fused_bias;
    
    merge_weights(prefill_weights, prefill_fused_weight, prefill_fused_bias, nullptr);
    if (!decode_weights.empty() && &decode_weights != &prefill_weights) {
        merge_weights(decode_weights, decode_fused_weight, decode_fused_bias, nullptr);
    }
    
    auto& mutable_prefill = const_cast<std::unordered_map<std::string, Tensor>&>(prefill_weights);
    auto& mutable_decode = const_cast<std::unordered_map<std::string, Tensor>&>(decode_weights);
    {
        // Only protect the in-place weight map mutation. The later layer setup can be slow and
        // should not hold the global WeightLoader modification mutex.
        std::lock_guard<std::mutex> lock(WeightLoader::instance().get_modification_mutex());
        mutable_prefill.erase(layer_prefix_base_ + ".gate_up_fused.weight");
        mutable_prefill.emplace(layer_prefix_base_ + ".gate_up_fused.weight", std::move(prefill_fused_weight));
        if (prefill_fused_bias.data_ptr() != nullptr) {
            mutable_prefill.erase(layer_prefix_base_ + ".gate_up_fused.bias");
            mutable_prefill.emplace(layer_prefix_base_ + ".gate_up_fused.bias", std::move(prefill_fused_bias));
        }
        mutable_prefill.erase(layer_prefix_base_ + ".gate_proj.weight");
        mutable_prefill.erase(layer_prefix_base_ + ".up_proj.weight");
        mutable_prefill.erase(layer_prefix_base_ + ".gate_proj.bias");
        mutable_prefill.erase(layer_prefix_base_ + ".up_proj.bias");
        if (!decode_weights.empty() && &decode_weights != &prefill_weights) {
            mutable_decode.erase(layer_prefix_base_ + ".gate_up_fused.weight");
            mutable_decode.emplace(layer_prefix_base_ + ".gate_up_fused.weight", std::move(decode_fused_weight));
            if (decode_fused_bias.data_ptr() != nullptr) {
                mutable_decode.erase(layer_prefix_base_ + ".gate_up_fused.bias");
                mutable_decode.emplace(layer_prefix_base_ + ".gate_up_fused.bias", std::move(decode_fused_bias));
            }

            mutable_decode.erase(layer_prefix_base_ + ".gate_proj.weight");
            mutable_decode.erase(layer_prefix_base_ + ".up_proj.weight");
            mutable_decode.erase(layer_prefix_base_ + ".gate_proj.bias");
            mutable_decode.erase(layer_prefix_base_ + ".up_proj.bias");
        }
    }
    // Call base class load_weights with modified weights (now fused weights are in the map)
    LinearLayer::load_weights(mutable_prefill, mutable_decode);
    
    weights_loaded_ = true;
}

FusedGateUpActivationOp* FusedGateUpLinearLayer::resolve_swiglu_impl(
    const FusedGateUpActivationOpContext& ctx,
    ModelStage stage,
    const std::string& cache_key)
{
    auto& cache_entry = swiglu_fast_path_cache_[cache_key];

    if (cache_entry.impl != nullptr) {
        return cache_entry.impl;
    }

    if (!cache_entry.impl_id.empty()) {
        if (FusedGateUpActivationOp* impl =
                FusedGateUpActivationOpRegistry::instance().find_impl_by_id(cache_entry.impl_id);
            impl != nullptr)
        {
            cache_entry.impl = impl;
            return cache_entry.impl;
        }
        cache_entry.impl_id.clear();
    }

    OperatorQuery query;
    query.op_kind = "fused_gate_up_activation";
    query.layer_role = layer_role_;
    query.op_name = layer_prefix_;
    query.stage = stage_key(stage);
    query.shape_sig = ctx.shape_sig();

    auto resolved = OperatorImplTable::instance().resolve(
        model_name_for_operator_resolution(engine_config_),
        engine_config_.resolved_hw_profile(),
        engine_config_.operator_impl_table_path(),
        query);

    if (resolved.has_value()) {
        if (FusedGateUpActivationOp* impl =
                FusedGateUpActivationOpRegistry::instance().find_impl_by_id(resolved->impl_id);
            impl != nullptr)
        {
            if (!impl->supports(ctx)) {
                throw ConfigurationError(
                    "FusedGateUpLinearLayer: operator_impl_table selected unsupported impl '" +
                    resolved->impl_id + "'");
            }
            cache_entry.impl_id = impl->impl_id();
            cache_entry.impl = impl;
            return cache_entry.impl;
        }
        throw ConfigurationError(
            "FusedGateUpLinearLayer: operator_impl_table selected unknown impl '" + resolved->impl_id + "'");
    }

    if (FusedGateUpActivationOp* impl = FusedGateUpActivationOpRegistry::instance().default_impl(ctx);
        impl != nullptr)
    {
        cache_entry.impl_id = impl->impl_id();
        cache_entry.impl = impl;
        return cache_entry.impl;
    }

    return nullptr;
}

bool FusedGateUpLinearLayer::try_forward_swiglu_fused_impl(
    const Tensor& input,
    Tensor& output,
    ModelStage stage,
    cudaStream_t stream)
{
    check<InvalidRequestError>(
        input.shape().size() == 2 &&
            input.shape()[1] == static_cast<int64_t>(in_features_),
        "FusedGateUpLinearLayer: fused SwiGLU expects input shape [m, in_features]");
    check<InvalidRequestError>(
        output.shape().size() == 2 && output.shape()[0] == input.shape()[0] &&
            output.shape()[1] == static_cast<int64_t>(up_out_features_),
        "FusedGateUpLinearLayer: fused SwiGLU expects output shape [m, up_out_features]");

    const WeightSet& weights = (stage == ModelStage::Decode) ? decode_weights_ : prefill_weights_;
    check<InvalidRequestError>(
        weights.weight_ != nullptr,
        "FusedGateUpLinearLayer: fused SwiGLU requires loaded weights");
    check<InvalidRequestError>(
        input.dtype() == output.dtype() && input.dtype() == weights.weight_->dtype(),
        "FusedGateUpLinearLayer: fused SwiGLU requires input/output/weight dtypes to match");

    FusedGateUpActivationOpContext ctx;
    ctx.layer_prefix = layer_prefix_;
    ctx.layer_role = layer_role_;
    ctx.device_id = device_id_;
    ctx.batch_rows = input.shape()[0];
    ctx.input_features = static_cast<int64_t>(in_features_);
    ctx.gate_output_features = static_cast<int64_t>(gate_out_features_);
    ctx.up_output_features = static_cast<int64_t>(up_out_features_);
    ctx.input_dtype = input.dtype();
    ctx.weight_dtype = weights.weight_->dtype();
    ctx.output_dtype = output.dtype();
    ctx.has_bias = weights.bias_ != nullptr;

    const std::string fast_path_cache_key = stage_key(stage) + "|" + ctx.shape_sig();

    FusedGateUpActivationOp* impl = resolve_swiglu_impl(ctx, stage, fast_path_cache_key);
    if (impl == nullptr) {
        return false;
    }

    auto& cache_entry = swiglu_fast_path_cache_[fast_path_cache_key];
    if (cache_entry.state == nullptr) {
        cache_entry.state = std::make_unique<FusedGateUpActivationOpState>();
    }
    if (!cache_entry.state->initialized) {
        impl->prepare(ctx, *weights.weight_, weights.bias_, *cache_entry.state);
    }
    if (!cache_entry.state->available) {
        return false;
    }

    impl->run(
        ctx,
        *weights.weight_,
        weights.bias_,
        *cache_entry.state,
        input,
        output,
        stream);
    return true;
}

bool FusedGateUpLinearLayer::try_forward_decode_swiglu_fused(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream)
{
    check<InvalidRequestError>(
        input.shape().size() == 2 && input.shape()[0] == 1,
        "FusedGateUpLinearLayer: decode fused SwiGLU expects input shape [1, in_features]");
    check<InvalidRequestError>(
        output.shape().size() == 2 && output.shape()[0] == 1,
        "FusedGateUpLinearLayer: decode fused SwiGLU expects output shape [1, up_out_features]");
    return try_forward_swiglu_fused_impl(input, output, ModelStage::Decode, stream);
}

// ============================================================================
// LMHeadLinearLayer implementation
// ============================================================================

LMHeadLinearLayer::LMHeadLinearLayer(
    const std::string& layer_prefix,
    const EngineConfig& engine_config,
    uint32_t in_features,
    uint32_t out_features,
    std::string layer_name)
    : LinearLayer(layer_prefix, engine_config, in_features, out_features, std::move(layer_name))
{
}

void LMHeadLinearLayer::load_weights(
    const std::unordered_map<std::string, Tensor>& prefill_weights,
    const std::unordered_map<std::string, Tensor>& decode_weights)
{
    // Try candidate weight names in order (support both separate lm_head and tied embedding)
    std::vector<std::string> candidates = {
        layer_prefix_ + ".weight",                    // e.g., "lm_head.weight" (Qwen2.5-7B)
        "model." + layer_prefix_ + ".weight",         // e.g., "model.lm_head.weight"
        "model.embed_tokens.weight",                  // tied weights (tie_word_embeddings=True)
    };

    auto find_weight = [&](const std::unordered_map<std::string, Tensor>& weights) -> const Tensor* {
        for (const auto& name : candidates) {
            auto it = weights.find(name);
            if (it != weights.end()) {
                return &it->second;
            }
        }
        return nullptr;
    };

    const Tensor* prefill_weight = find_weight(prefill_weights);
    if (prefill_weight == nullptr) {
        std::string tried = "";
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (i > 0) tried += ", ";
            tried += candidates[i];
        }
        throw ConfigurationError("LMHeadLinearLayer: none of [" + tried + "] found in prefill_weights");
    }

    prefill_weights_.weight_ = prefill_weight;
    prefill_weights_.quant_type_ = QuantType::FP16_BF16;

    if (!decode_weights.empty()) {
        const Tensor* decode_weight = find_weight(decode_weights);
        if (decode_weight != nullptr) {
            decode_weights_.weight_ = decode_weight;
            decode_weights_.quant_type_ = QuantType::FP16_BF16;
        } else {
            decode_weights_ = prefill_weights_;
        }
    } else {
        decode_weights_ = prefill_weights_;
    }

    weights_loaded_ = true;
}

}  // namespace edge_fm
