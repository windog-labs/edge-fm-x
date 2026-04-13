#include "operators/linear_registry.h"

#include "operators/operator_impl_table.h"
#include "utils/check.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace edge_fm {

namespace {

cudaDataType_t to_cuda_type(DType dtype) {
    switch (dtype) {
        case DType::Float16:
            return CUDA_R_16F;
        case DType::BFloat16:
            return CUDA_R_16BF;
        case DType::Float32:
            return CUDA_R_32F;
        default:
            throw InvalidRequestError("LinearLayer: unsupported dtype for CUDA matmul path");
    }
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

template <typename T>
__device__ __forceinline__ float scalar_to_float(T value);

template <>
__device__ __forceinline__ float scalar_to_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float scalar_to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T float_to_scalar(float value);

template <>
__device__ __forceinline__ float float_to_scalar<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ half float_to_scalar<half>(float value) {
    return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename OutT, typename BiasT>
__global__ void add_bias_inplace_kernel(
    OutT* __restrict__ output,
    const BiasT* __restrict__ bias,
    int32_t rows,
    int32_t cols)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total) {
        return;
    }
    int32_t col = static_cast<int32_t>(idx % cols);
    float out_val = scalar_to_float(output[idx]);
    float bias_val = scalar_to_float(bias[col]);
    output[idx] = float_to_scalar<OutT>(out_val + bias_val);
}

void add_bias_inplace(
    Tensor& output,
    const Tensor& bias,
    cudaStream_t stream)
{
    check<InvalidRequestError>(
        output.shape().size() == 2,
        "LinearLayer: add_bias_inplace expects output shape [m, n]");
    check<InvalidRequestError>(
        bias.shape().size() == 1 && bias.shape()[0] == output.shape()[1],
        "LinearLayer: add_bias_inplace expects bias shape [n]");

    const int32_t rows = static_cast<int32_t>(output.shape()[0]);
    const int32_t cols = static_cast<int32_t>(output.shape()[1]);
    const int64_t total = static_cast<int64_t>(rows) * cols;
    constexpr int32_t kBlockSize = 256;
    const int32_t grid = static_cast<int32_t>((total + kBlockSize - 1) / kBlockSize);

    if (output.dtype() == DType::Float16 && bias.dtype() == DType::Float16) {
        add_bias_inplace_kernel<half, half><<<grid, kBlockSize, 0, stream>>>(
            static_cast<half*>(output.data_ptr()),
            static_cast<const half*>(bias.data_ptr()),
            rows,
            cols);
    } else if (output.dtype() == DType::BFloat16 && bias.dtype() == DType::BFloat16) {
        add_bias_inplace_kernel<__nv_bfloat16, __nv_bfloat16><<<grid, kBlockSize, 0, stream>>>(
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<const __nv_bfloat16*>(bias.data_ptr()),
            rows,
            cols);
    } else if (output.dtype() == DType::Float32 && bias.dtype() == DType::Float16) {
        add_bias_inplace_kernel<float, half><<<grid, kBlockSize, 0, stream>>>(
            static_cast<float*>(output.data_ptr()),
            static_cast<const half*>(bias.data_ptr()),
            rows,
            cols);
    } else if (output.dtype() == DType::Float32 && bias.dtype() == DType::BFloat16) {
        add_bias_inplace_kernel<float, __nv_bfloat16><<<grid, kBlockSize, 0, stream>>>(
            static_cast<float*>(output.data_ptr()),
            static_cast<const __nv_bfloat16*>(bias.data_ptr()),
            rows,
            cols);
    } else {
        throw ConfigurationError(
            "LinearLayer: unsupported output/bias dtype combination for add_bias_inplace");
    }

    CUDA_CHECK_THROW(cudaGetLastError(), "LinearLayer: add bias kernel launch failed");
}

} // namespace

class LinearCublasLtImpl final : public LinearLayer::LinearImpl {
public:
    std::string impl_id() const override { return "cublasLt"; }

    bool supports(const LinearLayer::LinearOpContext& ctx, const LinearLayer::WeightSet& weight_set) const override {
        if (weight_set.quant_type_ != LinearLayer::QuantType::FP16_BF16) {
            return false;
        }
        return (ctx.shape.input_dtype == DType::Float16 || ctx.shape.input_dtype == DType::BFloat16) &&
            (ctx.shape.weight_dtype == DType::Float16 || ctx.shape.weight_dtype == DType::BFloat16) &&
            (ctx.shape.output_dtype == DType::Float16 ||
             ctx.shape.output_dtype == DType::BFloat16 ||
             ctx.shape.output_dtype == DType::Float32);
    }

    static bool use_external_bias(
        LinearLayer& owner,
        const LinearLayer::LinearOpContext& ctx,
        const LinearLayer::WeightSet& weight_set)
    {
        if (weight_set.bias_ == nullptr) {
            return false;
        }
        const auto model_config = owner.engine_config_.prefill_model_config();
        const std::string model_type = model_config.value("model_type", std::string{});
        return ctx.layer_role == "fused_qkv" && model_type == "qwen2_5_vl";
    }

    void prepare(
        LinearLayer& owner,
        const LinearLayer::LinearOpContext& ctx,
        const LinearLayer::WeightSet& weight_set,
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream,
        LinearLayer::CachedDescriptors& cached) override
    {
        (void)input;
        (void)output;
        (void)stream;

        check<InvalidRequestError>(
            owner.cublaslt_handle_ != nullptr,
            "LinearLayer: CUBLASLt handle is null. Cannot perform FP16/BF16 forward.");

        const bool external_bias = use_external_bias(owner, ctx, weight_set);
        const void* bias_ptr = external_bias ? nullptr :
            (weight_set.bias_ ? weight_set.bias_->data_ptr() : nullptr);
        cublasLtMatmulDesc_t matmul_desc = nullptr;
        cublasLtMatrixLayout_t Adesc = nullptr;
        cublasLtMatrixLayout_t Bdesc = nullptr;
        cublasLtMatrixLayout_t Cdesc = nullptr;
        cublasLtMatrixLayout_t Ddesc = nullptr;
        owner.get_or_create_descriptors(
            ctx.shape.m,
            to_cuda_type(ctx.shape.input_dtype),
            to_cuda_type(ctx.shape.weight_dtype),
            to_cuda_type(ctx.shape.output_dtype),
            bias_ptr,
            cached,
            matmul_desc,
            Adesc,
            Bdesc,
            Cdesc,
            Ddesc);

        if (cached.has_algo_ && cached.best_algo_index_ < 0) {
            const int32_t algo_index = cached.selected_impl_params_.value("algo_index", -1);
            if (algo_index >= 0 &&
                algo_index < static_cast<int32_t>(cached.heuristic_candidates_.size()))
            {
                cached.heuristic_ = cached.heuristic_candidates_[algo_index];
                cached.best_algo_index_ = algo_index;
                cached.heuristic_candidates_.clear();
            }
        }

        if (cached.has_algo_ && cached.best_algo_index_ < 0 && !cached.heuristic_candidates_.empty()) {
            cached.heuristic_ = cached.heuristic_candidates_.front();
            cached.best_algo_index_ = 0;
            cached.heuristic_candidates_.clear();
        }

        cached.selected_impl_id_ = impl_id();
    }

    void forward(
        LinearLayer& owner,
        const LinearLayer::LinearOpContext& ctx,
        const LinearLayer::WeightSet& weight_set,
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream,
        LinearLayer::CachedDescriptors& cached) override
    {
        prepare(owner, ctx, weight_set, input, output, stream, cached);

        const void* input_ptr = input.data_ptr();
        const void* weight_ptr = weight_set.weight_->data_ptr();
        void* output_ptr = output.data_ptr();
        const bool external_bias = use_external_bias(owner, ctx, weight_set);
        const void* bias_ptr = external_bias ? nullptr :
            (weight_set.bias_ ? weight_set.bias_->data_ptr() : nullptr);

        cublasLtMatmulDesc_t matmul_desc = nullptr;
        cublasLtMatrixLayout_t Adesc = nullptr;
        cublasLtMatrixLayout_t Bdesc = nullptr;
        cublasLtMatrixLayout_t Cdesc = nullptr;
        cublasLtMatrixLayout_t Ddesc = nullptr;
        owner.get_or_create_descriptors(
            ctx.shape.m,
            to_cuda_type(ctx.shape.input_dtype),
            to_cuda_type(ctx.shape.weight_dtype),
            to_cuda_type(ctx.shape.output_dtype),
            bias_ptr,
            cached,
            matmul_desc,
            Adesc,
            Bdesc,
            Cdesc,
            Ddesc);

        void* workspace_ptr = nullptr;
        size_t workspace_bytes = 0;
        if (cached.has_algo_) {
            workspace_bytes = cached.heuristic_.workspaceSize;
            if (workspace_bytes > 0) {
                std::string ws_key = "linear_ws_" + owner.layer_prefix_ + "_" +
                    (ctx.stage == ModelStage::Prefill ? "P_m" + std::to_string(ctx.shape.m) : "D");
                workspace_ptr = StaticBufferManager::get_cache_buf(ws_key, workspace_bytes, owner.device_id_);
            }
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t status = cublasLtMatmul(
            owner.cublaslt_handle_,
            matmul_desc,
            &alpha,
            weight_ptr, Bdesc,
            input_ptr, Adesc,
            &beta,
            output_ptr, Cdesc,
            output_ptr, Ddesc,
            cached.has_algo_ ? &cached.heuristic_.algo : nullptr,
            workspace_ptr,
            workspace_bytes,
            stream);

        check<DeviceError>(
            status == CUBLAS_STATUS_SUCCESS,
            "LinearLayer: cuBLASLt matmul failed with status " +
            std::to_string(static_cast<int>(status)));

        if (external_bias) {
            add_bias_inplace(output, *weight_set.bias_, stream);
        }
    }
};

class LinearCutlassImpl final : public LinearLayer::LinearImpl {
public:
    std::string impl_id() const override { return "cutlass"; }
    bool supports(const LinearLayer::LinearOpContext&, const LinearLayer::WeightSet&) const override { return false; }
    void prepare(
        LinearLayer&,
        const LinearLayer::LinearOpContext&,
        const LinearLayer::WeightSet&,
        const Tensor&,
        Tensor&,
        cudaStream_t,
        LinearLayer::CachedDescriptors&) override {}
    void forward(
        LinearLayer&,
        const LinearLayer::LinearOpContext&,
        const LinearLayer::WeightSet&,
        const Tensor&,
        Tensor&,
        cudaStream_t,
        LinearLayer::CachedDescriptors&) override
    {
        throw InternalError("CUTLASS linear impl is not implemented in this build");
    }
};

class LinearCutileImpl final : public LinearLayer::LinearImpl {
public:
    std::string impl_id() const override { return "cutile"; }
    bool supports(const LinearLayer::LinearOpContext&, const LinearLayer::WeightSet&) const override { return false; }
    void prepare(
        LinearLayer&,
        const LinearLayer::LinearOpContext&,
        const LinearLayer::WeightSet&,
        const Tensor&,
        Tensor&,
        cudaStream_t,
        LinearLayer::CachedDescriptors&) override {}
    void forward(
        LinearLayer&,
        const LinearLayer::LinearOpContext&,
        const LinearLayer::WeightSet&,
        const Tensor&,
        Tensor&,
        cudaStream_t,
        LinearLayer::CachedDescriptors&) override
    {
        throw InternalError(
            "cutile linear impl is not implemented in this build. "
            "Expected a generated kernel artifact / launcher integration.");
    }
};

class LinearAgentImpl final : public LinearLayer::LinearImpl {
public:
    std::string impl_id() const override { return "agent"; }
    bool supports(const LinearLayer::LinearOpContext&, const LinearLayer::WeightSet&) const override { return false; }
    void prepare(
        LinearLayer&,
        const LinearLayer::LinearOpContext&,
        const LinearLayer::WeightSet&,
        const Tensor&,
        Tensor&,
        cudaStream_t,
        LinearLayer::CachedDescriptors&) override {}
    void forward(
        LinearLayer&,
        const LinearLayer::LinearOpContext&,
        const LinearLayer::WeightSet&,
        const Tensor&,
        Tensor&,
        cudaStream_t,
        LinearLayer::CachedDescriptors&) override
    {
        throw InternalError("Agent-optimized linear impl is not implemented in this build");
    }
};

LinearOpRegistry::LinearOpRegistry() {
    impls_.emplace_back(std::make_unique<LinearCublasLtImpl>());
    impls_.emplace_back(std::make_unique<LinearCutlassImpl>());
    impls_.emplace_back(std::make_unique<LinearCutileImpl>());
    impls_.emplace_back(std::make_unique<LinearAgentImpl>());
}

LinearOpRegistry& LinearOpRegistry::instance() {
    static LinearOpRegistry registry;
    return registry;
}

LinearLayer::LinearImpl* LinearOpRegistry::find_impl_by_id(const std::string& impl_id) const {
    for (const auto& impl : impls_) {
        if (impl->impl_id() == impl_id) {
            return impl.get();
        }
    }
    return nullptr;
}

LinearLayer::LinearImpl* LinearOpRegistry::default_impl(
    const LinearLayer::LinearOpContext& ctx,
    const LinearLayer::WeightSet& weight_set) const
{
    for (const auto& impl : impls_) {
        if (impl->supports(ctx, weight_set)) {
            return impl.get();
        }
    }
    return nullptr;
}

LinearLayer::LinearImpl* LinearLayer::find_impl_by_id(const std::string& impl_id) const {
    return LinearOpRegistry::instance().find_impl_by_id(impl_id);
}

LinearLayer::LinearImpl* LinearLayer::resolve_impl(
    const LinearOpContext& ctx,
    const WeightSet& weight_set,
    CachedDescriptors& cached) const
{
    if (!cached.selected_impl_id_.empty()) {
        if (LinearImpl* impl = find_impl_by_id(cached.selected_impl_id_); impl != nullptr) {
            return impl;
        }
        cached.selected_impl_id_.clear();
    }

    OperatorQuery query;
    query.op_kind = "linear";
    query.layer_role = ctx.layer_role;
    query.op_name = ctx.layer_prefix;
    query.stage = stage_key(ctx.stage);
    query.shape_sig = ctx.shape.to_string();

    auto resolved = OperatorImplTable::instance().resolve(
        model_name_for_operator_resolution(engine_config_),
        engine_config_.resolved_hw_profile(),
        engine_config_.operator_impl_table_path(),
        query);

    if (resolved.has_value()) {
        if (LinearImpl* impl = LinearOpRegistry::instance().find_impl_by_id(resolved->impl_id); impl != nullptr) {
            if (!impl->supports(ctx, weight_set)) {
                throw ConfigurationError(
                    "LinearLayer: operator_impl_table selected unsupported impl '" + resolved->impl_id +
                    "' for layer '" + layer_prefix_ + "'");
            }
            cached.selected_impl_id_ = impl->impl_id();
            cached.selected_impl_params_ = resolved->impl_params;
            return impl;
        }
        throw ConfigurationError(
            "LinearLayer: operator_impl_table selected unknown impl '" + resolved->impl_id +
            "' for layer '" + layer_prefix_ + "'");
    }

    if (LinearImpl* impl = LinearOpRegistry::instance().default_impl(ctx, weight_set); impl != nullptr) {
        cached.selected_impl_id_ = impl->impl_id();
        cached.selected_impl_params_ = nlohmann::json::object();
        return impl;
    }

    throw ConfigurationError(
        "LinearLayer: no supported implementation found for layer '" + layer_prefix_ + "'");
}

} // namespace edge_fm
