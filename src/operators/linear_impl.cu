#include "layers/linear.h"

#include "operators/operator_impl_table.h"
#include "utils/check.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"

#include <cublasLt.h>

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

        const void* bias_ptr = weight_set.bias_ ? weight_set.bias_->data_ptr() : nullptr;
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
        const void* bias_ptr = weight_set.bias_ ? weight_set.bias_->data_ptr() : nullptr;

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

void LinearLayer::register_default_impls() {
    impls_.emplace_back(std::make_unique<LinearCublasLtImpl>());
    impls_.emplace_back(std::make_unique<LinearCutlassImpl>());
    impls_.emplace_back(std::make_unique<LinearCutileImpl>());
    impls_.emplace_back(std::make_unique<LinearAgentImpl>());
}

LinearLayer::LinearImpl* LinearLayer::find_impl_by_id(const std::string& impl_id) const {
    for (const auto& impl : impls_) {
        if (impl->impl_id() == impl_id) {
            return impl.get();
        }
    }
    return nullptr;
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
    }

    OperatorQuery query;
    query.op_kind = "linear";
    query.layer_role = ctx.layer_role;
    query.op_name = ctx.layer_prefix;
    query.stage = stage_key(ctx.stage);
    query.shape_sig = ctx.shape.to_string();

    auto resolved = OperatorImplTable::instance().resolve(
        engine_config_.resolved_model_name(),
        engine_config_.resolved_hw_profile(),
        engine_config_.operator_impl_table_path(),
        query);

    if (resolved.has_value()) {
        if (LinearImpl* impl = find_impl_by_id(resolved->impl_id); impl != nullptr) {
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

    for (const auto& impl : impls_) {
        if (impl->supports(ctx, weight_set)) {
            cached.selected_impl_id_ = impl->impl_id();
            cached.selected_impl_params_ = nlohmann::json::object();
            return impl.get();
        }
    }

    throw ConfigurationError(
        "LinearLayer: no supported implementation found for layer '" + layer_prefix_ + "'");
}

} // namespace edge_fm
