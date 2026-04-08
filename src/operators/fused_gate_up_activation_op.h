#pragma once

#include <edge-fm/core.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace edge_fm {

struct FusedGateUpActivationOpContext {
    std::string layer_prefix;
    std::string layer_role;
    int32_t device_id = 0;
    int64_t input_features = 0;
    int64_t gate_output_features = 0;
    int64_t up_output_features = 0;
    DType input_dtype = DType::Float16;
    DType weight_dtype = DType::Float16;
    DType output_dtype = DType::Float16;
    bool has_bias = false;

    std::string shape_sig() const;
};

struct FusedGateUpActivationOpState {
    bool initialized = false;
    bool available = false;
    std::string unavailable_reason;
    void* expert_offsets_device_ptr = nullptr;
    int sm_count = 0;
    int selected_kernel_config = 0;
    std::string selected_kernel_config_name;
};

class FusedGateUpActivationOp {
public:
    virtual ~FusedGateUpActivationOp() = default;

    virtual std::string impl_id() const = 0;
    virtual bool supports(const FusedGateUpActivationOpContext& ctx) const = 0;
    virtual void prepare(
        const FusedGateUpActivationOpContext& ctx,
        const Tensor& weight,
        const Tensor* bias,
        FusedGateUpActivationOpState& state) = 0;
    virtual void run(
        const FusedGateUpActivationOpContext& ctx,
        const Tensor& weight,
        const Tensor* bias,
        const FusedGateUpActivationOpState& state,
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream) = 0;
};

class FusedGateUpActivationOpRegistry {
public:
    static FusedGateUpActivationOpRegistry& instance();

    FusedGateUpActivationOp* find_impl_by_id(const std::string& impl_id) const;
    FusedGateUpActivationOp* default_impl(const FusedGateUpActivationOpContext& ctx) const;

private:
    FusedGateUpActivationOpRegistry();

    std::vector<std::unique_ptr<FusedGateUpActivationOp>> impls_;
};

} // namespace edge_fm
