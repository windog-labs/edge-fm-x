#pragma once

#include <edge-fm/core.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace edge_fm {

enum class ActivationKind {
    kSilu,
};

enum class ActivationInputLayout {
    kGateUp,
    kUpGate,
};

struct ActivationOpContext {
    int64_t batch_size = 0;
    int64_t hidden_size = 0;
    DType dtype = DType::Float16;
    ActivationKind kind = ActivationKind::kSilu;
    ActivationInputLayout input_layout = ActivationInputLayout::kGateUp;
};

class ActivationOp {
public:
    virtual ~ActivationOp() = default;

    virtual std::string impl_id() const = 0;
    virtual bool supports(const ActivationOpContext& ctx) const = 0;
    virtual void act_and_mul(
        const ActivationOpContext& ctx,
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream) = 0;
};

class ActivationOpRegistry {
public:
    static ActivationOpRegistry& instance();

    ActivationOp* find_impl_by_id(const std::string& impl_id) const;
    ActivationOp* default_impl(const ActivationOpContext& ctx) const;

private:
    ActivationOpRegistry();

    std::vector<std::unique_ptr<ActivationOp>> impls_;
};

void activation_act_and_mul(
    const ActivationOpContext& ctx,
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream);

} // namespace edge_fm
