#pragma once

#include "backends/runtime_backend.h"

#include <memory>

namespace edge_fm {

class HorizonRuntimeBackend : public IRuntimeBackend {
public:
    HorizonRuntimeBackend();
    ~HorizonRuntimeBackend() override;

    bool init(const RuntimeInitParams& params) override;
    bool warmup(int batch_size = 1) override;
    RuntimeStreamHandle default_stream() override;

    int forward_sync() override;
    int forward_async(RuntimeStreamHandle stream = nullptr) override;
    int wait(RuntimeStreamHandle stream = nullptr) override;

    std::vector<std::string> input_names() const override;
    std::vector<std::string> output_names() const override;
    bool get_input_shape(const std::string& name, std::vector<int64_t>* out_shape) const override;
    bool get_output_shape(const std::string& name, std::vector<int64_t>* out_shape) const override;
    bool get_input_buffer(const std::string& name, RuntimeTensorView* out_tensor) override;
    bool get_output_buffer(const std::string& name, RuntimeTensorView* out_tensor) override;
    std::string last_error() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace edge_fm
