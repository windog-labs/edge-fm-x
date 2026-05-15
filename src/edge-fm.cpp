#include <edge-fm/edge-fm.h>

#include "engine/engine.h"
#include "engine/engine_factory.h"

namespace edge_fm {

struct EdgeFM::Impl {
    std::unique_ptr<Engine> engine;
};

EdgeFM::EdgeFM(const std::string& config_path) : impl_(std::make_unique<Impl>()) {
    EngineConfig config(config_path);
    impl_->engine = create_engine(config);
    impl_->engine->warmup();
}

EdgeFM::~EdgeFM() noexcept = default;

Response EdgeFM::generate(const Request& request) const {
    return impl_->engine->generate(request);
}

TensorMap EdgeFM::plan(int32_t request_id, const TensorRefMap& inputs) const {
    return impl_->engine->plan(request_id, inputs);
}

TensorMap EdgeFM::run_stage(int32_t request_id, const std::string& stage_name, const TensorRefMap& inputs) const {
    return impl_->engine->run_stage(request_id, stage_name, inputs);
}

TensorMap EdgeFM::prefill(int32_t request_id, const TensorRefMap& inputs) const {
    return impl_->engine->prefill(request_id, inputs);
}

TensorMap EdgeFM::decode(int32_t request_id, const TensorRefMap& inputs) const {
    return impl_->engine->decode(request_id, inputs);
}

std::unordered_map<std::string, double> EdgeFM::last_generate_metrics() const {
    return impl_->engine->get_last_generate_metrics();
}

std::unordered_map<std::string, double> EdgeFM::last_plan_metrics() const {
    return impl_->engine->get_last_plan_metrics();
}

std::unordered_map<std::string, double> EdgeFM::last_stage_metrics() const {
    return impl_->engine->get_last_stage_metrics();
}

void EdgeFM::tune() {
    impl_->engine->tune();
}

} // namespace edge_fm
