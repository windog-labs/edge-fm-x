#pragma once

#include "engine/engine.h"

namespace edge_fm {

class HorizonEngine : public Engine {
public:
    explicit HorizonEngine(const EngineConfig& config);
    ~HorizonEngine() override = default;

    void warmup() override;
    void tune() override;
    Response generate(const Request& request) override;
    void prepare_tensors(ModelStage stage, Context& context) override;
};

} // namespace edge_fm
