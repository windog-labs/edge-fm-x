#pragma once

#include "engine/engine.h"

#include <cstdint>
#include <memory>
#include <string>

namespace edge_fm {

std::unique_ptr<Engine> create_engine(const EngineConfig& config);
std::unique_ptr<Engine> create_horizon_engine(const EngineConfig& config);
std::unique_ptr<Engine> create_cuda_engine(const EngineConfig& config);
std::string cuda_hardware_fingerprint(int32_t device_id);
std::string cuda_hw_profile(int32_t device_id);

} // namespace edge_fm
