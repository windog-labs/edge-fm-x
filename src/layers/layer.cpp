#include "layers/layer.h"
#include "utils/logging.h"
#include "utils/device/weight_loader.h"

namespace edge_fm {

Layer::Layer(const EngineConfig& engine_config, std::string layer_name)
    : engine_config_(engine_config), weights_loaded_(false), layer_name_(std::move(layer_name))
{
    // 从 engine_config 读取 runtime 配置并初始化设备信息
    nlohmann::json runtime_config = engine_config_.runtime();
    auto device_str = runtime_config.value("device", std::string("cuda"));
    device_ = device_from_string(device_str);
    device_id_ = runtime_config.value("device_id", 0);
}

} // namespace edge_fm