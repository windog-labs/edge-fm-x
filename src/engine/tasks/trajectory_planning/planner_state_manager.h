#pragma once

#include <edge-fm/core.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace edge_fm {

class PlannerStateManager {
public:
    void put(int32_t request_id, const std::string& name, const Tensor& tensor);
    void put_all(int32_t request_id, const TensorMap& tensors);
    const Tensor* get(int32_t request_id, const std::string& name) const;
    TensorRefMap refs(int32_t request_id) const;
    void clear(int32_t request_id);
    void clear_all();

private:
    std::unordered_map<int32_t, TensorMap> states_;
};

} // namespace edge_fm
