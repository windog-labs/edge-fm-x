#include "engine/tasks/trajectory_planning/planner_state_manager.h"

#include "engine/tasks/trajectory_planning/planner_tensor_utils.h"

namespace edge_fm {

void PlannerStateManager::put(int32_t request_id, const std::string& name, const Tensor& tensor) {
    states_[request_id][name] = planner::clone_tensor_to_cpu(tensor);
}

void PlannerStateManager::put_all(int32_t request_id, const TensorMap& tensors) {
    for (const auto& item : tensors) {
        put(request_id, item.first, item.second);
    }
}

const Tensor* PlannerStateManager::get(int32_t request_id, const std::string& name) const {
    auto state_it = states_.find(request_id);
    if (state_it == states_.end()) {
        return nullptr;
    }
    auto tensor_it = state_it->second.find(name);
    if (tensor_it == state_it->second.end()) {
        return nullptr;
    }
    return &tensor_it->second;
}

TensorRefMap PlannerStateManager::refs(int32_t request_id) const {
    TensorRefMap out;
    auto state_it = states_.find(request_id);
    if (state_it == states_.end()) {
        return out;
    }
    for (const auto& item : state_it->second) {
        out.emplace(item.first, &item.second);
    }
    return out;
}

void PlannerStateManager::clear(int32_t request_id) {
    states_.erase(request_id);
}

void PlannerStateManager::clear_all() {
    states_.clear();
}

} // namespace edge_fm
