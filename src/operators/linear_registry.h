#pragma once

#include "layers/linear.h"

#include <memory>
#include <string>
#include <vector>

namespace edge_fm {

class LinearOpRegistry {
public:
    static LinearOpRegistry& instance();

    LinearLayer::LinearImpl* find_impl_by_id(const std::string& impl_id) const;
    LinearLayer::LinearImpl* default_impl(
        const LinearLayer::LinearOpContext& ctx,
        const LinearLayer::WeightSet& weight_set) const;

private:
    LinearOpRegistry();

    std::vector<std::unique_ptr<LinearLayer::LinearImpl>> impls_;
};

} // namespace edge_fm
