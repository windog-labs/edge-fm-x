#pragma once

#include <string>

namespace edge_fm {

enum class BackendTarget {
    Cuda,
    Horizon,
};

inline const char* backend_target_to_string(BackendTarget backend) noexcept {
    switch (backend) {
        case BackendTarget::Cuda:
            return "cuda";
        case BackendTarget::Horizon:
            return "horizon";
    }
    return "unknown";
}

} // namespace edge_fm
