#include <cstdint>
#include <string>

namespace edge_fm {

std::string cuda_hardware_fingerprint(int32_t device_id) {
    return "compact-vocab-test-device-" + std::to_string(device_id);
}

std::string cuda_hw_profile(int32_t) {
    return "cuda_test";
}

} // namespace edge_fm
