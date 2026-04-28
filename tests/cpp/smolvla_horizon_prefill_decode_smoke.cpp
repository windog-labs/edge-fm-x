#include <edge-fm/edge-fm.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

void require_tensor(
    const edge_fm::Tensor& tensor,
    const std::vector<int64_t>& expected_shape,
    edge_fm::DType expected_dtype,
    const std::string& name)
{
    require(!tensor.empty(), name + " is empty");
    require(
        tensor.shape() == expected_shape,
        name + " shape mismatch, expected " + shape_to_string(expected_shape) +
            ", got " + shape_to_string(tensor.shape()));
    require(tensor.dtype() == expected_dtype, name + " dtype mismatch");
    const auto [device, device_id] = tensor.device();
    (void)device_id;
    require(device == edge_fm::Device::CPU, name + " should be a CPU tensor");
}

edge_fm::Tensor view_float(std::vector<float>& buffer, const std::vector<int64_t>& shape) {
    return edge_fm::Tensor::view(
        buffer.data(),
        shape,
        edge_fm::DType::Float32,
        edge_fm::Device::CPU);
}

edge_fm::Tensor view_u8(std::vector<uint8_t>& buffer, const std::vector<int64_t>& shape) {
    return edge_fm::Tensor::view(
        buffer.data(),
        shape,
        edge_fm::DType::UInt8,
        edge_fm::Device::CPU);
}

edge_fm::Tensor view_i32(std::vector<int32_t>& buffer, const std::vector<int64_t>& shape) {
    return edge_fm::Tensor::view(
        buffer.data(),
        shape,
        edge_fm::DType::Int32,
        edge_fm::Device::CPU);
}

void run_smoke(const std::string& config_path) {
    constexpr int64_t kPrefixLen = 128;
    constexpr int64_t kSuffixLen = 50;
    constexpr int64_t kHiddenSize = 960;
    constexpr int64_t kExpertHiddenSize = 720;
    constexpr int64_t kNumLayers = 16;
    constexpr int64_t kNumKvHeads = 5;
    constexpr int64_t kHeadDim = 64;

    edge_fm::EdgeFM engine(config_path);

    std::vector<float> prefix_embeds(kPrefixLen * kHiddenSize, 0.0f);
    std::vector<uint8_t> prefix_attention_mask(kPrefixLen * kPrefixLen, 1);
    std::vector<int32_t> prefix_position_ids(kPrefixLen);
    std::iota(prefix_position_ids.begin(), prefix_position_ids.end(), 0);

    edge_fm::Tensor prefix_embeds_tensor = view_float(prefix_embeds, {1, kPrefixLen, kHiddenSize});
    edge_fm::Tensor prefix_mask_tensor = view_u8(prefix_attention_mask, {1, kPrefixLen, kPrefixLen});
    edge_fm::Tensor prefix_pos_tensor = view_i32(prefix_position_ids, {1, kPrefixLen});

    edge_fm::TensorRefMap prefill_inputs{
        {"prefix_embeds", &prefix_embeds_tensor},
        {"prefix_attention_mask", &prefix_mask_tensor},
        {"prefix_position_ids", &prefix_pos_tensor},
    };

    edge_fm::TensorMap prefill_outputs = engine.prefill(0, prefill_inputs);
    for (int64_t layer = 0; layer < kNumLayers; ++layer) {
        const std::string name = "prefix_kv_layer_" + std::to_string(layer);
        auto it = prefill_outputs.find(name);
        require(it != prefill_outputs.end(), "prefill output missing " + name);
        require_tensor(
            it->second,
            {2, kPrefixLen, kNumKvHeads, kHeadDim},
            edge_fm::DType::Float32,
            name);
    }

    std::vector<float> suffix_embeds(kSuffixLen * kExpertHiddenSize, 0.0f);
    std::vector<uint8_t> denoise_attention_mask(kSuffixLen * (kPrefixLen + kSuffixLen), 1);
    std::vector<int32_t> suffix_position_ids(kSuffixLen);
    std::iota(suffix_position_ids.begin(), suffix_position_ids.end(), static_cast<int32_t>(kPrefixLen));

    edge_fm::Tensor suffix_embeds_tensor = view_float(suffix_embeds, {1, kSuffixLen, kExpertHiddenSize});
    edge_fm::Tensor denoise_mask_tensor = view_u8(
        denoise_attention_mask,
        {1, kSuffixLen, kPrefixLen + kSuffixLen});
    edge_fm::Tensor suffix_pos_tensor = view_i32(suffix_position_ids, {1, kSuffixLen});

    edge_fm::TensorRefMap decode_inputs{
        {"suffix_embeds", &suffix_embeds_tensor},
        {"denoise_attention_mask", &denoise_mask_tensor},
        {"suffix_position_ids", &suffix_pos_tensor},
    };

    edge_fm::TensorMap decode_outputs = engine.decode(0, decode_inputs);
    auto expert_hidden_it = decode_outputs.find("expert_hidden");
    require(expert_hidden_it != decode_outputs.end(), "decode output missing expert_hidden");
    require_tensor(
        expert_hidden_it->second,
        {1, kSuffixLen, kExpertHiddenSize},
        edge_fm::DType::Float32,
        "expert_hidden");

    std::cout << "SmolVLA Horizon prefill/decode smoke passed\n";
    std::cout << "prefill outputs: " << prefill_outputs.size() << "\n";
    std::cout << "expert_hidden shape: " << shape_to_string(expert_hidden_it->second.shape()) << "\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <smolvla_horizon_engine.json>\n";
        return 2;
    }

    try {
        run_smoke(argv[1]);
    } catch (const std::exception& exc) {
        std::cerr << "SmolVLA Horizon prefill/decode smoke failed: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
