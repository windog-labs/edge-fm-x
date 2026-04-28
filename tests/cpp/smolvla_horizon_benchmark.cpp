#include <edge-fm/edge-fm.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int64_t kHiddenSize = 960;
constexpr int64_t kExpertHiddenSize = 720;
constexpr int64_t kNumLayers = 16;
constexpr int64_t kNumKvHeads = 5;
constexpr int64_t kHeadDim = 64;

struct Args {
    std::string config_path;
    int64_t prefix_len = 128;
    int64_t suffix_len = 50;
    int warmup = 3;
    int iterations = 20;
    std::string stage = "both";
};

struct Stats {
    double min_ms = 0.0;
    double median_ms = 0.0;
    double mean_ms = 0.0;
    double max_ms = 0.0;
};

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

int64_t parse_i64(const std::string& text, const std::string& name) {
    size_t parsed = 0;
    const int64_t value = std::stoll(text, &parsed);
    require(parsed == text.size() && value > 0, name + " must be a positive integer");
    return value;
}

int parse_int(const std::string& text, const std::string& name) {
    size_t parsed = 0;
    const int value = std::stoi(text, &parsed);
    require(parsed == text.size() && value >= 0, name + " must be a non-negative integer");
    return value;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string value_after_equals(const std::string& arg, const std::string& option) {
    const std::string prefix = option + "=";
    require(starts_with(arg, prefix), "internal parser error for " + option);
    return arg.substr(prefix.size());
}

Args parse_args(int argc, char** argv) {
    if (argc < 2) {
        throw std::runtime_error("missing config path");
    }

    Args args;
    args.config_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (starts_with(arg, "--prefix-len=")) {
            args.prefix_len = parse_i64(value_after_equals(arg, "--prefix-len"), "--prefix-len");
        } else if (starts_with(arg, "--suffix-len=")) {
            args.suffix_len = parse_i64(value_after_equals(arg, "--suffix-len"), "--suffix-len");
        } else if (starts_with(arg, "--warmup=")) {
            args.warmup = parse_int(value_after_equals(arg, "--warmup"), "--warmup");
        } else if (starts_with(arg, "--iterations=")) {
            args.iterations = parse_int(value_after_equals(arg, "--iterations"), "--iterations");
        } else if (starts_with(arg, "--stage=")) {
            args.stage = value_after_equals(arg, "--stage");
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    require(args.iterations > 0, "--iterations must be greater than zero");
    require(
        args.stage == "prefill" || args.stage == "decode" || args.stage == "both",
        "--stage must be one of: prefill, decode, both");
    return args;
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

Stats compute_stats(std::vector<double> values) {
    require(!values.empty(), "cannot compute stats over an empty sample set");
    std::sort(values.begin(), values.end());
    Stats stats;
    stats.min_ms = values.front();
    stats.max_ms = values.back();
    stats.mean_ms = std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        stats.median_ms = (values[mid - 1] + values[mid]) * 0.5;
    } else {
        stats.median_ms = values[mid];
    }
    return stats;
}

class SmolVLABenchmarkInputs {
public:
    SmolVLABenchmarkInputs(int64_t prefix_len, int64_t suffix_len)
        : prefix_len_(prefix_len),
          suffix_len_(suffix_len),
          prefix_embeds_(prefix_len_ * kHiddenSize, 0.0f),
          prefix_attention_mask_(prefix_len_ * prefix_len_, 1),
          prefix_position_ids_(prefix_len_),
          suffix_embeds_(suffix_len_ * kExpertHiddenSize, 0.0f),
          denoise_attention_mask_(suffix_len_ * (prefix_len_ + suffix_len_), 1),
          suffix_position_ids_(suffix_len_)
    {
        std::iota(prefix_position_ids_.begin(), prefix_position_ids_.end(), 0);
        std::iota(
            suffix_position_ids_.begin(),
            suffix_position_ids_.end(),
            static_cast<int32_t>(prefix_len_));

        prefix_embeds_tensor_ = view_float(prefix_embeds_, {1, prefix_len_, kHiddenSize});
        prefix_mask_tensor_ = view_u8(prefix_attention_mask_, {1, prefix_len_, prefix_len_});
        prefix_pos_tensor_ = view_i32(prefix_position_ids_, {1, prefix_len_});
        suffix_embeds_tensor_ = view_float(suffix_embeds_, {1, suffix_len_, kExpertHiddenSize});
        denoise_mask_tensor_ = view_u8(
            denoise_attention_mask_,
            {1, suffix_len_, prefix_len_ + suffix_len_});
        suffix_pos_tensor_ = view_i32(suffix_position_ids_, {1, suffix_len_});
    }

    edge_fm::TensorRefMap prefill_inputs() const {
        return {
            {"prefix_embeds", &prefix_embeds_tensor_},
            {"prefix_attention_mask", &prefix_mask_tensor_},
            {"prefix_position_ids", &prefix_pos_tensor_},
        };
    }

    edge_fm::TensorRefMap decode_inputs() const {
        return {
            {"suffix_embeds", &suffix_embeds_tensor_},
            {"denoise_attention_mask", &denoise_mask_tensor_},
            {"suffix_position_ids", &suffix_pos_tensor_},
        };
    }

private:
    int64_t prefix_len_;
    int64_t suffix_len_;
    std::vector<float> prefix_embeds_;
    std::vector<uint8_t> prefix_attention_mask_;
    std::vector<int32_t> prefix_position_ids_;
    std::vector<float> suffix_embeds_;
    std::vector<uint8_t> denoise_attention_mask_;
    std::vector<int32_t> suffix_position_ids_;
    edge_fm::Tensor prefix_embeds_tensor_;
    edge_fm::Tensor prefix_mask_tensor_;
    edge_fm::Tensor prefix_pos_tensor_;
    edge_fm::Tensor suffix_embeds_tensor_;
    edge_fm::Tensor denoise_mask_tensor_;
    edge_fm::Tensor suffix_pos_tensor_;
};

void validate_prefill_outputs(
    const edge_fm::TensorMap& outputs,
    int64_t prefix_len)
{
    for (int64_t layer = 0; layer < kNumLayers; ++layer) {
        const std::string name = "prefix_kv_layer_" + std::to_string(layer);
        const auto it = outputs.find(name);
        require(it != outputs.end(), "prefill output missing " + name);
        require_tensor(
            it->second,
            {2, prefix_len, kNumKvHeads, kHeadDim},
            edge_fm::DType::Float32,
            name);
    }
}

void validate_decode_outputs(
    const edge_fm::TensorMap& outputs,
    int64_t suffix_len)
{
    const auto it = outputs.find("expert_hidden");
    require(it != outputs.end(), "decode output missing expert_hidden");
    require_tensor(
        it->second,
        {1, suffix_len, kExpertHiddenSize},
        edge_fm::DType::Float32,
        "expert_hidden");
}

template <typename Fn>
std::vector<double> time_iterations(int warmup, int iterations, Fn&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    std::vector<double> elapsed_ms;
    elapsed_ms.reserve(static_cast<size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        elapsed_ms.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }
    return elapsed_ms;
}

void print_result(
    const std::string& stage,
    int64_t prefix_len,
    int64_t suffix_len,
    int warmup,
    int iterations,
    const Stats& stats)
{
    std::cout << std::fixed << std::setprecision(3)
              << "EDGE_FM_SMOLVLA_BENCH"
              << " stage=" << stage
              << " prefix_len=" << prefix_len
              << " suffix_len=" << suffix_len
              << " warmup=" << warmup
              << " iterations=" << iterations
              << " mean_ms=" << stats.mean_ms
              << " median_ms=" << stats.median_ms
              << " min_ms=" << stats.min_ms
              << " max_ms=" << stats.max_ms
              << "\n";
}

void run_benchmark(const Args& args) {
    edge_fm::EdgeFM engine(args.config_path);
    SmolVLABenchmarkInputs inputs(args.prefix_len, args.suffix_len);
    constexpr int32_t kRequestId = 0;

    if (args.stage == "prefill" || args.stage == "both") {
        const auto samples = time_iterations(args.warmup, args.iterations, [&]() {
            edge_fm::TensorMap outputs = engine.prefill(kRequestId, inputs.prefill_inputs());
            validate_prefill_outputs(outputs, args.prefix_len);
        });
        print_result(
            "prefill",
            args.prefix_len,
            args.suffix_len,
            args.warmup,
            args.iterations,
            compute_stats(samples));
    }

    if (args.stage == "decode" || args.stage == "both") {
        {
            edge_fm::TensorMap outputs = engine.prefill(kRequestId, inputs.prefill_inputs());
            validate_prefill_outputs(outputs, args.prefix_len);
        }

        const auto samples = time_iterations(args.warmup, args.iterations, [&]() {
            edge_fm::TensorMap outputs = engine.decode(kRequestId, inputs.decode_inputs());
            validate_decode_outputs(outputs, args.suffix_len);
        });
        print_result(
            "decode",
            args.prefix_len,
            args.suffix_len,
            args.warmup,
            args.iterations,
            compute_stats(samples));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        run_benchmark(args);
    } catch (const std::exception& exc) {
        std::cerr
            << "Usage: " << argv[0]
            << " <smolvla_horizon_engine.json>"
            << " [--prefix-len=N]"
            << " [--suffix-len=N]"
            << " [--warmup=N]"
            << " [--iterations=N]"
            << " [--stage=prefill|decode|both]\n";
        std::cerr << "SmolVLA Horizon benchmark failed: " << exc.what() << "\n";
        return 1;
    }
    return 0;
}
