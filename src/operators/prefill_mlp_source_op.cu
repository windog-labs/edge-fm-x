#include "operators/prefill_mlp_source_op.h"
#include "operators/operator_impl_table.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"
#include "utils/logging.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"

namespace edge_fm {
namespace {

constexpr int kThreads = 256;
constexpr int kSwiGluThreads3BLongPrefill = 128;

int select_swiglu_threads(int m, int hidden, int intermediate)
{
    if (m >= 2048 && hidden == 2048 && intermediate == 11008) {
        return kSwiGluThreads3BLongPrefill;
    }
    return kThreads;
}

bool env_flag_enabled(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return false;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

bool json_bool_or(const nlohmann::json& params, const char* key, bool default_value)
{
    const auto it = params.find(key);
    if (it == params.end() || it->is_null()) {
        return default_value;
    }
    if (it->is_boolean()) {
        return it->get<bool>();
    }
    if (it->is_number_integer()) {
        return it->get<int>() != 0;
    }
    if (it->is_string()) {
        std::string value = it->get<std::string>();
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value == "1" || value == "true" || value == "yes" || value == "on";
    }
    return default_value;
}

int64_t json_i64_or(const nlohmann::json& params, const char* key, int64_t default_value)
{
    const auto it = params.find(key);
    if (it == params.end() || it->is_null()) {
        return default_value;
    }
    if (!it->is_number_integer()) {
        throw ConfigurationError(std::string("CUTLASS prefill MLP source-op impl_params.") + key +
                                 " must be an integer");
    }
    return it->get<int64_t>();
}

std::string json_string_or(const nlohmann::json& params, const char* key, const std::string& default_value)
{
    const auto it = params.find(key);
    if (it == params.end() || it->is_null()) {
        return default_value;
    }
    if (!it->is_string()) {
        throw ConfigurationError(std::string("CUTLASS prefill MLP source-op impl_params.") + key +
                                 " must be a string");
    }
    return it->get<std::string>();
}

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

int64_t parse_env_i64(const char* name, int64_t default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0')) {
        return default_value;
    }
    return static_cast<int64_t>(parsed);
}

std::string env_value_lower(const char* name, const std::string& default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool is_gpu_dtype_2d(const Tensor& tensor, DType dtype)
{
    if (tensor.empty() || tensor.dtype() != dtype || tensor.shape().size() != 2) {
        return false;
    }
    auto [device, _device_id] = tensor.device();
    return device == Device::GPU;
}

int ceil_div(int x, int y)
{
    return (x + y - 1) / y;
}

std::string shape_key(int64_t m, int64_t hidden, int64_t intermediate)
{
    return "m" + std::to_string(m) +
        "_h" + std::to_string(hidden) +
        "_i" + std::to_string(intermediate);
}

std::string matrix_key(const std::vector<int64_t>& shape)
{
    if (shape.size() != 2) {
        return "invalid";
    }
    return "r" + std::to_string(shape[0]) + "_c" + std::to_string(shape[1]);
}

__global__ void bf16_to_half_kernel(
    const __nv_bfloat16* __restrict__ input,
    half* __restrict__ output,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half(input[idx]);
    }
}

__global__ void bf16_to_half2_kernel(
    const __nv_bfloat162* __restrict__ input,
    __half2* __restrict__ output,
    int pairs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pairs) {
        const __nv_bfloat162 value = input[idx];
        output[idx] = __halves2half2(
            __half(__low2bfloat16(value)),
            __half(__high2bfloat16(value)));
    }
}

__global__ void half_to_bf16_kernel(
    const half* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __nv_bfloat16(input[idx]);
    }
}

__global__ void half2_to_bf16_kernel(
    const __half2* __restrict__ input,
    __nv_bfloat162* __restrict__ output,
    int pairs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pairs) {
        const __half2 value = input[idx];
        output[idx] = make_bfloat162(
            __nv_bfloat16(__low2half(value)),
            __nv_bfloat16(__high2half(value)));
    }
}

__global__ void swiglu_kernel(
    const half* __restrict__ gateup,
    half* __restrict__ hidden,
    int rows,
    int intermediate)
{
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= intermediate) {
        return;
    }
    const int idx = row * intermediate + col;
    const int gateup_row_offset = row * intermediate * 2;
    const float up = __half2float(gateup[gateup_row_offset + col]);
    const float gate = __half2float(gateup[gateup_row_offset + intermediate + col]);
    const float silu = gate / (1.0f + expf(-gate));
    hidden[idx] = __float2half(silu * up);
}

__global__ void swiglu_half2_kernel(
    const __half2* __restrict__ gateup,
    __half2* __restrict__ hidden,
    int rows,
    int intermediate_pairs)
{
    const int row = blockIdx.y;
    const int pair_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || pair_col >= intermediate_pairs) {
        return;
    }
    const int idx = row * intermediate_pairs + pair_col;
    const int gateup_row_offset = row * intermediate_pairs * 2;
    const float2 up = __half22float2(gateup[gateup_row_offset + pair_col]);
    const float2 gate = __half22float2(gateup[gateup_row_offset + intermediate_pairs + pair_col]);
    const float2 out{
        (gate.x / (1.0f + expf(-gate.x))) * up.x,
        (gate.y / (1.0f + expf(-gate.y))) * up.y,
    };
    hidden[idx] = __float22half2_rn(out);
}

__global__ void swiglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ gateup,
    __nv_bfloat16* __restrict__ hidden,
    int rows,
    int intermediate)
{
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= intermediate) {
        return;
    }
    const int idx = row * intermediate + col;
    const int gateup_row_offset = row * intermediate * 2;
    const float up = __bfloat162float(gateup[gateup_row_offset + col]);
    const float gate = __bfloat162float(gateup[gateup_row_offset + intermediate + col]);
    const float silu = gate / (1.0f + expf(-gate));
    hidden[idx] = __float2bfloat16(silu * up);
}

__global__ void swiglu_bf162_kernel(
    const __nv_bfloat162* __restrict__ gateup,
    __nv_bfloat162* __restrict__ hidden,
    int rows,
    int intermediate_pairs)
{
    const int row = blockIdx.y;
    const int pair_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || pair_col >= intermediate_pairs) {
        return;
    }
    const int idx = row * intermediate_pairs + pair_col;
    const int gateup_row_offset = row * intermediate_pairs * 2;
    const float2 up = __bfloat1622float2(gateup[gateup_row_offset + pair_col]);
    const float2 gate = __bfloat1622float2(gateup[gateup_row_offset + intermediate_pairs + pair_col]);
    const float2 out{
        (gate.x / (1.0f + expf(-gate.x))) * up.x,
        (gate.y / (1.0f + expf(-gate.y))) * up.y,
    };
    hidden[idx] = __float22bfloat162_rn(out);
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int Stages>
using CutlassGemmF32Accum = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int Stages>
using CutlassGemmF16Accum = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, int Stages>
using CutlassGemmF16AccumTuned = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int Stages>
using CutlassGemmF16AccumBf16Out = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, int Stages>
using CutlassGemmF16AccumBf16OutTuned = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, int Stages>
using CutlassGemmBf16AHalfBBf16OutMixedTuned = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAddMixedInputUpcast>;

template <typename Gemm>
bool launch_gemm(
    const half* a,
    const half* b_transposed_row_major,
    half* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    Gemm gemm_op;
    using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
    using ElementCompute = typename EpilogueOutputOp::ElementCompute;
    typename EpilogueOutputOp::Params epilogue_params(ElementCompute(1), ElementCompute(0));
    typename Gemm::Arguments args(
        {m, n, k},
        {reinterpret_cast<const cutlass::half_t*>(a), k},
        {reinterpret_cast<const cutlass::half_t*>(b_transposed_row_major), k},
        {reinterpret_cast<cutlass::half_t*>(d), n},
        {reinterpret_cast<cutlass::half_t*>(d), n},
        epilogue_params);
    const cutlass::Status status = gemm_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        if (error != nullptr) {
            *error = cutlassGetStatusString(status);
        }
        return false;
    }
    return true;
}

template <typename Gemm>
bool launch_gemm_to_bf16(
    const half* a,
    const half* b_transposed_row_major,
    __nv_bfloat16* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    Gemm gemm_op;
    using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
    using ElementCompute = typename EpilogueOutputOp::ElementCompute;
    typename EpilogueOutputOp::Params epilogue_params(ElementCompute(1), ElementCompute(0));
    typename Gemm::Arguments args(
        {m, n, k},
        {reinterpret_cast<const cutlass::half_t*>(a), k},
        {reinterpret_cast<const cutlass::half_t*>(b_transposed_row_major), k},
        {reinterpret_cast<cutlass::bfloat16_t*>(d), n},
        {reinterpret_cast<cutlass::bfloat16_t*>(d), n},
        epilogue_params);
    const cutlass::Status status = gemm_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        if (error != nullptr) {
            *error = cutlassGetStatusString(status);
        }
        return false;
    }
    return true;
}

template <typename Gemm>
bool launch_mixed_gemm_to_bf16(
    const __nv_bfloat16* a,
    const half* b_transposed_row_major,
    __nv_bfloat16* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    Gemm gemm_op;
    using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
    using ElementCompute = typename EpilogueOutputOp::ElementCompute;
    typename EpilogueOutputOp::Params epilogue_params(ElementCompute(1), ElementCompute(0));
    typename Gemm::Arguments args(
        {m, n, k},
        {reinterpret_cast<const cutlass::bfloat16_t*>(a), k},
        {reinterpret_cast<const cutlass::half_t*>(b_transposed_row_major), k},
        {reinterpret_cast<cutlass::bfloat16_t*>(d), n},
        {reinterpret_cast<cutlass::bfloat16_t*>(d), n},
        epilogue_params);
    const cutlass::Status status = gemm_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        if (error != nullptr) {
            *error = cutlassGetStatusString(status);
        }
        return false;
    }
    return true;
}

enum class AccumMode {
    Fp16,
    Fp32,
};

enum class ActivationMode {
    Fp16Cast,
    MixedBf16,
};

enum class MlpTileMode {
    Default,
    Auto,
    Tile128x128x32,
    Tile128x256x32,
    Tile128x256x32Stage4,
    Tile256x128x32,
    Tile128x128x32Warp64x32,
    Tile128x128x32Warp32x64,
    Tile128x128x64Warp64x32,
    Tile128x128x64Warp32x64,
};

enum class DownOutputMode {
    DirectBf16,
    Fp16Cast,
};

AccumMode parse_accum_mode_value(const std::string& raw)
{
    const std::string value = to_lower(raw);
    if (value == "fp32" || value == "float32") {
        return AccumMode::Fp32;
    }
    if (value == "fp16" || value == "float16") {
        return AccumMode::Fp16;
    }
    throw ConfigurationError("CUTLASS prefill MLP source-op unknown accum mode: " + raw);
}

ActivationMode parse_activation_mode_value(const std::string& raw)
{
    const std::string value = to_lower(raw);
    if (value == "fp16_cast" || value == "half_cast" || value == "fp16") {
        return ActivationMode::Fp16Cast;
    }
    if (value == "mixed_bf16" || value == "bf16_mixed" || value == "bf16") {
        return ActivationMode::MixedBf16;
    }
    throw ConfigurationError("CUTLASS prefill MLP source-op unknown activation_mode: " + raw);
}

AccumMode parse_accum_mode()
{
    const std::string value = env_value_lower("EDGE_FM_CUTLASS_MLP_ACCUM", "fp16");
    if (value == "fp32" || value == "float32") {
        return AccumMode::Fp32;
    }
    return AccumMode::Fp16;
}

MlpTileMode parse_tile_mode_value(const std::string& raw)
{
    const std::string value = to_lower(raw);
    if (value == "auto") {
        return MlpTileMode::Auto;
    }
    if (value == "128x256x32" || value == "tile128x256x32") {
        return MlpTileMode::Tile128x256x32;
    }
    if (value == "128x256x32_s4" || value == "128x256x32_stage4" || value == "tile128x256x32_s4") {
        return MlpTileMode::Tile128x256x32Stage4;
    }
    if (value == "128x128x32" || value == "tile128x128x32") {
        return MlpTileMode::Tile128x128x32;
    }
    if (value == "256x128x32" || value == "tile256x128x32") {
        return MlpTileMode::Tile256x128x32;
    }
    if (value == "128x128x32_warp64x32" || value == "tile128x128x32_warp64x32") {
        return MlpTileMode::Tile128x128x32Warp64x32;
    }
    if (value == "128x128x32_warp32x64" || value == "tile128x128x32_warp32x64") {
        return MlpTileMode::Tile128x128x32Warp32x64;
    }
    if (value == "128x128x64_warp64x32" || value == "tile128x128x64_warp64x32") {
        return MlpTileMode::Tile128x128x64Warp64x32;
    }
    if (value == "128x128x64_warp32x64" || value == "tile128x128x64_warp32x64") {
        return MlpTileMode::Tile128x128x64Warp32x64;
    }
    if (value == "default") {
        return MlpTileMode::Default;
    }
    throw ConfigurationError("CUTLASS prefill MLP source-op unknown tile mode: " + raw);
}

MlpTileMode parse_tile_mode()
{
    const std::string value = env_value_lower("EDGE_FM_CUTLASS_MLP_TILE", "default");
    if (value == "auto") {
        return MlpTileMode::Auto;
    }
    if (value == "128x256x32" || value == "tile128x256x32") {
        return MlpTileMode::Tile128x256x32;
    }
    if (value == "128x256x32_s4" || value == "128x256x32_stage4" || value == "tile128x256x32_s4") {
        return MlpTileMode::Tile128x256x32Stage4;
    }
    if (value == "128x128x32" || value == "tile128x128x32") {
        return MlpTileMode::Tile128x128x32;
    }
    if (value == "256x128x32" || value == "tile256x128x32") {
        return MlpTileMode::Tile256x128x32;
    }
    if (value == "128x128x32_warp64x32" || value == "tile128x128x32_warp64x32") {
        return MlpTileMode::Tile128x128x32Warp64x32;
    }
    if (value == "128x128x32_warp32x64" || value == "tile128x128x32_warp32x64") {
        return MlpTileMode::Tile128x128x32Warp32x64;
    }
    if (value == "128x128x64_warp64x32" || value == "tile128x128x64_warp64x32") {
        return MlpTileMode::Tile128x128x64Warp64x32;
    }
    if (value == "128x128x64_warp32x64" || value == "tile128x128x64_warp32x64") {
        return MlpTileMode::Tile128x128x64Warp32x64;
    }
    return MlpTileMode::Default;
}

DownOutputMode parse_down_output_mode_value(const std::string& raw)
{
    const std::string value = to_lower(raw);
    if (value == "bf16" || value == "direct_bf16") {
        return DownOutputMode::DirectBf16;
    }
    if (value == "fp16_cast" || value == "fp16_then_bf16" || value == "half_cast") {
        return DownOutputMode::Fp16Cast;
    }
    throw ConfigurationError("CUTLASS prefill MLP source-op unknown down_output mode: " + raw);
}

const char* activation_mode_name(ActivationMode mode)
{
    return mode == ActivationMode::MixedBf16 ? "mixed_bf16" : "fp16_cast";
}

const char* accum_mode_name(AccumMode mode)
{
    return mode == AccumMode::Fp32 ? "fp32" : "fp16";
}

const char* down_output_mode_name(DownOutputMode mode)
{
    return mode == DownOutputMode::Fp16Cast ? "fp16_cast" : "bf16";
}

const char* tile_mode_name(MlpTileMode mode)
{
    switch (mode) {
        case MlpTileMode::Auto:
            return "auto";
        case MlpTileMode::Tile128x256x32:
            return "128x256x32";
        case MlpTileMode::Tile128x256x32Stage4:
            return "128x256x32_s4";
        case MlpTileMode::Tile128x128x32:
            return "128x128x32";
        case MlpTileMode::Tile256x128x32:
            return "256x128x32";
        case MlpTileMode::Tile128x128x32Warp64x32:
            return "128x128x32_warp64x32";
        case MlpTileMode::Tile128x128x32Warp32x64:
            return "128x128x32_warp32x64";
        case MlpTileMode::Tile128x128x64Warp64x32:
            return "128x128x64_warp64x32";
        case MlpTileMode::Tile128x128x64Warp32x64:
            return "128x128x64_warp32x64";
        case MlpTileMode::Default:
        default:
            return "default";
    }
}

MlpTileMode resolve_tile_mode(MlpTileMode requested, int hidden, int intermediate)
{
    if (requested != MlpTileMode::Auto) {
        return requested;
    }
    if ((hidden == 2048 && intermediate == 11008) ||
        (hidden == 1536 && intermediate == 8960) ||
        (hidden == 896 && intermediate == 4864)) {
        return MlpTileMode::Tile256x128x32;
    }
    return MlpTileMode::Default;
}

bool launch_best_gemm(
    AccumMode mode,
    MlpTileMode tile_mode,
    const half* a,
    const half* b_transposed_row_major,
    half* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    if (mode == AccumMode::Fp32) {
        return launch_gemm<CutlassGemmF32Accum<128, 128, 64, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x256x32) {
        return launch_gemm<CutlassGemmF16AccumTuned<128, 256, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x256x32Stage4) {
        return launch_gemm<CutlassGemmF16AccumTuned<128, 256, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x32) {
        return launch_gemm<CutlassGemmF16AccumTuned<128, 128, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile256x128x32) {
        return launch_gemm<CutlassGemmF16AccumTuned<256, 128, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x32Warp64x32) {
        return launch_gemm<CutlassGemmF16AccumTuned<128, 128, 32, 64, 32, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x32Warp32x64) {
        return launch_gemm<CutlassGemmF16AccumTuned<128, 128, 32, 32, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x64Warp64x32) {
        return launch_gemm<CutlassGemmF16AccumTuned<128, 128, 64, 64, 32, 64, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x64Warp32x64) {
        return launch_gemm<CutlassGemmF16AccumTuned<128, 128, 64, 32, 64, 64, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    return launch_gemm<CutlassGemmF16Accum<128, 128, 64, 3>>(
        a, b_transposed_row_major, d, m, n, k, stream, error);
}

bool launch_best_gemm_to_bf16(
    AccumMode mode,
    MlpTileMode tile_mode,
    const half* a,
    const half* b_transposed_row_major,
    __nv_bfloat16* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    if (mode != AccumMode::Fp16) {
        if (error != nullptr) {
            *error = "direct BF16 output is only enabled for fp16 accumulation";
        }
        return false;
    }
    if (tile_mode == MlpTileMode::Tile128x256x32) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<128, 256, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x256x32Stage4) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<128, 256, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x32) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<128, 128, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile256x128x32) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<256, 128, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x32Warp64x32) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<128, 128, 32, 64, 32, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x32Warp32x64) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<128, 128, 32, 32, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x64Warp64x32) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<128, 128, 64, 64, 32, 64, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x64Warp32x64) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<128, 128, 64, 32, 64, 64, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    return launch_gemm_to_bf16<CutlassGemmF16AccumBf16Out<128, 128, 64, 3>>(
        a, b_transposed_row_major, d, m, n, k, stream, error);
}

bool launch_best_mixed_gemm_to_bf16(
    AccumMode mode,
    MlpTileMode tile_mode,
    const __nv_bfloat16* a,
    const half* b_transposed_row_major,
    __nv_bfloat16* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    if (mode != AccumMode::Fp16) {
        if (error != nullptr) {
            *error = "mixed BF16 activation GEMM is only enabled for fp16 accumulation";
        }
        return false;
    }
    if (tile_mode == MlpTileMode::Tile128x256x32) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<128, 256, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x256x32Stage4) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<128, 256, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x32) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<128, 128, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile256x128x32) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<256, 128, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x32Warp64x32) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<128, 128, 32, 64, 32, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x32Warp32x64) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<128, 128, 32, 32, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x64Warp64x32) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<128, 128, 64, 64, 32, 64, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == MlpTileMode::Tile128x128x64Warp32x64) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<128, 128, 64, 32, 64, 64, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (error != nullptr) {
        *error = "unsupported mixed BF16 activation tile mode";
    }
    return false;
}

bool launch_bf16_to_half(const Tensor& source, Tensor& destination, cudaStream_t stream)
{
    const auto& shape = source.shape();
    if (!is_gpu_dtype_2d(source, DType::BFloat16) ||
        !is_gpu_dtype_2d(destination, DType::Float16) ||
        destination.shape() != shape) {
        return false;
    }
    const int64_t elements64 = shape[0] * shape[1];
    if (elements64 <= 0 || elements64 > std::numeric_limits<int>::max()) {
        return false;
    }
    const int elements = static_cast<int>(elements64);
    const auto source_addr = reinterpret_cast<std::uintptr_t>(source.data_ptr());
    const auto destination_addr = reinterpret_cast<std::uintptr_t>(destination.data_ptr());
    if ((elements % 2) == 0 && (source_addr % alignof(__nv_bfloat162)) == 0 &&
        (destination_addr % alignof(__half2)) == 0) {
        const int pairs = elements / 2;
        bf16_to_half2_kernel<<<ceil_div(pairs, kThreads), kThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162*>(source.data_ptr()),
            reinterpret_cast<__half2*>(destination.data_ptr()),
            pairs);
    } else {
        bf16_to_half_kernel<<<ceil_div(elements, kThreads), kThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(source.data_ptr()),
            reinterpret_cast<half*>(destination.data_ptr()),
            elements);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool launch_half_to_bf16(const Tensor& source, Tensor& destination, cudaStream_t stream)
{
    const auto& shape = source.shape();
    if (!is_gpu_dtype_2d(source, DType::Float16) ||
        !is_gpu_dtype_2d(destination, DType::BFloat16) ||
        destination.shape() != shape) {
        return false;
    }
    const int64_t elements64 = shape[0] * shape[1];
    if (elements64 <= 0 || elements64 > std::numeric_limits<int>::max()) {
        return false;
    }
    const int elements = static_cast<int>(elements64);
    const auto source_addr = reinterpret_cast<std::uintptr_t>(source.data_ptr());
    const auto destination_addr = reinterpret_cast<std::uintptr_t>(destination.data_ptr());
    if ((elements % 2) == 0 && (source_addr % alignof(__half2)) == 0 &&
        (destination_addr % alignof(__nv_bfloat162)) == 0) {
        const int pairs = elements / 2;
        half2_to_bf16_kernel<<<ceil_div(pairs, kThreads), kThreads, 0, stream>>>(
            reinterpret_cast<const __half2*>(source.data_ptr()),
            reinterpret_cast<__nv_bfloat162*>(destination.data_ptr()),
            pairs);
    } else {
        half_to_bf16_kernel<<<ceil_div(elements, kThreads), kThreads, 0, stream>>>(
            reinterpret_cast<const half*>(source.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(destination.data_ptr()),
            elements);
    }
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

struct PrefillMlpSourceOp::Impl {
    explicit Impl(const EngineConfig& config)
        : model_name(config.resolved_model_name())
        , hw_profile(config.resolved_hw_profile())
        , operator_impl_table_path(config.operator_impl_table_path())
    {
        env_enabled = env_flag_enabled("EDGE_FM_PREFILL_CUTLASS_MLP");
        env_persistent_weights = env_flag_enabled("EDGE_FM_CUTLASS_MLP_PERSISTENT_WEIGHTS");
        env_accum_mode = parse_accum_mode();
        env_requested_tile_mode = parse_tile_mode();
        env_min_m = parse_env_i64("EDGE_FM_CUTLASS_MLP_MIN_M", 64);
        device_id = config.runtime_device_id();

        if (env_enabled) {
            Logging::instance().log_warn(
                "CUTLASS prefill MLP source-op is experimental/default-off; accum={}, tile={}, persistent_weights={}, min_m={}",
                accum_mode_name(env_accum_mode),
                tile_mode_name(env_requested_tile_mode),
                env_persistent_weights ? "on" : "off",
                env_min_m);
        }
    }

    struct RuntimeConfig {
        bool enabled = false;
        bool persistent_weights = false;
        AccumMode accum_mode = AccumMode::Fp16;
        ActivationMode activation_mode = ActivationMode::Fp16Cast;
        MlpTileMode requested_tile_mode = MlpTileMode::Default;
        MlpTileMode gateup_tile_mode = MlpTileMode::Default;
        MlpTileMode down_tile_mode = MlpTileMode::Default;
        DownOutputMode down_output_mode = DownOutputMode::DirectBf16;
        int64_t min_m = 64;
        int swiglu_threads = 0;
        bool table_selected = false;
    };

    std::string source_op_shape_sig(int64_t m, int64_t hidden, int64_t intermediate) const
    {
        return "m=" + std::to_string(m) +
            "|input=" + std::to_string(static_cast<int>(DType::BFloat16)) +
            "|weight=" + std::to_string(static_cast<int>(DType::BFloat16)) +
            "|output=" + std::to_string(static_cast<int>(DType::BFloat16)) +
            "|hidden=" + std::to_string(hidden) +
            "|intermediate=" + std::to_string(intermediate);
    }

    RuntimeConfig resolve_runtime_config(int64_t m, int64_t hidden, int64_t intermediate)
    {
        const std::string shape_sig = source_op_shape_sig(m, hidden, intermediate);
        auto cache_it = runtime_config_cache.find(shape_sig);
        if (cache_it != runtime_config_cache.end()) {
            return cache_it->second;
        }

        RuntimeConfig runtime;
        runtime.enabled = env_enabled;
        runtime.persistent_weights = env_persistent_weights;
        runtime.accum_mode = env_accum_mode;
        runtime.activation_mode = ActivationMode::Fp16Cast;
        runtime.requested_tile_mode = env_requested_tile_mode;
        runtime.gateup_tile_mode = env_requested_tile_mode;
        runtime.down_tile_mode = env_requested_tile_mode;
        runtime.min_m = env_min_m;

        OperatorQuery query;
        query.op_kind = "mlp";
        query.layer_role = "fused_mlp";
        query.stage = "prefill";
        query.shape_sig = shape_sig;
        auto resolved = OperatorImplTable::instance().resolve(
            model_name,
            hw_profile,
            operator_impl_table_path,
            query);
        if (!resolved.has_value() ||
            (resolved->impl_id != "cutlass_prefill_mlp_source_op" &&
             resolved->impl_id != "cutlass_prefill_mlp_bridge"))
        {
            runtime_config_cache.emplace(shape_sig, runtime);
            return runtime;
        }

        runtime.table_selected = true;
        const auto& params = resolved->impl_params;
        runtime.enabled = json_bool_or(params, "enabled", true);
        runtime.persistent_weights = json_bool_or(params, "persistent_weights", runtime.persistent_weights);
        runtime.accum_mode = parse_accum_mode_value(
            json_string_or(params, "accum", accum_mode_name(runtime.accum_mode)));
        runtime.activation_mode = parse_activation_mode_value(
            json_string_or(params, "activation_mode", activation_mode_name(runtime.activation_mode)));
        runtime.requested_tile_mode = parse_tile_mode_value(
            json_string_or(params, "tile", tile_mode_name(runtime.requested_tile_mode)));
        runtime.gateup_tile_mode = parse_tile_mode_value(
            json_string_or(params, "gateup_tile", tile_mode_name(runtime.requested_tile_mode)));
        runtime.down_tile_mode = parse_tile_mode_value(
            json_string_or(params, "down_tile", tile_mode_name(runtime.requested_tile_mode)));
        runtime.down_output_mode = parse_down_output_mode_value(
            json_string_or(params, "down_output", down_output_mode_name(runtime.down_output_mode)));
        runtime.min_m = json_i64_or(params, "min_m", runtime.min_m);
        runtime.swiglu_threads = static_cast<int>(
            json_i64_or(params, "swiglu_threads", select_swiglu_threads(
                static_cast<int>(m), static_cast<int>(hidden), static_cast<int>(intermediate))));
        if (runtime.swiglu_threads <= 0 || runtime.swiglu_threads > 1024) {
            throw ConfigurationError("CUTLASS prefill MLP source-op impl_params.swiglu_threads must be in [1, 1024]");
        }

        if (logged_table_shapes.insert(query.shape_sig).second) {
            Logging::instance().log_debug(
                "CUTLASS prefill MLP source-op selected by operator_impl_table; accum={}, activation_mode={}, tile={}, gateup_tile={}, down_tile={}, down_output={}, persistent_weights={}, min_m={}, swiglu_threads={}, shape={}",
                accum_mode_name(runtime.accum_mode),
                activation_mode_name(runtime.activation_mode),
                tile_mode_name(runtime.requested_tile_mode),
                tile_mode_name(runtime.gateup_tile_mode),
                tile_mode_name(runtime.down_tile_mode),
                down_output_mode_name(runtime.down_output_mode),
                runtime.persistent_weights ? "on" : "off",
                runtime.min_m,
                runtime.swiglu_threads,
                query.shape_sig);
        }
        runtime_config_cache.emplace(shape_sig, runtime);
        return runtime;
    }

    bool try_forward(
        int32_t layer_id,
        const Tensor& input,
        const Tensor& gate_up_weight,
        const Tensor& down_weight,
        Tensor& output,
        cudaStream_t stream)
    {
        if (!is_gpu_dtype_2d(input, DType::BFloat16) ||
            !is_gpu_dtype_2d(gate_up_weight, DType::BFloat16) ||
            !is_gpu_dtype_2d(down_weight, DType::BFloat16) ||
            !is_gpu_dtype_2d(output, DType::BFloat16)) {
            return false;
        }

        const auto& input_shape = input.shape();
        const auto& gate_up_shape = gate_up_weight.shape();
        const auto& down_shape = down_weight.shape();
        const auto& output_shape = output.shape();
        const int64_t m64 = input_shape[0];
        const int64_t hidden64 = input_shape[1];
        if (hidden64 <= 0 ||
            gate_up_shape[0] % 2 != 0 ||
            gate_up_shape[1] != hidden64 ||
            down_shape[0] != hidden64 ||
            output_shape[0] != m64 ||
            output_shape[1] != hidden64) {
            return false;
        }
        const int64_t intermediate64 = gate_up_shape[0] / 2;
        const RuntimeConfig runtime = resolve_runtime_config(m64, hidden64, intermediate64);
        if (!runtime.enabled || persistent_weight_copy_disabled || m64 < runtime.min_m) {
            return false;
        }
        if (down_shape[1] != intermediate64 ||
            m64 > std::numeric_limits<int>::max() ||
            hidden64 > std::numeric_limits<int>::max() ||
            intermediate64 > std::numeric_limits<int>::max()) {
            return false;
        }

        const int m = static_cast<int>(m64);
        const int hidden = static_cast<int>(hidden64);
        const int intermediate = static_cast<int>(intermediate64);
        const int two_intermediate = intermediate * 2;
        const int swiglu_threads = runtime.swiglu_threads > 0
            ? runtime.swiglu_threads
            : select_swiglu_threads(m, hidden, intermediate);
        const std::string shape = shape_key(m64, hidden64, intermediate64);
        const MlpTileMode gateup_tile_mode = resolve_tile_mode(
            runtime.gateup_tile_mode, hidden, intermediate);
        const MlpTileMode down_tile_mode = resolve_tile_mode(
            runtime.down_tile_mode, hidden, intermediate);

        Tensor gate_up_scratch;
        Tensor down_scratch;
        const Tensor* gate_up_bind = bind_fp16_weight(
            layer_id, "gateup", gate_up_weight, gate_up_scratch, runtime.persistent_weights, stream);
        const Tensor* down_bind = bind_fp16_weight(
            layer_id, "down", down_weight, down_scratch, runtime.persistent_weights, stream);
        if (gate_up_bind == nullptr || down_bind == nullptr) {
            return false;
        }

        std::string error;
        if (runtime.activation_mode == ActivationMode::MixedBf16) {
            void* gateup_ptr = StaticBufferManager::get_cache_buf(
                "cutlass_prefill_mlp_gateup_bf16_" + shape,
                static_cast<size_t>(m64) * static_cast<size_t>(two_intermediate) * get_dtype_size(DType::BFloat16),
                device_id);
            void* hidden_ptr = StaticBufferManager::get_cache_buf(
                "cutlass_prefill_mlp_hidden_bf16_" + shape,
                static_cast<size_t>(m64) * static_cast<size_t>(intermediate64) * get_dtype_size(DType::BFloat16),
                device_id);
            Tensor gateup = Tensor::view(gateup_ptr, {m64, two_intermediate}, DType::BFloat16, Device::GPU, device_id);
            Tensor hidden_tensor = Tensor::view(hidden_ptr, {m64, intermediate64}, DType::BFloat16, Device::GPU, device_id);

            if (!launch_best_mixed_gemm_to_bf16(
                    runtime.accum_mode,
                    gateup_tile_mode,
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    reinterpret_cast<const half*>(gate_up_bind->data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(gateup.data_ptr()),
                    m,
                    two_intermediate,
                    hidden,
                    stream,
                    &error)) {
                log_once("gateup_mixed_gemm_failed_" + shape,
                         "CUTLASS prefill MLP source-op mixed gateup GEMM failed for {}: {}; using fallback",
                         shape,
                         error);
                return false;
            }

            const auto gateup_addr = reinterpret_cast<std::uintptr_t>(gateup.data_ptr());
            const auto hidden_addr = reinterpret_cast<std::uintptr_t>(hidden_tensor.data_ptr());
            if ((intermediate % 2) == 0 &&
                (gateup_addr % alignof(__nv_bfloat162)) == 0 &&
                (hidden_addr % alignof(__nv_bfloat162)) == 0) {
                const int intermediate_pairs = intermediate / 2;
                const dim3 grid(ceil_div(intermediate_pairs, swiglu_threads), m);
                swiglu_bf162_kernel<<<grid, swiglu_threads, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat162*>(gateup.data_ptr()),
                    reinterpret_cast<__nv_bfloat162*>(hidden_tensor.data_ptr()),
                    m,
                    intermediate_pairs);
            } else {
                const dim3 grid(ceil_div(intermediate, swiglu_threads), m);
                swiglu_bf16_kernel<<<grid, swiglu_threads, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(gateup.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(hidden_tensor.data_ptr()),
                    m,
                    intermediate);
            }
            if (cudaGetLastError() != cudaSuccess) {
                log_once("swiglu_mixed_failed_" + shape,
                         "CUTLASS prefill MLP source-op mixed SwiGLU kernel launch failed for {}; using fallback",
                         shape);
                return false;
            }

            if (!launch_best_mixed_gemm_to_bf16(
                    runtime.accum_mode,
                    down_tile_mode,
                    reinterpret_cast<const __nv_bfloat16*>(hidden_tensor.data_ptr()),
                    reinterpret_cast<const half*>(down_bind->data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    m,
                    hidden,
                    intermediate,
                    stream,
                    &error)) {
                log_once("down_mixed_gemm_failed_" + shape,
                         "CUTLASS prefill MLP source-op mixed down GEMM failed for {}: {}; using fallback",
                         shape,
                         error);
                return false;
            }
        } else {
            void* x_half_ptr = StaticBufferManager::get_cache_buf(
                "cutlass_prefill_mlp_x_half_" + shape,
                static_cast<size_t>(m64) * static_cast<size_t>(hidden64) * get_dtype_size(DType::Float16),
                device_id);
            void* gateup_ptr = StaticBufferManager::get_cache_buf(
                "cutlass_prefill_mlp_gateup_" + shape,
                static_cast<size_t>(m64) * static_cast<size_t>(two_intermediate) * get_dtype_size(DType::Float16),
                device_id);
            void* hidden_ptr = StaticBufferManager::get_cache_buf(
                "cutlass_prefill_mlp_hidden_" + shape,
                static_cast<size_t>(m64) * static_cast<size_t>(intermediate64) * get_dtype_size(DType::Float16),
                device_id);
            Tensor x_half = Tensor::view(x_half_ptr, {m64, hidden64}, DType::Float16, Device::GPU, device_id);
            Tensor gateup = Tensor::view(gateup_ptr, {m64, two_intermediate}, DType::Float16, Device::GPU, device_id);
            Tensor hidden_tensor = Tensor::view(hidden_ptr, {m64, intermediate64}, DType::Float16, Device::GPU, device_id);

            if (!launch_bf16_to_half(input, x_half, stream)) {
                log_once("input_cast_failed_" + shape,
                         "CUTLASS prefill MLP source-op failed BF16->FP16 input cast for {}; using fallback",
                         shape);
                return false;
            }

            if (!launch_best_gemm(
                    runtime.accum_mode,
                    gateup_tile_mode,
                    reinterpret_cast<const half*>(x_half.data_ptr()),
                    reinterpret_cast<const half*>(gate_up_bind->data_ptr()),
                    reinterpret_cast<half*>(gateup.data_ptr()),
                    m,
                    two_intermediate,
                    hidden,
                    stream,
                    &error)) {
                log_once("gateup_gemm_failed_" + shape,
                         "CUTLASS prefill MLP source-op gateup GEMM failed for {}: {}; using fallback",
                         shape,
                         error);
                return false;
            }

            const auto gateup_addr = reinterpret_cast<std::uintptr_t>(gateup.data_ptr());
            const auto hidden_addr = reinterpret_cast<std::uintptr_t>(hidden_tensor.data_ptr());
            if ((intermediate % 2) == 0 &&
                (gateup_addr % alignof(__half2)) == 0 &&
                (hidden_addr % alignof(__half2)) == 0) {
                const int intermediate_pairs = intermediate / 2;
                const dim3 grid(ceil_div(intermediate_pairs, swiglu_threads), m);
                swiglu_half2_kernel<<<grid, swiglu_threads, 0, stream>>>(
                    reinterpret_cast<const __half2*>(gateup.data_ptr()),
                    reinterpret_cast<__half2*>(hidden_tensor.data_ptr()),
                    m,
                    intermediate_pairs);
            } else {
                const dim3 grid(ceil_div(intermediate, swiglu_threads), m);
                swiglu_kernel<<<grid, swiglu_threads, 0, stream>>>(
                    reinterpret_cast<const half*>(gateup.data_ptr()),
                    reinterpret_cast<half*>(hidden_tensor.data_ptr()),
                    m,
                    intermediate);
            }
            if (cudaGetLastError() != cudaSuccess) {
                log_once("swiglu_failed_" + shape,
                         "CUTLASS prefill MLP source-op SwiGLU kernel launch failed for {}; using fallback",
                         shape);
                return false;
            }

            if (runtime.accum_mode == AccumMode::Fp16 &&
                runtime.down_output_mode == DownOutputMode::DirectBf16) {
                if (!launch_best_gemm_to_bf16(
                        runtime.accum_mode,
                        down_tile_mode,
                        reinterpret_cast<const half*>(hidden_tensor.data_ptr()),
                        reinterpret_cast<const half*>(down_bind->data_ptr()),
                        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                        m,
                        hidden,
                        intermediate,
                        stream,
                        &error)) {
                    log_once("down_gemm_bf16_failed_" + shape,
                             "CUTLASS prefill MLP source-op down GEMM BF16-output failed for {}: {}; using fallback",
                             shape,
                             error);
                    return false;
                }
            } else {
                void* out_half_ptr = StaticBufferManager::get_cache_buf(
                    "cutlass_prefill_mlp_out_half_" + shape,
                    static_cast<size_t>(m64) * static_cast<size_t>(hidden64) * get_dtype_size(DType::Float16),
                    device_id);
                Tensor out_half = Tensor::view(out_half_ptr, {m64, hidden64}, DType::Float16, Device::GPU, device_id);
                if (!launch_best_gemm(
                        runtime.accum_mode,
                        down_tile_mode,
                        reinterpret_cast<const half*>(hidden_tensor.data_ptr()),
                        reinterpret_cast<const half*>(down_bind->data_ptr()),
                        reinterpret_cast<half*>(out_half.data_ptr()),
                        m,
                        hidden,
                        intermediate,
                        stream,
                        &error)) {
                    log_once("down_gemm_failed_" + shape,
                             "CUTLASS prefill MLP source-op down GEMM failed for {}: {}; using fallback",
                             shape,
                             error);
                    return false;
                }

                if (!launch_half_to_bf16(out_half, output, stream)) {
                    log_once("output_cast_failed_" + shape,
                             "CUTLASS prefill MLP source-op failed FP16->BF16 output cast for {}; using fallback",
                             shape);
                    return false;
                }
            }
        }

        return true;
    }

    void reset_runtime_caches()
    {
        persistent_weight_copies.clear();
        persistent_weight_copy_disabled = false;
        logged_messages.clear();
        runtime_config_cache.clear();
    }

    struct PersistentWeightCopy {
        Tensor tensor;
        void* source_ptr = nullptr;
    };

    const Tensor* bind_fp16_weight(
        int32_t layer_id,
        const std::string& role,
        const Tensor& source,
        Tensor& scratch_tensor,
        bool persistent_weight_mode,
        cudaStream_t stream)
    {
        if (persistent_weight_mode) {
            return get_or_create_persistent_fp16_weight(layer_id, role, source, stream);
        }

        const auto& shape = source.shape();
        const size_t bytes = static_cast<size_t>(shape[0]) *
            static_cast<size_t>(shape[1]) * get_dtype_size(DType::Float16);
        void* ptr = StaticBufferManager::get_cache_buf(
            "cutlass_prefill_mlp_" + role + "_fp16_scratch_" + matrix_key(shape),
            bytes,
            device_id);
        scratch_tensor = Tensor::view(ptr, shape, DType::Float16, Device::GPU, device_id);
        if (!launch_bf16_to_half(source, scratch_tensor, stream)) {
            log_once("scratch_weight_cast_failed_" + role + "_" + matrix_key(shape),
                     "CUTLASS prefill MLP source-op failed scratch FP16 {} weight cast for {}; using fallback",
                     role,
                     matrix_key(shape));
            return nullptr;
        }
        return &scratch_tensor;
    }

    const Tensor* get_or_create_persistent_fp16_weight(
        int32_t layer_id,
        const std::string& role,
        const Tensor& source,
        cudaStream_t stream)
    {
        const std::string key = "layer" + std::to_string(layer_id) + "_" + role;
        auto existing = persistent_weight_copies.find(key);
        if (existing != persistent_weight_copies.end() &&
            existing->second.source_ptr == source.data_ptr()) {
            return &existing->second.tensor;
        }
        persistent_weight_copies.erase(key);

        const auto& shape = source.shape();
        if (shape.size() != 2) {
            return nullptr;
        }
        const size_t bytes = static_cast<size_t>(shape[0]) *
            static_cast<size_t>(shape[1]) * get_dtype_size(DType::Float16);
        void* data = nullptr;
        const cudaError_t alloc_status = cudaMalloc(&data, bytes);
        if (alloc_status != cudaSuccess) {
            disable_persistent_weight_copies_after_failure("alloc_failed_" + role, bytes, alloc_status);
            return nullptr;
        }

        Tensor fp16_tensor = Tensor::adopt(
            data,
            shape,
            DType::Float16,
            Device::GPU,
            device_id,
            MemoryOwnership::OwnCudaMalloc);
        if (!launch_bf16_to_half(source, fp16_tensor, stream)) {
            disable_persistent_weight_copies_after_failure(
                "cast_failed_" + role, bytes, cudaGetLastError());
            return nullptr;
        }

        PersistentWeightCopy copy;
        copy.tensor = std::move(fp16_tensor);
        copy.source_ptr = source.data_ptr();
        auto [inserted, _ok] = persistent_weight_copies.emplace(key, std::move(copy));
        Logging::instance().log_debug(
            "CUTLASS prefill MLP source-op created persistent FP16 {} weight copy for layer {} ({} bytes)",
            role,
            layer_id,
            bytes);
        return &inserted->second.tensor;
    }

    void disable_persistent_weight_copies_after_failure(
        const std::string& reason,
        size_t bytes,
        cudaError_t status)
    {
        if (persistent_weight_copy_disabled) {
            return;
        }
        persistent_weight_copy_disabled = true;
        log_once("persistent_weight_mode_disabled_" + reason,
                 "CUTLASS prefill MLP source-op disabled persistent FP16 weight mode after {} ({} bytes, status={}): {}; using fallback",
                 reason,
                 bytes,
                 static_cast<int>(status),
                 cudaGetErrorString(status));
    }

    template <typename... Args>
    void log_once(const std::string& key, const char* fmt, Args&&... args)
    {
        if (logged_messages.insert(key).second) {
            Logging::instance().log_warn(fmt, std::forward<Args>(args)...);
        }
    }

    std::string model_name;
    std::string hw_profile;
    std::string operator_impl_table_path;
    bool env_enabled = false;
    bool env_persistent_weights = false;
    bool persistent_weight_copy_disabled = false;
    AccumMode env_accum_mode = AccumMode::Fp16;
    MlpTileMode env_requested_tile_mode = MlpTileMode::Default;
    int64_t env_min_m = 64;
    int32_t device_id = 0;
    std::unordered_map<std::string, PersistentWeightCopy> persistent_weight_copies;
    std::unordered_map<std::string, RuntimeConfig> runtime_config_cache;
    std::unordered_set<std::string> logged_messages;
    std::unordered_set<std::string> logged_table_shapes;
};

PrefillMlpSourceOp::PrefillMlpSourceOp(const EngineConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
}

PrefillMlpSourceOp::~PrefillMlpSourceOp() = default;

bool PrefillMlpSourceOp::try_forward(
    int32_t layer_id,
    const Tensor& input,
    const Tensor& gate_up_weight,
    const Tensor& down_weight,
    Tensor& output,
    cudaStream_t stream)
{
    return impl_->try_forward(layer_id, input, gate_up_weight, down_weight, output, stream);
}

void PrefillMlpSourceOp::reset_runtime_caches()
{
    impl_->reset_runtime_caches();
}

} // namespace edge_fm
