#include "operators/prefill_linear_source_op.h"
#include "operators/operator_impl_table.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"
#include "utils/logging.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
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
        throw ConfigurationError(std::string("CUTLASS prefill linear source-op impl_params.") + key +
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
        throw ConfigurationError(std::string("CUTLASS prefill linear source-op impl_params.") + key +
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

bool is_gpu_dtype_2d(const Tensor& tensor, DType dtype)
{
    if (tensor.empty() || tensor.dtype() != dtype || tensor.shape().size() != 2) {
        return false;
    }
    auto [device, _device_id] = tensor.device();
    return device == Device::GPU;
}

bool is_gpu_dtype_1d(const Tensor& tensor, DType dtype)
{
    if (tensor.empty() || tensor.dtype() != dtype || tensor.shape().size() != 1) {
        return false;
    }
    auto [device, _device_id] = tensor.device();
    return device == Device::GPU;
}

int ceil_div(int x, int y)
{
    return (x + y - 1) / y;
}

std::string shape_key(int64_t m, int64_t in_features, int64_t out_features)
{
    return "m" + std::to_string(m) +
        "_k" + std::to_string(in_features) +
        "_n" + std::to_string(out_features);
}

std::string matrix_key(const std::vector<int64_t>& shape)
{
    if (shape.size() != 2) {
        return "invalid";
    }
    return "r" + std::to_string(shape[0]) + "_c" + std::to_string(shape[1]);
}

enum class LinearRole {
    Qkv,
    OProj,
};

const char* role_name(LinearRole role)
{
    return role == LinearRole::Qkv ? "qkv" : "oproj";
}

const char* linear_layer_role(LinearRole role)
{
    return role == LinearRole::Qkv ? "fused_qkv" : "attention_output";
}

bool parse_role(const std::string& raw, LinearRole* out)
{
    const std::string value = to_lower(raw);
    if (value == "qkv" || value == "fused_qkv") {
        *out = LinearRole::Qkv;
        return true;
    }
    if (value == "oproj" || value == "o_proj" || value == "attention_output") {
        *out = LinearRole::OProj;
        return true;
    }
    return false;
}

bool role_list_enables(LinearRole role)
{
    const char* raw = std::getenv("EDGE_FM_CUTLASS_LINEAR_ROLES");
    if (raw == nullptr || *raw == '\0') {
        return true;
    }
    std::string value = to_lower(raw);
    if (value == "both" || value == "all") {
        return true;
    }
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(
            std::remove_if(token.begin(), token.end(),
                           [](unsigned char c) { return std::isspace(c) != 0; }),
            token.end());
        LinearRole parsed = LinearRole::Qkv;
        if (parse_role(token, &parsed) && parsed == role) {
            return true;
        }
    }
    return false;
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
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int rows,
    int cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) {
        return;
    }
    if (bias == nullptr) {
        output[idx] = __nv_bfloat16(input[idx]);
        return;
    }
    float value = __half2float(input[idx]);
    const int col = idx - (idx / cols) * cols;
    value += __bfloat162float(bias[col]);
    output[idx] = __float2bfloat16(value);
}

__global__ void half2_to_bf16_kernel(
    const __half2* __restrict__ input,
    const __nv_bfloat162* __restrict__ bias,
    __nv_bfloat162* __restrict__ output,
    int pairs,
    int cols)
{
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= pairs) {
        return;
    }
    if (bias == nullptr) {
        const __half2 value = input[pair_idx];
        output[pair_idx] = make_bfloat162(
            __nv_bfloat16(__low2half(value)),
            __nv_bfloat16(__high2half(value)));
        return;
    }
    float2 value = __half22float2(input[pair_idx]);
    const int col = (pair_idx * 2) % cols;
    const float2 bias_value = __bfloat1622float2(
        reinterpret_cast<const __nv_bfloat162*>(
            reinterpret_cast<const __nv_bfloat16*>(bias) + col)[0]);
    value.x += bias_value.x;
    value.y += bias_value.y;
    output[pair_idx] = __float22bfloat162_rn(value);
}

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

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int Stages>
using CutlassGemmBf16AHalfBBf16OutMixed = cutlass::gemm::device::Gemm<
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
    cutlass::gemm::GemmShape<64, 64, ThreadblockK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAddMixedInputUpcast>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, int Stages>
using CutlassGemmBf16AHalfBHalfOutMixedTuned = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,
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
    Stages,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAddMixedInputUpcast>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int Stages>
using CutlassGemmBf16AHalfBHalfOutMixed = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<64, 64, ThreadblockK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAddMixedInputUpcast>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, int Stages>
using CutlassGemmBf16ABf16BHalfOutTuned = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM, int WarpN, int WarpK, int Stages>
using CutlassGemmBf16ABf16BBf16OutTuned = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages>;

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
bool launch_mixed_gemm_to_half(
    const __nv_bfloat16* a,
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
        {reinterpret_cast<const cutlass::bfloat16_t*>(a), k},
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

template <typename Gemm>
bool launch_bf16_weight_gemm_to_half(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b_transposed_row_major,
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
        {reinterpret_cast<const cutlass::bfloat16_t*>(a), k},
        {reinterpret_cast<const cutlass::bfloat16_t*>(b_transposed_row_major), k},
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
bool launch_bf16_weight_gemm_to_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b_transposed_row_major,
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
        {reinterpret_cast<const cutlass::bfloat16_t*>(b_transposed_row_major), k},
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

enum class LinearTileMode {
    Default,
    Auto,
    Tile128x128x32,
    Tile128x256x32,
    Tile256x128x32,
};

enum class LinearInputMode {
    Fp16Cast,
    MixedBf16,
};

enum class LinearWeightMode {
    Fp16Cast,
    Bf16Direct,
};

LinearTileMode parse_tile_mode()
{
    const char* raw = std::getenv("EDGE_FM_CUTLASS_LINEAR_TILE");
    if (raw == nullptr || *raw == '\0') {
        return LinearTileMode::Default;
    }
    const std::string value = to_lower(raw);
    if (value == "auto") {
        return LinearTileMode::Auto;
    }
    if (value == "128x256x32" || value == "tile128x256x32") {
        return LinearTileMode::Tile128x256x32;
    }
    if (value == "128x128x32" || value == "tile128x128x32") {
        return LinearTileMode::Tile128x128x32;
    }
    if (value == "256x128x32" || value == "tile256x128x32") {
        return LinearTileMode::Tile256x128x32;
    }
    return LinearTileMode::Default;
}

LinearTileMode parse_tile_mode_value(const std::string& raw)
{
    const std::string value = to_lower(raw);
    if (value == "auto") {
        return LinearTileMode::Auto;
    }
    if (value == "128x256x32" || value == "tile128x256x32") {
        return LinearTileMode::Tile128x256x32;
    }
    if (value == "128x128x32" || value == "tile128x128x32") {
        return LinearTileMode::Tile128x128x32;
    }
    if (value == "256x128x32" || value == "tile256x128x32") {
        return LinearTileMode::Tile256x128x32;
    }
    if (value == "default") {
        return LinearTileMode::Default;
    }
    throw ConfigurationError("CUTLASS prefill linear source-op unknown tile mode: " + raw);
}

LinearInputMode parse_input_mode_value(const std::string& raw)
{
    const std::string value = to_lower(raw);
    if (value == "mixed_bf16" || value == "bf16_mixed" || value == "bf16") {
        return LinearInputMode::MixedBf16;
    }
    if (value == "fp16_cast" || value == "cast" || value == "fp16") {
        return LinearInputMode::Fp16Cast;
    }
    throw ConfigurationError("CUTLASS prefill linear source-op unknown input_mode: " + raw);
}

LinearWeightMode parse_weight_mode_value(const std::string& raw)
{
    const std::string value = to_lower(raw);
    if (value == "bf16_direct" || value == "direct_bf16" || value == "bf16") {
        return LinearWeightMode::Bf16Direct;
    }
    if (value == "fp16_cast" || value == "cast" || value == "fp16") {
        return LinearWeightMode::Fp16Cast;
    }
    throw ConfigurationError("CUTLASS prefill linear source-op unknown weight_mode: " + raw);
}

const char* tile_mode_name(LinearTileMode mode)
{
    switch (mode) {
        case LinearTileMode::Auto:
            return "auto";
        case LinearTileMode::Tile128x256x32:
            return "128x256x32";
        case LinearTileMode::Tile128x128x32:
            return "128x128x32";
        case LinearTileMode::Tile256x128x32:
            return "256x128x32";
        case LinearTileMode::Default:
        default:
            return "default";
    }
}

const char* input_mode_name(LinearInputMode mode)
{
    return mode == LinearInputMode::MixedBf16 ? "mixed_bf16" : "fp16_cast";
}

const char* weight_mode_name(LinearWeightMode mode)
{
    return mode == LinearWeightMode::Bf16Direct ? "bf16_direct" : "fp16_cast";
}

LinearTileMode resolve_tile_mode(
    LinearTileMode requested,
    LinearRole role,
    int in_features,
    int out_features)
{
    if (requested != LinearTileMode::Auto) {
        return requested;
    }
    if (role == LinearRole::Qkv && in_features == 2048 && out_features == 6144) {
        return LinearTileMode::Tile128x256x32;
    }
    if (role == LinearRole::OProj && in_features == 2048 && out_features == 2048) {
        return LinearTileMode::Tile128x256x32;
    }
    return LinearTileMode::Default;
}

bool launch_best_gemm(
    LinearTileMode tile_mode,
    const half* a,
    const half* b_transposed_row_major,
    half* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    if (tile_mode == LinearTileMode::Tile128x256x32) {
        return launch_gemm<CutlassGemmF16AccumTuned<128, 256, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile128x128x32) {
        return launch_gemm<CutlassGemmF16AccumTuned<128, 128, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile256x128x32) {
        return launch_gemm<CutlassGemmF16AccumTuned<256, 128, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    return launch_gemm<CutlassGemmF16Accum<128, 128, 64, 3>>(
        a, b_transposed_row_major, d, m, n, k, stream, error);
}

bool launch_best_gemm_to_bf16(
    LinearTileMode tile_mode,
    const half* a,
    const half* b_transposed_row_major,
    __nv_bfloat16* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    if (tile_mode == LinearTileMode::Tile128x256x32) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<128, 256, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile128x128x32) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<128, 128, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile256x128x32) {
        return launch_gemm_to_bf16<CutlassGemmF16AccumBf16OutTuned<256, 128, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    return launch_gemm_to_bf16<CutlassGemmF16AccumBf16Out<128, 128, 64, 3>>(
        a, b_transposed_row_major, d, m, n, k, stream, error);
}

bool launch_best_mixed_gemm_to_half(
    LinearTileMode tile_mode,
    const __nv_bfloat16* a,
    const half* b_transposed_row_major,
    half* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    if (tile_mode == LinearTileMode::Tile128x256x32) {
        return launch_mixed_gemm_to_half<CutlassGemmBf16AHalfBHalfOutMixedTuned<128, 256, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile128x128x32) {
        return launch_mixed_gemm_to_half<CutlassGemmBf16AHalfBHalfOutMixedTuned<128, 128, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile256x128x32) {
        return launch_mixed_gemm_to_half<CutlassGemmBf16AHalfBHalfOutMixedTuned<256, 128, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    return launch_mixed_gemm_to_half<CutlassGemmBf16AHalfBHalfOutMixed<128, 128, 64, 3>>(
        a, b_transposed_row_major, d, m, n, k, stream, error);
}

bool launch_best_mixed_gemm_to_bf16(
    LinearTileMode tile_mode,
    const __nv_bfloat16* a,
    const half* b_transposed_row_major,
    __nv_bfloat16* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    if (tile_mode == LinearTileMode::Tile128x256x32) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<128, 256, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile128x128x32) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<128, 128, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile256x128x32) {
        return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixedTuned<256, 128, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    return launch_mixed_gemm_to_bf16<CutlassGemmBf16AHalfBBf16OutMixed<128, 128, 64, 3>>(
        a, b_transposed_row_major, d, m, n, k, stream, error);
}

bool launch_best_bf16_weight_gemm_to_half(
    LinearTileMode tile_mode,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b_transposed_row_major,
    half* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    if (tile_mode == LinearTileMode::Tile128x256x32) {
        return launch_bf16_weight_gemm_to_half<CutlassGemmBf16ABf16BHalfOutTuned<128, 256, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile128x128x32) {
        return launch_bf16_weight_gemm_to_half<CutlassGemmBf16ABf16BHalfOutTuned<128, 128, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile256x128x32) {
        return launch_bf16_weight_gemm_to_half<CutlassGemmBf16ABf16BHalfOutTuned<256, 128, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    return launch_bf16_weight_gemm_to_half<CutlassGemmBf16ABf16BHalfOutTuned<128, 128, 64, 64, 64, 64, 3>>(
        a, b_transposed_row_major, d, m, n, k, stream, error);
}

bool launch_best_bf16_weight_gemm_to_bf16(
    LinearTileMode tile_mode,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b_transposed_row_major,
    __nv_bfloat16* d,
    int m,
    int n,
    int k,
    cudaStream_t stream,
    std::string* error)
{
    if (tile_mode == LinearTileMode::Tile128x256x32) {
        return launch_bf16_weight_gemm_to_bf16<CutlassGemmBf16ABf16BBf16OutTuned<128, 256, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile128x128x32) {
        return launch_bf16_weight_gemm_to_bf16<CutlassGemmBf16ABf16BBf16OutTuned<128, 128, 32, 64, 64, 32, 4>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    if (tile_mode == LinearTileMode::Tile256x128x32) {
        return launch_bf16_weight_gemm_to_bf16<CutlassGemmBf16ABf16BBf16OutTuned<256, 128, 32, 64, 64, 32, 3>>(
            a, b_transposed_row_major, d, m, n, k, stream, error);
    }
    return launch_bf16_weight_gemm_to_bf16<CutlassGemmBf16ABf16BBf16OutTuned<128, 128, 64, 64, 64, 64, 3>>(
        a, b_transposed_row_major, d, m, n, k, stream, error);
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

bool launch_half_to_bf16(
    const Tensor& source,
    const Tensor* bias,
    Tensor& destination,
    cudaStream_t stream)
{
    const auto& shape = source.shape();
    if (!is_gpu_dtype_2d(source, DType::Float16) ||
        !is_gpu_dtype_2d(destination, DType::BFloat16) ||
        destination.shape() != shape) {
        return false;
    }
    if (bias != nullptr &&
        (!is_gpu_dtype_1d(*bias, DType::BFloat16) || bias->shape()[0] != shape[1])) {
        return false;
    }
    const int64_t elements64 = shape[0] * shape[1];
    if (elements64 <= 0 || elements64 > std::numeric_limits<int>::max()) {
        return false;
    }
    const int rows = static_cast<int>(shape[0]);
    const int cols = static_cast<int>(shape[1]);
    const int elements = static_cast<int>(elements64);
    const auto* bias_ptr = bias == nullptr
        ? nullptr
        : reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr());
    const auto source_addr = reinterpret_cast<std::uintptr_t>(source.data_ptr());
    const auto destination_addr = reinterpret_cast<std::uintptr_t>(destination.data_ptr());
    const auto bias_addr = bias_ptr == nullptr ? std::uintptr_t{0} : reinterpret_cast<std::uintptr_t>(bias_ptr);
    if ((elements % 2) == 0 && (cols % 2) == 0 &&
        (source_addr % alignof(__half2)) == 0 &&
        (destination_addr % alignof(__nv_bfloat162)) == 0 &&
        (bias_ptr == nullptr || (bias_addr % alignof(__nv_bfloat162)) == 0)) {
        const int pairs = elements / 2;
        half2_to_bf16_kernel<<<ceil_div(pairs, kThreads), kThreads, 0, stream>>>(
            reinterpret_cast<const __half2*>(source.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(bias_ptr),
            reinterpret_cast<__nv_bfloat162*>(destination.data_ptr()),
            pairs,
            cols);
    } else {
        half_to_bf16_kernel<<<ceil_div(elements, kThreads), kThreads, 0, stream>>>(
            reinterpret_cast<const half*>(source.data_ptr()),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(destination.data_ptr()),
            rows,
            cols);
    }
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

struct PrefillLinearSourceOp::Impl {
    explicit Impl(const EngineConfig& config)
        : model_name(config.resolved_model_name())
        , hw_profile(config.resolved_hw_profile())
        , operator_impl_table_path(config.operator_impl_table_path())
    {
        env_enabled = env_flag_enabled("EDGE_FM_PREFILL_CUTLASS_LINEAR");
        env_persistent_weights = env_flag_enabled("EDGE_FM_CUTLASS_LINEAR_PERSISTENT_WEIGHTS");
        env_overlap_casts = env_flag_enabled("EDGE_FM_CUTLASS_LINEAR_OVERLAP_CASTS");
        const char* raw_weight_mode = std::getenv("EDGE_FM_CUTLASS_LINEAR_WEIGHT_MODE");
        if (raw_weight_mode != nullptr && *raw_weight_mode != '\0') {
            env_weight_mode = parse_weight_mode_value(raw_weight_mode);
        }
        env_requested_tile_mode = parse_tile_mode();
        env_min_m = parse_env_i64("EDGE_FM_CUTLASS_LINEAR_MIN_M", 64);
        device_id = config.runtime_device_id();
        if (env_enabled) {
            Logging::instance().log_warn(
                "CUTLASS prefill linear source-op is experimental/default-off; tile={}, input_mode={}, weight_mode={}, persistent_weights={}, overlap_casts={}, min_m={}",
                tile_mode_name(env_requested_tile_mode),
                input_mode_name(env_input_mode),
                weight_mode_name(env_weight_mode),
                env_persistent_weights ? "on" : "off",
                env_overlap_casts ? "on" : "off",
                env_min_m);
        }
    }

    ~Impl()
    {
        if (cast_weight_done != nullptr) {
            cudaEventDestroy(cast_weight_done);
        }
        if (cast_main_ready != nullptr) {
            cudaEventDestroy(cast_main_ready);
        }
        if (cast_stream != nullptr) {
            cudaStreamDestroy(cast_stream);
        }
    }

    struct RuntimeConfig {
        bool enabled = false;
        bool persistent_weights = false;
        bool overlap_casts = false;
        LinearTileMode requested_tile_mode = LinearTileMode::Default;
        LinearInputMode input_mode = LinearInputMode::Fp16Cast;
        LinearWeightMode weight_mode = LinearWeightMode::Fp16Cast;
        int64_t min_m = 64;
        int64_t persistent_min_free_mb = 0;
        bool table_selected = false;
    };

    std::string source_op_shape_sig(int64_t m, int64_t in_features, int64_t out_features) const
    {
        return "m=" + std::to_string(m) +
            "|input=" + std::to_string(static_cast<int>(DType::BFloat16)) +
            "|weight=" + std::to_string(static_cast<int>(DType::BFloat16)) +
            "|output=" + std::to_string(static_cast<int>(DType::BFloat16)) +
            "|in_features=" + std::to_string(in_features) +
            "|out_features=" + std::to_string(out_features);
    }

    RuntimeConfig resolve_runtime_config(
        LinearRole role,
        int64_t m,
        int64_t in_features,
        int64_t out_features)
    {
        const std::string shape_sig = source_op_shape_sig(m, in_features, out_features);
        const std::string cache_key = std::string(linear_layer_role(role)) + "|" + shape_sig;
        auto cache_it = runtime_config_cache.find(cache_key);
        if (cache_it != runtime_config_cache.end()) {
            return cache_it->second;
        }

        RuntimeConfig runtime;
        runtime.enabled = env_enabled;
        runtime.persistent_weights = env_persistent_weights;
        runtime.overlap_casts = env_overlap_casts;
        runtime.requested_tile_mode = env_requested_tile_mode;
        runtime.input_mode = env_input_mode;
        runtime.weight_mode = env_weight_mode;
        runtime.min_m = env_min_m;

        OperatorQuery query;
        query.op_kind = "linear";
        query.layer_role = linear_layer_role(role);
        query.stage = "prefill";
        query.shape_sig = shape_sig;
        auto resolved = OperatorImplTable::instance().resolve(
            model_name,
            hw_profile,
            operator_impl_table_path,
            query);
        if (!resolved.has_value() || resolved->impl_id != "cutlass_prefill_linear_source_op") {
            runtime_config_cache.emplace(cache_key, runtime);
            return runtime;
        }

        runtime.table_selected = true;
        const auto& params = resolved->impl_params;
        runtime.enabled = json_bool_or(params, "enabled", true);
        runtime.persistent_weights = json_bool_or(params, "persistent_weights", runtime.persistent_weights);
        runtime.overlap_casts = json_bool_or(params, "overlap_casts", runtime.overlap_casts);
        runtime.requested_tile_mode = parse_tile_mode_value(
            json_string_or(params, "tile", tile_mode_name(runtime.requested_tile_mode)));
        runtime.input_mode = parse_input_mode_value(
            json_string_or(params, "input_mode", input_mode_name(runtime.input_mode)));
        runtime.weight_mode = parse_weight_mode_value(
            json_string_or(params, "weight_mode", weight_mode_name(runtime.weight_mode)));
        runtime.min_m = json_i64_or(params, "min_m", runtime.min_m);
        runtime.persistent_min_free_mb = json_i64_or(params, "persistent_min_free_mb", runtime.persistent_min_free_mb);
        if (runtime.persistent_min_free_mb < 0) {
            throw ConfigurationError("CUTLASS prefill linear source-op impl_params.persistent_min_free_mb must be >= 0");
        }

        if (logged_table_shapes.insert(cache_key).second) {
            Logging::instance().log_debug(
                "CUTLASS prefill linear source-op selected by operator_impl_table; role={}, tile={}, input_mode={}, weight_mode={}, persistent_weights={}, overlap_casts={}, min_m={}, persistent_min_free_mb={}, shape={}",
                linear_layer_role(role),
                tile_mode_name(runtime.requested_tile_mode),
                input_mode_name(runtime.input_mode),
                weight_mode_name(runtime.weight_mode),
                runtime.persistent_weights ? "on" : "off",
                runtime.overlap_casts ? "on" : "off",
                runtime.min_m,
                runtime.persistent_min_free_mb,
                query.shape_sig);
        }
        runtime_config_cache.emplace(cache_key, runtime);
        return runtime;
    }

    bool try_forward(
        const std::string& role_name_value,
        int32_t layer_id,
        const Tensor& input,
        const Tensor& weight,
        const Tensor* bias,
        Tensor& output,
        cudaStream_t stream)
    {
        LinearRole role = LinearRole::Qkv;
        if (!parse_role(role_name_value, &role)) {
            return false;
        }
        if (!is_gpu_dtype_2d(input, DType::BFloat16) ||
            !is_gpu_dtype_2d(weight, DType::BFloat16) ||
            !is_gpu_dtype_2d(output, DType::BFloat16)) {
            return false;
        }
        if (bias != nullptr && !is_gpu_dtype_1d(*bias, DType::BFloat16)) {
            return false;
        }

        const auto& input_shape = input.shape();
        const auto& weight_shape = weight.shape();
        const auto& output_shape = output.shape();
        const int64_t m64 = input_shape[0];
        const int64_t in_features64 = input_shape[1];
        if (in_features64 <= 0 ||
            weight_shape[1] != in_features64 ||
            output_shape[0] != m64 ||
            output_shape[1] != weight_shape[0]) {
            return false;
        }
        const int64_t out_features64 = weight_shape[0];
        const RuntimeConfig runtime = resolve_runtime_config(role, m64, in_features64, out_features64);
        if (!runtime.table_selected && !role_list_enables(role)) {
            return false;
        }
        if (!runtime.enabled ||
            m64 < runtime.min_m) {
            return false;
        }
        if (bias != nullptr && bias->shape()[0] != out_features64) {
            return false;
        }
        if (m64 > std::numeric_limits<int>::max() ||
            in_features64 > std::numeric_limits<int>::max() ||
            out_features64 > std::numeric_limits<int>::max()) {
            return false;
        }

        const int m = static_cast<int>(m64);
        const int in_features = static_cast<int>(in_features64);
        const int out_features = static_cast<int>(out_features64);
        const std::string shape = shape_key(m64, in_features64, out_features64);
        const std::string role_name_string = role_name(role);
        const LinearTileMode tile_mode =
            resolve_tile_mode(runtime.requested_tile_mode, role, in_features, out_features);

        std::string error;
        if (runtime.weight_mode == LinearWeightMode::Bf16Direct) {
            if (runtime.input_mode != LinearInputMode::MixedBf16) {
                log_once("bf16_weight_requires_mixed_input_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op BF16-direct weight mode requires mixed_bf16 input for {} {}; using fallback",
                         role_name_string,
                         shape);
                return false;
            }
            if (bias == nullptr) {
                if (!launch_best_bf16_weight_gemm_to_bf16(
                        tile_mode,
                        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
                        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                        m,
                        out_features,
                        in_features,
                        stream,
                        &error)) {
                    log_once("bf16_weight_gemm_bf16_failed_" + role_name_string + "_" + shape,
                             "CUTLASS prefill linear source-op BF16-direct GEMM failed for {} {}: {}; using fallback",
                             role_name_string,
                             shape,
                             error);
                    return false;
                }
                return true;
            }

            void* output_half_ptr = StaticBufferManager::get_cache_buf(
                "cutlass_prefill_linear_" + role_name_string + "_output_half_" + shape,
                static_cast<size_t>(m64) * static_cast<size_t>(out_features64) * get_dtype_size(DType::Float16),
                device_id);
            Tensor output_half = Tensor::view(
                output_half_ptr, {m64, out_features64}, DType::Float16, Device::GPU, device_id);
            if (!launch_best_bf16_weight_gemm_to_half(
                    tile_mode,
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
                    reinterpret_cast<half*>(output_half.data_ptr()),
                    m,
                    out_features,
                    in_features,
                    stream,
                    &error)) {
                log_once("bf16_weight_gemm_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op BF16-direct GEMM failed for {} {}: {}; using fallback",
                         role_name_string,
                         shape,
                         error);
                return false;
            }
            if (!launch_half_to_bf16(output_half, bias, output, stream)) {
                log_once("bf16_weight_output_cast_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op failed BF16-direct FP16->BF16 output cast for {} {}; using fallback",
                         role_name_string,
                         shape);
                return false;
            }
            return true;
        }

        Tensor scratch_weight;
        const bool can_overlap_casts =
            runtime.overlap_casts &&
            runtime.weight_mode == LinearWeightMode::Fp16Cast &&
            runtime.input_mode == LinearInputMode::Fp16Cast &&
            !runtime.persistent_weights;
        bool overlapped_weight_cast = false;
        cudaStream_t weight_cast_stream = stream;
        if (can_overlap_casts && ensure_cast_overlap_runtime(role_name_string, shape)) {
            cudaError_t status = cudaEventRecord(cast_main_ready, stream);
            if (status == cudaSuccess) {
                status = cudaStreamWaitEvent(cast_stream, cast_main_ready, 0);
            }
            if (status == cudaSuccess) {
                weight_cast_stream = cast_stream;
                overlapped_weight_cast = true;
            } else {
                log_once("overlap_cast_start_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op failed to start overlapped casts for {} {} (status={}): {}; using same-stream casts",
                         role_name_string,
                         shape,
                         static_cast<int>(status),
                         cudaGetErrorString(status));
            }
        }

        const Tensor* weight_bind = bind_fp16_weight(
            role,
            layer_id,
            weight,
            scratch_weight,
            runtime.persistent_weights,
            runtime.persistent_min_free_mb,
            weight_cast_stream);
        if (weight_bind == nullptr) {
            return false;
        }
        if (overlapped_weight_cast) {
            cudaError_t status = cudaEventRecord(cast_weight_done, cast_stream);
            if (status != cudaSuccess) {
                log_once("overlap_cast_record_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op failed to record overlapped weight cast for {} {} (status={}): {}; using fallback",
                         role_name_string,
                         shape,
                         static_cast<int>(status),
                         cudaGetErrorString(status));
                return false;
            }
        }

        if (runtime.input_mode == LinearInputMode::MixedBf16 && bias == nullptr) {
            if (!launch_best_mixed_gemm_to_bf16(
                    tile_mode,
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    reinterpret_cast<const half*>(weight_bind->data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    m,
                    out_features,
                    in_features,
                    stream,
                    &error)) {
                log_once("mixed_gemm_bf16_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op mixed BF16-input GEMM failed for {} {}: {}; using fallback",
                         role_name_string,
                         shape,
                         error);
                return false;
            }
            return true;
        }

        if (runtime.input_mode == LinearInputMode::MixedBf16 && bias != nullptr) {
            void* output_half_ptr = StaticBufferManager::get_cache_buf(
                "cutlass_prefill_linear_" + role_name_string + "_output_half_" + shape,
                static_cast<size_t>(m64) * static_cast<size_t>(out_features64) * get_dtype_size(DType::Float16),
                device_id);
            Tensor output_half = Tensor::view(
                output_half_ptr, {m64, out_features64}, DType::Float16, Device::GPU, device_id);
            if (!launch_best_mixed_gemm_to_half(
                    tile_mode,
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    reinterpret_cast<const half*>(weight_bind->data_ptr()),
                    reinterpret_cast<half*>(output_half.data_ptr()),
                    m,
                    out_features,
                    in_features,
                    stream,
                    &error)) {
                log_once("mixed_gemm_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op mixed BF16-input GEMM failed for {} {}: {}; using fallback",
                         role_name_string,
                         shape,
                         error);
                return false;
            }

            if (!launch_half_to_bf16(output_half, bias, output, stream)) {
                log_once("mixed_output_cast_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op failed mixed FP16->BF16 output cast for {} {}; using fallback",
                         role_name_string,
                         shape);
                return false;
            }
            return true;
        }

        void* input_half_ptr = StaticBufferManager::get_cache_buf(
            "cutlass_prefill_linear_" + role_name_string + "_input_half_" + shape,
            static_cast<size_t>(m64) * static_cast<size_t>(in_features64) * get_dtype_size(DType::Float16),
            device_id);
        Tensor input_half = Tensor::view(
            input_half_ptr, {m64, in_features64}, DType::Float16, Device::GPU, device_id);

        if (!launch_bf16_to_half(input, input_half, stream)) {
            log_once("input_cast_failed_" + role_name_string + "_" + shape,
                     "CUTLASS prefill linear source-op failed BF16->FP16 input cast for {} {}; using fallback",
                     role_name_string,
                     shape);
            return false;
        }
        if (overlapped_weight_cast) {
            cudaError_t status = cudaStreamWaitEvent(stream, cast_weight_done, 0);
            if (status != cudaSuccess) {
                log_once("overlap_cast_wait_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op failed to wait overlapped weight cast for {} {} (status={}): {}; using fallback",
                         role_name_string,
                         shape,
                         static_cast<int>(status),
                         cudaGetErrorString(status));
                return false;
            }
        }

        if (bias == nullptr) {
            if (!launch_best_gemm_to_bf16(
                    tile_mode,
                    reinterpret_cast<const half*>(input_half.data_ptr()),
                    reinterpret_cast<const half*>(weight_bind->data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    m,
                    out_features,
                    in_features,
                    stream,
                    &error)) {
                log_once("gemm_bf16_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op BF16-output GEMM failed for {} {}: {}; using fallback",
                         role_name_string,
                         shape,
                         error);
                return false;
            }
        } else {
            void* output_half_ptr = StaticBufferManager::get_cache_buf(
                "cutlass_prefill_linear_" + role_name_string + "_output_half_" + shape,
                static_cast<size_t>(m64) * static_cast<size_t>(out_features64) * get_dtype_size(DType::Float16),
                device_id);
            Tensor output_half = Tensor::view(
                output_half_ptr, {m64, out_features64}, DType::Float16, Device::GPU, device_id);
            if (!launch_best_gemm(
                    tile_mode,
                    reinterpret_cast<const half*>(input_half.data_ptr()),
                    reinterpret_cast<const half*>(weight_bind->data_ptr()),
                    reinterpret_cast<half*>(output_half.data_ptr()),
                    m,
                    out_features,
                    in_features,
                    stream,
                    &error)) {
                log_once("gemm_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op GEMM failed for {} {}: {}; using fallback",
                         role_name_string,
                         shape,
                         error);
                return false;
            }

            if (!launch_half_to_bf16(output_half, bias, output, stream)) {
                log_once("output_cast_failed_" + role_name_string + "_" + shape,
                         "CUTLASS prefill linear source-op failed FP16->BF16 output cast for {} {}; using fallback",
                         role_name_string,
                         shape);
                return false;
            }
        }

        return true;
    }

    bool ensure_cast_overlap_runtime(const std::string& role_name_string, const std::string& shape)
    {
        if (cast_stream != nullptr) {
            return true;
        }
        cudaError_t status = cudaStreamCreateWithFlags(&cast_stream, cudaStreamNonBlocking);
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&cast_main_ready, cudaEventDisableTiming);
        }
        if (status == cudaSuccess) {
            status = cudaEventCreateWithFlags(&cast_weight_done, cudaEventDisableTiming);
        }
        if (status != cudaSuccess) {
            log_once("overlap_cast_runtime_failed_" + role_name_string + "_" + shape,
                     "CUTLASS prefill linear source-op failed to create overlapped cast runtime for {} {} (status={}): {}; using same-stream casts",
                     role_name_string,
                     shape,
                     static_cast<int>(status),
                     cudaGetErrorString(status));
            if (cast_weight_done != nullptr) {
                cudaEventDestroy(cast_weight_done);
                cast_weight_done = nullptr;
            }
            if (cast_main_ready != nullptr) {
                cudaEventDestroy(cast_main_ready);
                cast_main_ready = nullptr;
            }
            if (cast_stream != nullptr) {
                cudaStreamDestroy(cast_stream);
                cast_stream = nullptr;
            }
            return false;
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
        LinearRole role,
        int32_t layer_id,
        const Tensor& source,
        Tensor& scratch_tensor,
        bool persistent_weight_mode,
        int64_t persistent_min_free_mb,
        cudaStream_t stream)
    {
        if (persistent_weight_mode) {
            const Tensor* persistent = get_or_create_persistent_fp16_weight(
                role, layer_id, source, persistent_min_free_mb, stream);
            if (persistent != nullptr) {
                return persistent;
            }
        }
        const auto& shape = source.shape();
        const size_t bytes = static_cast<size_t>(shape[0]) *
            static_cast<size_t>(shape[1]) * get_dtype_size(DType::Float16);
        void* ptr = StaticBufferManager::get_cache_buf(
            "cutlass_prefill_linear_" + std::string(role_name(role)) +
                "_fp16_scratch_" + matrix_key(shape),
            bytes,
            device_id);
        scratch_tensor = Tensor::view(ptr, shape, DType::Float16, Device::GPU, device_id);
        if (!launch_bf16_to_half(source, scratch_tensor, stream)) {
            log_once("scratch_weight_cast_failed_" + std::string(role_name(role)) + "_" + matrix_key(shape),
                     "CUTLASS prefill linear source-op failed scratch FP16 {} weight cast for {}; using fallback",
                     role_name(role),
                     matrix_key(shape));
            return nullptr;
        }
        return &scratch_tensor;
    }

    const Tensor* get_or_create_persistent_fp16_weight(
        LinearRole role,
        int32_t layer_id,
        const Tensor& source,
        int64_t persistent_min_free_mb,
        cudaStream_t stream)
    {
        if (persistent_weight_copy_disabled) {
            return nullptr;
        }
        const std::string key = "layer" + std::to_string(layer_id) + "_" + role_name(role);
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
        if (persistent_min_free_mb > 0) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            const cudaError_t mem_status = cudaMemGetInfo(&free_bytes, &total_bytes);
            if (mem_status == cudaSuccess) {
                const size_t reserve_bytes = static_cast<size_t>(persistent_min_free_mb) * 1024ULL * 1024ULL;
                if (free_bytes <= bytes + reserve_bytes) {
                    log_once("persistent_weight_reserve_skip_" + key,
                             "CUTLASS prefill linear source-op skipped persistent FP16 {} weight copy for layer {} to keep {} MB free (need {} bytes, free {} bytes); using scratch weight",
                             role_name(role),
                             layer_id,
                             persistent_min_free_mb,
                             bytes,
                             free_bytes);
                    return nullptr;
                }
            }
        }
        void* data = nullptr;
        const cudaError_t alloc_status = cudaMalloc(&data, bytes);
        if (alloc_status != cudaSuccess) {
            disable_persistent_weight_copies_after_failure(
                std::string("alloc_failed_") + role_name(role), bytes, alloc_status);
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
                std::string("cast_failed_") + role_name(role), bytes, cudaGetLastError());
            return nullptr;
        }

        PersistentWeightCopy copy;
        copy.tensor = std::move(fp16_tensor);
        copy.source_ptr = source.data_ptr();
        auto [inserted, _ok] = persistent_weight_copies.emplace(key, std::move(copy));
        Logging::instance().log_debug(
            "CUTLASS prefill linear source-op created persistent FP16 {} weight copy for layer {} ({} bytes)",
            role_name(role),
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
                 "CUTLASS prefill linear source-op disabled persistent FP16 weight mode after {} ({} bytes, status={}): {}; using fallback",
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
    bool env_overlap_casts = false;
    bool persistent_weight_copy_disabled = false;
    LinearTileMode env_requested_tile_mode = LinearTileMode::Default;
    LinearInputMode env_input_mode = LinearInputMode::Fp16Cast;
    LinearWeightMode env_weight_mode = LinearWeightMode::Fp16Cast;
    int64_t env_min_m = 64;
    int32_t device_id = 0;
    cudaStream_t cast_stream = nullptr;
    cudaEvent_t cast_main_ready = nullptr;
    cudaEvent_t cast_weight_done = nullptr;
    std::unordered_map<std::string, PersistentWeightCopy> persistent_weight_copies;
    std::unordered_map<std::string, RuntimeConfig> runtime_config_cache;
    std::unordered_set<std::string> logged_messages;
    std::unordered_set<std::string> logged_table_shapes;
};

PrefillLinearSourceOp::PrefillLinearSourceOp(const EngineConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
}

PrefillLinearSourceOp::~PrefillLinearSourceOp() = default;

bool PrefillLinearSourceOp::try_forward(
    const std::string& role,
    int32_t layer_id,
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& output,
    cudaStream_t stream)
{
    return impl_->try_forward(role, layer_id, input, weight, bias, output, stream);
}

void PrefillLinearSourceOp::reset_runtime_caches()
{
    impl_->reset_runtime_caches();
}

} // namespace edge_fm
