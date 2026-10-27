#include "vlaforge/backends/tensorrt_precision_plugins.h"

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#if NV_TENSORRT_MAJOR != 8
#error "The precision plugin implements the audited TensorRT 8 IPluginV2DynamicExt ABI"
#endif

namespace {

constexpr char kName[] = "VLAForgeBFloat16Round";
constexpr char kVersion[] = "1";
constexpr std::uint32_t kSerializationTag = 0x56464231u;
using Namespace = std::array<char, 1024>;

bool SetNamespace(Namespace* destination, const char* value) noexcept {
  if (value == nullptr) {
    (*destination)[0] = '\0';
    return true;
  }
  std::size_t length = 0;
  while (length < destination->size() && value[length] != '\0') ++length;
  if (length == destination->size()) return false;
  std::memcpy(destination->data(), value, length + 1);
  return true;
}

__global__ void RoundKernel(const float* input, float* output, std::size_t count) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    std::uint32_t bits = __float_as_uint(input[index]);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
      bits = (bits | 0x00400000u) & 0xffff0000u;
    } else {
      bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    }
    output[index] = __uint_as_float(bits);
  }
}

class RoundPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  const char* getPluginType() const noexcept override { return kName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t initialize() noexcept override { return valid_ ? 0 : 1; }
  void terminate() noexcept override {}
  void destroy() noexcept override { delete this; }
  std::size_t getSerializationSize() const noexcept override {
    return sizeof(kSerializationTag);
  }
  void serialize(void* buffer) const noexcept override {
    if (buffer != nullptr) std::memcpy(buffer, &kSerializationTag, sizeof(kSerializationTag));
  }
  void setPluginNamespace(const char* value) noexcept override {
    valid_ = SetNamespace(&namespace_, value) && valid_;
  }
  const char* getPluginNamespace() const noexcept override { return namespace_.data(); }
  RoundPlugin* clone() const noexcept override { return new (std::nothrow) RoundPlugin(*this); }

  nvinfer1::DimsExprs getOutputDimensions(int32_t output, const nvinfer1::DimsExprs* inputs,
      int32_t count, nvinfer1::IExprBuilder&) noexcept override {
    return output == 0 && count == 1 && inputs != nullptr ? inputs[0] : nvinfer1::DimsExprs{-1, {}};
  }
  nvinfer1::DataType getOutputDataType(int32_t, const nvinfer1::DataType*, int32_t) const noexcept override {
    return nvinfer1::DataType::kFLOAT;
  }
  bool supportsFormatCombination(int32_t position, const nvinfer1::PluginTensorDesc* io,
      int32_t inputs, int32_t outputs) noexcept override {
    return inputs == 1 && outputs == 1 && position >= 0 && position < 2 && io != nullptr
        && io[position].type == nvinfer1::DataType::kFLOAT
        && io[position].format == nvinfer1::TensorFormat::kLINEAR;
  }
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t input_count,
      const nvinfer1::DynamicPluginTensorDesc* outputs, int32_t output_count) noexcept override {
    valid_ = valid_ && input_count == 1 && output_count == 1 && inputs != nullptr && outputs != nullptr;
    if (valid_) {
      valid_ = inputs[0].desc.type == nvinfer1::DataType::kFLOAT
          && outputs[0].desc.type == nvinfer1::DataType::kFLOAT
          && inputs[0].desc.format == nvinfer1::TensorFormat::kLINEAR
          && outputs[0].desc.format == nvinfer1::TensorFormat::kLINEAR
          && inputs[0].desc.dims.nbDims >= 0
          && inputs[0].desc.dims.nbDims == outputs[0].desc.dims.nbDims;
    }
  }
  std::size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int32_t,
      const nvinfer1::PluginTensorDesc*, int32_t) const noexcept override { return 0; }
  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
      const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
      void* const* outputs, void*, cudaStream_t stream) noexcept override {
    if (!valid_ || input_desc == nullptr || output_desc == nullptr || inputs == nullptr
        || outputs == nullptr || input_desc[0].dims.nbDims < 0
        || input_desc[0].type != nvinfer1::DataType::kFLOAT
        || output_desc[0].type != nvinfer1::DataType::kFLOAT
        || input_desc[0].format != nvinfer1::TensorFormat::kLINEAR
        || output_desc[0].format != nvinfer1::TensorFormat::kLINEAR
        || input_desc[0].dims.nbDims != output_desc[0].dims.nbDims) return 1;
    std::size_t count = 1;
    for (int32_t axis = 0; axis < input_desc[0].dims.nbDims; ++axis) {
      const int32_t dimension = input_desc[0].dims.d[axis];
      if (dimension < 0 || dimension != output_desc[0].dims.d[axis]
          || (dimension != 0 && count > std::numeric_limits<std::size_t>::max()
              / static_cast<std::size_t>(dimension))) return 1;
      count *= static_cast<std::size_t>(dimension);
    }
    if (count == 0) return 0;
    if (inputs[0] == nullptr || outputs[0] == nullptr) return 1;
    const auto blocks = static_cast<unsigned int>(std::min<std::size_t>((count - 1) / 256 + 1, 65535));
    RoundKernel<<<blocks, 256, 0, stream>>>(static_cast<const float*>(inputs[0]),
        static_cast<float*>(outputs[0]), count);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
  }
  void attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept override {}
  void detachFromContext() noexcept override {}

 private:
  Namespace namespace_{};
  bool valid_ = true;
};

class RoundCreator final : public nvinfer1::IPluginCreator {
 public:
  const char* getPluginName() const noexcept override { return kName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
  nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection* fields) noexcept override {
    if (!valid_ || (fields != nullptr && fields->nbFields != 0)) return nullptr;
    auto* plugin = new (std::nothrow) RoundPlugin;
    if (plugin != nullptr) plugin->setPluginNamespace(namespace_.data());
    return plugin;
  }
  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* data, std::size_t length) noexcept override {
    std::uint32_t tag = 0;
    if (data == nullptr || length != sizeof(tag)) return nullptr;
    std::memcpy(&tag, data, sizeof(tag));
    return tag == kSerializationTag ? createPlugin(name, nullptr) : nullptr;
  }
  void setPluginNamespace(const char* value) noexcept override {
    valid_ = SetNamespace(&namespace_, value) && valid_;
  }
  const char* getPluginNamespace() const noexcept override { return namespace_.data(); }

 private:
  nvinfer1::PluginFieldCollection fields_{};
  Namespace namespace_{};
  bool valid_ = true;
};

constexpr char kMatMulName[] = "VLAForgeBFloat16MatMul";
constexpr std::uint32_t kMatMulTag = 0x56464d31u;
constexpr std::size_t kBlasWorkspace = 4u << 20u;

struct MatrixLayout {
  nvinfer1::Dims output{};
  int m = 0, n = 0, k = 0, batches = 0;
  std::size_t a_count = 0, b_count = 0, c_count = 0;
  std::size_t b_offset = 0, blas_offset = 0, workspace = 0;
  bool a_affine = false, b_affine = false;
  long long a_stride = 0, b_stride = 0;
};

bool MatrixElements(const nvinfer1::Dims& shape, std::size_t* count) noexcept {
  if (shape.nbDims < 2 || shape.nbDims > nvinfer1::Dims::MAX_DIMS) return false;
  *count = 1;
  for (int axis = 0; axis < shape.nbDims; ++axis) {
    if (shape.d[axis] <= 0 || *count > std::numeric_limits<std::size_t>::max() / 4u
        / static_cast<std::size_t>(shape.d[axis])) return false;
    *count *= static_cast<std::size_t>(shape.d[axis]);
  }
  return true;
}

bool MatMulLayout(const nvinfer1::Dims& a, const nvinfer1::Dims& b,
                 const nvinfer1::Dims* output, MatrixLayout* layout) noexcept {
  if (!MatrixElements(a, &layout->a_count) || !MatrixElements(b, &layout->b_count)) return false;
  layout->m = a.d[a.nbDims - 2];
  layout->k = a.d[a.nbDims - 1];
  layout->n = b.d[b.nbDims - 1];
  if (layout->k != b.d[b.nbDims - 2]) return false;
  auto& result = layout->output;
  result.nbDims = std::max(a.nbDims, b.nbDims);
  std::size_t batches = 1;
  for (int axis = 0; axis < result.nbDims - 2; ++axis) {
    const int ai = axis - (result.nbDims - a.nbDims), bi = axis - (result.nbDims - b.nbDims);
    const int ad = ai < 0 ? 1 : a.d[ai], bd = bi < 0 ? 1 : b.d[bi];
    if (ad != bd && ad != 1 && bd != 1) return false;
    result.d[axis] = std::max(ad, bd);
    if (batches > static_cast<std::size_t>(std::numeric_limits<int>::max()) / result.d[axis]) return false;
    batches *= static_cast<std::size_t>(result.d[axis]);
  }
  result.d[result.nbDims - 2] = layout->m;
  result.d[result.nbDims - 1] = layout->n;
  if (!MatrixElements(result, &layout->c_count)) return false;
  if (output != nullptr) {
    if (output->nbDims != result.nbDims) return false;
    for (int axis = 0; axis < result.nbDims; ++axis) if (output->d[axis] != result.d[axis]) return false;
  }
  layout->batches = static_cast<int>(batches);
  const auto a_matrix = static_cast<std::size_t>(layout->m) * layout->k;
  const auto b_matrix = static_cast<std::size_t>(layout->k) * layout->n;
  const auto a_batches = layout->a_count / a_matrix, b_batches = layout->b_count / b_matrix;
  layout->a_affine = a_batches == 1 || a_batches == batches;
  layout->b_affine = b_batches == 1 || b_batches == batches;
  layout->a_stride = a_batches == 1 ? 0 : static_cast<long long>(a_matrix);
  layout->b_stride = b_batches == 1 ? 0 : static_cast<long long>(b_matrix);
  layout->b_offset = (layout->a_count * 2u + 255u) & ~std::size_t{255u};
  const auto b_bytes = (layout->b_count * 2u + 255u) & ~std::size_t{255u};
  if (layout->b_offset > std::numeric_limits<std::size_t>::max() - b_bytes) return false;
  layout->blas_offset = layout->b_offset + b_bytes;
  if (layout->blas_offset > std::numeric_limits<std::size_t>::max() - kBlasWorkspace) return false;
  layout->workspace = layout->blas_offset + kBlasWorkspace;
  return true;
}

std::size_t MatrixBatchOffset(const nvinfer1::Dims& shape, const MatrixLayout& layout,
                             int batch) noexcept {
  std::size_t offset = 0, stride = 1;
  for (int axis = layout.output.nbDims - 3; axis >= 0; --axis) {
    const auto coordinate = batch % layout.output.d[axis];
    batch /= layout.output.d[axis];
    const int source = axis - (layout.output.nbDims - shape.nbDims);
    if (source >= 0) {
      if (shape.d[source] != 1) offset += static_cast<std::size_t>(coordinate) * stride;
      stride *= static_cast<std::size_t>(shape.d[source]);
    }
  }
  return offset * static_cast<std::size_t>(shape.d[shape.nbDims - 2]) * shape.d[shape.nbDims - 1];
}

__global__ void PackBFloat16(const float* input, __nv_bfloat16* output, std::size_t count) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = __float2bfloat16_rn(input[index]);
  }
}

bool FloatLinear(const nvinfer1::PluginTensorDesc& value) noexcept {
  return value.type == nvinfer1::DataType::kFLOAT && value.format == nvinfer1::TensorFormat::kLINEAR;
}

class MatMulPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  ~MatMulPlugin() override { Release(); }
  const char* getPluginType() const noexcept override { return kMatMulName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t initialize() noexcept override { return valid_ && EnsureHandle() ? 0 : 1; }
  void terminate() noexcept override { Release(); }
  void destroy() noexcept override { delete this; }
  std::size_t getSerializationSize() const noexcept override { return sizeof(kMatMulTag); }
  void serialize(void* buffer) const noexcept override {
    if (buffer != nullptr) std::memcpy(buffer, &kMatMulTag, sizeof(kMatMulTag));
  }
  void setPluginNamespace(const char* value) noexcept override { valid_ = SetNamespace(&namespace_, value) && valid_; }
  const char* getPluginNamespace() const noexcept override { return namespace_.data(); }
  MatMulPlugin* clone() const noexcept override {
    auto* result = new (std::nothrow) MatMulPlugin;
    if (result != nullptr) { result->namespace_ = namespace_; result->valid_ = valid_; }
    return result;
  }
  nvinfer1::DataType getOutputDataType(int32_t, const nvinfer1::DataType*, int32_t) const noexcept override {
    return nvinfer1::DataType::kFLOAT;
  }
  nvinfer1::DimsExprs getOutputDimensions(int32_t output, const nvinfer1::DimsExprs* inputs,
      int32_t count, nvinfer1::IExprBuilder& builder) noexcept override {
    if (output != 0 || count != 2 || inputs == nullptr) return {-1, {}};
    nvinfer1::Dims dimensions[2]{};
    for (int slot = 0; slot < 2; ++slot) {
      dimensions[slot].nbDims = inputs[slot].nbDims;
      if (inputs[slot].nbDims < 2 || inputs[slot].nbDims > nvinfer1::Dims::MAX_DIMS) return {-1, {}};
      for (int axis = 0; axis < inputs[slot].nbDims; ++axis) {
        const auto* expression = inputs[slot].d[axis];
        if (expression == nullptr || !expression->isConstant() || expression->getConstantValue() <= 0
            || expression->getConstantValue() > std::numeric_limits<int32_t>::max()) return {-1, {}};
        dimensions[slot].d[axis] = static_cast<int32_t>(expression->getConstantValue());
      }
    }
    MatrixLayout layout;
    if (!MatMulLayout(dimensions[0], dimensions[1], nullptr, &layout)) return {-1, {}};
    nvinfer1::DimsExprs result{};
    result.nbDims = layout.output.nbDims;
    for (int axis = 0; axis < result.nbDims; ++axis) result.d[axis] = builder.constant(layout.output.d[axis]);
    return result;
  }
  bool supportsFormatCombination(int32_t position, const nvinfer1::PluginTensorDesc* io,
      int32_t inputs, int32_t outputs) noexcept override {
    return inputs == 2 && outputs == 1 && io != nullptr && position >= 0 && position < 3 && FloatLinear(io[position]);
  }
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t count,
      const nvinfer1::DynamicPluginTensorDesc* outputs, int32_t output_count) noexcept override {
    MatrixLayout layout;
    valid_ = valid_ && count == 2 && output_count == 1 && inputs != nullptr && outputs != nullptr
        && FloatLinear(inputs[0].desc) && FloatLinear(inputs[1].desc) && FloatLinear(outputs[0].desc)
        && MatMulLayout(inputs[0].desc.dims, inputs[1].desc.dims, &outputs[0].desc.dims, &layout);
  }
  std::size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t count,
      const nvinfer1::PluginTensorDesc* outputs, int32_t output_count) const noexcept override {
    MatrixLayout layout;
    return count == 2 && output_count == 1 && inputs != nullptr && outputs != nullptr
        && FloatLinear(inputs[0]) && FloatLinear(inputs[1]) && FloatLinear(outputs[0])
        && MatMulLayout(inputs[0].dims, inputs[1].dims, &outputs[0].dims, &layout) ? layout.workspace : 0;
  }
  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
      const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
      void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
    MatrixLayout layout;
    int device = -1;
    if (!valid_ || handle_ == nullptr || input_desc == nullptr || output_desc == nullptr || inputs == nullptr
        || outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr || workspace == nullptr
        || reinterpret_cast<std::uintptr_t>(workspace) % 256u != 0u
        || !FloatLinear(input_desc[0]) || !FloatLinear(input_desc[1]) || !FloatLinear(output_desc[0])
        || !MatMulLayout(input_desc[0].dims, input_desc[1].dims, &output_desc[0].dims, &layout)
        || cudaGetDevice(&device) != cudaSuccess || device != device_) return 1;
    auto* a = static_cast<__nv_bfloat16*>(workspace);
    auto* b = reinterpret_cast<__nv_bfloat16*>(static_cast<char*>(workspace) + layout.b_offset);
    auto blocks = [](std::size_t count) { return static_cast<unsigned int>(std::min<std::size_t>((count - 1) / 256 + 1, 65535)); };
    PackBFloat16<<<blocks(layout.a_count), 256, 0, stream>>>(static_cast<const float*>(inputs[0]), a, layout.a_count);
    PackBFloat16<<<blocks(layout.b_count), 256, 0, stream>>>(static_cast<const float*>(inputs[1]), b, layout.b_count);
    if (cudaPeekAtLastError() != cudaSuccess || cublasSetStream(handle_, stream) != CUBLAS_STATUS_SUCCESS
        || cublasSetWorkspace(handle_, static_cast<char*>(workspace) + layout.blas_offset, kBlasWorkspace) != CUBLAS_STATUS_SUCCESS) return 1;
    const float alpha = 1, beta = 0;
    auto* result = static_cast<float*>(outputs[0]);
    const auto c_stride = static_cast<long long>(layout.m) * layout.n;
    // Row-major A*B is column-major transpose(B)*transpose(A), with no data transpose.
    if (layout.a_affine && layout.b_affine) {
      if (cublasGemmStridedBatchedEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, layout.n, layout.m, layout.k,
          &alpha, b, CUDA_R_16BF, layout.n, layout.b_stride, a, CUDA_R_16BF, layout.k, layout.a_stride,
          &beta, result, CUDA_R_32F, layout.n, c_stride, layout.batches,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return 1;
    } else {
      for (int batch = 0; batch < layout.batches; ++batch) {
        if (cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, layout.n, layout.m, layout.k, &alpha,
            b + MatrixBatchOffset(input_desc[1].dims, layout, batch), CUDA_R_16BF, layout.n,
            a + MatrixBatchOffset(input_desc[0].dims, layout, batch), CUDA_R_16BF, layout.k,
            &beta, result + static_cast<std::size_t>(batch) * c_stride, CUDA_R_32F, layout.n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) return 1;
      }
    }
    RoundKernel<<<blocks(layout.c_count), 256, 0, stream>>>(result, result, layout.c_count);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
  }
  void attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept override {
    valid_ = valid_ && EnsureHandle();
  }
  void detachFromContext() noexcept override { Release(); }

 private:
  bool EnsureHandle() noexcept {
    if (handle_ != nullptr) return true;
    if (cudaGetDevice(&device_) != cudaSuccess || cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS
        || cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS
        || cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) {
      Release();
      return false;
    }
    return true;
  }
  void Release() noexcept {
    if (handle_ != nullptr) { (void)cublasDestroy(handle_); handle_ = nullptr; }
    device_ = -1;
  }
  Namespace namespace_{};
  bool valid_ = true;
  cublasHandle_t handle_ = nullptr;
  int device_ = -1;
};

class MatMulCreator final : public nvinfer1::IPluginCreator {
 public:
  const char* getPluginName() const noexcept override { return kMatMulName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
  nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection* fields) noexcept override {
    if (!valid_ || (fields != nullptr && fields->nbFields != 0)) return nullptr;
    auto* plugin = new (std::nothrow) MatMulPlugin;
    if (plugin != nullptr) plugin->setPluginNamespace(namespace_.data());
    return plugin;
  }
  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* data, std::size_t length) noexcept override {
    std::uint32_t tag = 0;
    if (data == nullptr || length != sizeof(tag)) return nullptr;
    std::memcpy(&tag, data, sizeof(tag));
    return tag == kMatMulTag ? createPlugin(name, nullptr) : nullptr;
  }
  void setPluginNamespace(const char* value) noexcept override { valid_ = SetNamespace(&namespace_, value) && valid_; }
  const char* getPluginNamespace() const noexcept override { return namespace_.data(); }

 private:
  nvinfer1::PluginFieldCollection fields_{};
  Namespace namespace_{};
  bool valid_ = true;
};

constexpr char kScalarSinCosName[] = "VLAForgeScalarSinCosF64";
constexpr std::uint32_t kScalarSinCosTag = 0x56465331u;
constexpr std::size_t kMaxScalarFrequencies = 16384;
constexpr std::size_t kScalarHeaderBytes = 16;
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559
    && std::numeric_limits<double>::digits == 53, "IEEE754 binary64 is required");

std::uint32_t ReadScalarWord(const unsigned char* data) noexcept {
  return static_cast<std::uint32_t>(data[0])
      | (static_cast<std::uint32_t>(data[1]) << 8u)
      | (static_cast<std::uint32_t>(data[2]) << 16u)
      | (static_cast<std::uint32_t>(data[3]) << 24u);
}

void WriteScalarWord(unsigned char* data, std::uint32_t word) noexcept {
  for (unsigned int byte = 0; byte < 4; ++byte) data[byte] = static_cast<unsigned char>(word >> (byte * 8u));
}

double ScalarFrequency(std::uint32_t low, std::uint32_t high) noexcept {
  const std::uint64_t bits = (static_cast<std::uint64_t>(high) << 32u) | low;
  double value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint64_t ScalarChecksum(const std::vector<double>& frequencies) noexcept {
  std::uint64_t hash = 14695981039346656037ull;
  auto word = [&hash](std::uint32_t value) {
    for (unsigned int byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<unsigned char>(value >> (byte * 8u));
      hash *= 1099511628211ull;
    }
  };
  word(kScalarSinCosTag);
  word(static_cast<std::uint32_t>(frequencies.size()));
  for (double frequency : frequencies) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &frequency, sizeof(bits));
    word(static_cast<std::uint32_t>(bits));
    word(static_cast<std::uint32_t>(bits >> 32u));
  }
  return hash;
}

bool ScalarShape(const nvinfer1::Dims& shape) noexcept {
  if (shape.nbDims < 0 || shape.nbDims > nvinfer1::Dims::MAX_DIMS) return false;
  for (int axis = 0; axis < shape.nbDims; ++axis) if (shape.d[axis] != 1) return false;
  return true;
}

bool SinCosShape(const nvinfer1::Dims& shape, std::size_t count) noexcept {
  return shape.nbDims == 2 && shape.d[0] == 1 && shape.d[1] == static_cast<int32_t>(2 * count);
}

__global__ void ScalarSinCosKernel(const float* time, const double* frequencies,
                                 float* output, std::size_t count) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const double angle = __dmul_rn(static_cast<double>(time[0]), frequencies[index]);
    output[index] = __double2float_rn(::sin(angle));
    output[count + index] = __double2float_rn(::cos(angle));
  }
}

class ScalarSinCosPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  explicit ScalarSinCosPlugin(std::vector<double> frequencies) : frequencies_(std::move(frequencies)) {}
  ~ScalarSinCosPlugin() override { Release(); }
  const char* getPluginType() const noexcept override { return kScalarSinCosName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t initialize() noexcept override { return valid_ && EnsureFrequencies() ? 0 : 1; }
  void terminate() noexcept override { Release(); }
  void destroy() noexcept override { delete this; }
  std::size_t getSerializationSize() const noexcept override {
    return kScalarHeaderBytes + frequencies_.size() * sizeof(double);
  }
  void serialize(void* buffer) const noexcept override {
    if (buffer == nullptr) return;
    auto* bytes = static_cast<unsigned char*>(buffer);
    const auto checksum = ScalarChecksum(frequencies_);
    WriteScalarWord(bytes, kScalarSinCosTag);
    WriteScalarWord(bytes + 4, static_cast<std::uint32_t>(frequencies_.size()));
    WriteScalarWord(bytes + 8, static_cast<std::uint32_t>(checksum));
    WriteScalarWord(bytes + 12, static_cast<std::uint32_t>(checksum >> 32u));
    for (std::size_t index = 0; index < frequencies_.size(); ++index) {
      std::uint64_t bits = 0;
      std::memcpy(&bits, &frequencies_[index], sizeof(bits));
      WriteScalarWord(bytes + kScalarHeaderBytes + index * 8, static_cast<std::uint32_t>(bits));
      WriteScalarWord(bytes + kScalarHeaderBytes + index * 8 + 4, static_cast<std::uint32_t>(bits >> 32u));
    }
  }
  void setPluginNamespace(const char* value) noexcept override {
    valid_ = SetNamespace(&namespace_, value) && valid_;
  }
  const char* getPluginNamespace() const noexcept override { return namespace_.data(); }
  ScalarSinCosPlugin* clone() const noexcept override {
    try {
      auto* result = new (std::nothrow) ScalarSinCosPlugin(frequencies_);
      if (result != nullptr) { result->namespace_ = namespace_; result->valid_ = valid_; }
      return result;
    } catch (...) { return nullptr; }
  }
  nvinfer1::DataType getOutputDataType(int32_t, const nvinfer1::DataType*, int32_t) const noexcept override {
    return nvinfer1::DataType::kFLOAT;
  }
  nvinfer1::DimsExprs getOutputDimensions(int32_t output, const nvinfer1::DimsExprs* inputs,
      int32_t count, nvinfer1::IExprBuilder& builder) noexcept override {
    if (!valid_ || output != 0 || count != 1 || inputs == nullptr || inputs[0].nbDims < 0
        || inputs[0].nbDims > nvinfer1::Dims::MAX_DIMS) return {-1, {}};
    for (int axis = 0; axis < inputs[0].nbDims; ++axis) {
      const auto* dimension = inputs[0].d[axis];
      if (dimension == nullptr || !dimension->isConstant() || dimension->getConstantValue() != 1) return {-1, {}};
    }
    nvinfer1::DimsExprs result{};
    result.nbDims = 2;
    result.d[0] = builder.constant(1);
    result.d[1] = builder.constant(static_cast<int32_t>(2 * frequencies_.size()));
    return result.d[0] != nullptr && result.d[1] != nullptr ? result : nvinfer1::DimsExprs{-1, {}};
  }
  bool supportsFormatCombination(int32_t position, const nvinfer1::PluginTensorDesc* io,
      int32_t inputs, int32_t outputs) noexcept override {
    if (!valid_ || inputs != 1 || outputs != 1 || position < 0 || position > 1 || io == nullptr) return false;
    for (int32_t slot = 0; slot <= position; ++slot) {
      if (!FloatLinear(io[slot]) || (slot == 0 ? !ScalarShape(io[slot].dims)
          : !SinCosShape(io[slot].dims, frequencies_.size()))) return false;
    }
    return true;
  }
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t count,
      const nvinfer1::DynamicPluginTensorDesc* outputs, int32_t output_count) noexcept override {
    valid_ = valid_ && count == 1 && output_count == 1 && inputs != nullptr && outputs != nullptr
        && FloatLinear(inputs[0].desc) && FloatLinear(outputs[0].desc)
        && ScalarShape(inputs[0].desc.dims) && ScalarShape(inputs[0].min) && ScalarShape(inputs[0].max)
        && inputs[0].min.nbDims == inputs[0].desc.dims.nbDims
        && inputs[0].max.nbDims == inputs[0].desc.dims.nbDims
        && SinCosShape(outputs[0].desc.dims, frequencies_.size())
        && SinCosShape(outputs[0].min, frequencies_.size()) && SinCosShape(outputs[0].max, frequencies_.size());
  }
  std::size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int32_t,
      const nvinfer1::PluginTensorDesc*, int32_t) const noexcept override { return 0; }
  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
      const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
      void* const* outputs, void*, cudaStream_t stream) noexcept override {
    int device = -1;
    if (!valid_ || device_frequencies_ == nullptr || input_desc == nullptr || output_desc == nullptr
        || inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr
        || inputs[0] == outputs[0] || !FloatLinear(input_desc[0]) || !FloatLinear(output_desc[0])
        || !ScalarShape(input_desc[0].dims) || !SinCosShape(output_desc[0].dims, frequencies_.size())
        || cudaGetDevice(&device) != cudaSuccess || device != device_) return 1;
    const auto blocks = static_cast<unsigned int>((frequencies_.size() + 255u) / 256u);
    ScalarSinCosKernel<<<blocks, 256, 0, stream>>>(static_cast<const float*>(inputs[0]),
        device_frequencies_, static_cast<float*>(outputs[0]), frequencies_.size());
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
  }
  void attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept override {
    valid_ = valid_ && EnsureFrequencies();
  }
  void detachFromContext() noexcept override { Release(); }

 private:
  bool EnsureFrequencies() noexcept {
    int device = -1;
    if (!valid_ || cudaGetDevice(&device) != cudaSuccess) return false;
    if (device_frequencies_ != nullptr) return device == device_;
    device_ = device;
    if (cudaMalloc(reinterpret_cast<void**>(&device_frequencies_), frequencies_.size() * sizeof(double)) != cudaSuccess
        || cudaMemcpy(device_frequencies_, frequencies_.data(), frequencies_.size() * sizeof(double),
            cudaMemcpyHostToDevice) != cudaSuccess
        // H2D copy can return before device completion; nonblocking consumers need an explicit fence.
        || cudaStreamSynchronize(nullptr) != cudaSuccess) {
      valid_ = false;
      Release();
      return false;
    }
    return true;
  }
  void Release() noexcept {
    if (device_frequencies_ == nullptr) return;
    auto* memory = device_frequencies_;
    device_frequencies_ = nullptr;
    int device = -1;
    // Captured graph owners must retire graphs before the plugin. On a CUDA error or wrong
    // device, completion is unproven: poison this instance and leak its allocation until exit.
    if (cudaGetDevice(&device) != cudaSuccess || device != device_ || cudaDeviceSynchronize() != cudaSuccess) {
      valid_ = false;
    } else if (cudaFree(memory) != cudaSuccess) {
      valid_ = false;
    }
    device_ = -1;
  }
  const std::vector<double> frequencies_;
  Namespace namespace_{};
  bool valid_ = true;
  double* device_frequencies_ = nullptr;
  int device_ = -1;
};

class ScalarSinCosCreator final : public nvinfer1::IPluginCreator {
 public:
  const char* getPluginName() const noexcept override { return kScalarSinCosName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
  nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection* fields) noexcept override {
    if (!valid_ || fields == nullptr || fields->nbFields != 1 || fields->fields == nullptr) return nullptr;
    const auto& field = fields->fields[0];
    if (field.name == nullptr || std::strcmp(field.name, "frequency_words") != 0
        || field.type != nvinfer1::PluginFieldType::kINT32 || field.data == nullptr
        || field.length < 2 || field.length > static_cast<int32_t>(2 * kMaxScalarFrequencies)
        || field.length % 2 != 0) return nullptr;
    try {
      std::vector<double> frequencies;
      frequencies.reserve(static_cast<std::size_t>(field.length) / 2);
      const auto* bytes = static_cast<const unsigned char*>(field.data);
      for (int32_t index = 0; index < field.length; index += 2) {
        // Signed INT32 fields carry bits. memcpy also accepts unaligned parser storage.
        std::uint32_t low = 0, high = 0;
        std::memcpy(&low, bytes + static_cast<std::size_t>(index) * 4, sizeof(low));
        std::memcpy(&high, bytes + static_cast<std::size_t>(index + 1) * 4, sizeof(high));
        const double frequency = ScalarFrequency(low, high);
        if (!std::isfinite(frequency)) return nullptr;
        frequencies.push_back(frequency);
      }
      return MakePlugin(std::move(frequencies));
    } catch (...) { return nullptr; }
  }
  nvinfer1::IPluginV2* deserializePlugin(const char*, const void* data, std::size_t length) noexcept override {
    if (!valid_ || data == nullptr || length < kScalarHeaderBytes) return nullptr;
    const auto* bytes = static_cast<const unsigned char*>(data);
    const auto count = ReadScalarWord(bytes + 4);
    if (ReadScalarWord(bytes) != kScalarSinCosTag || count == 0 || count > kMaxScalarFrequencies
        || length != kScalarHeaderBytes + static_cast<std::size_t>(count) * 8) return nullptr;
    try {
      std::vector<double> frequencies;
      frequencies.reserve(count);
      for (std::size_t index = 0; index < count; ++index) {
        const auto* value = bytes + kScalarHeaderBytes + index * 8;
        const double frequency = ScalarFrequency(ReadScalarWord(value), ReadScalarWord(value + 4));
        if (!std::isfinite(frequency)) return nullptr;
        frequencies.push_back(frequency);
      }
      const std::uint64_t checksum = (static_cast<std::uint64_t>(ReadScalarWord(bytes + 12)) << 32u)
          | ReadScalarWord(bytes + 8);
      return checksum == ScalarChecksum(frequencies) ? MakePlugin(std::move(frequencies)) : nullptr;
    } catch (...) { return nullptr; }
  }
  void setPluginNamespace(const char* value) noexcept override {
    valid_ = SetNamespace(&namespace_, value) && valid_;
  }
  const char* getPluginNamespace() const noexcept override { return namespace_.data(); }

 private:
  nvinfer1::IPluginV2* MakePlugin(std::vector<double> frequencies) noexcept {
    auto* plugin = new (std::nothrow) ScalarSinCosPlugin(std::move(frequencies));
    if (plugin != nullptr) plugin->setPluginNamespace(namespace_.data());
    return plugin;
  }
  const nvinfer1::PluginField frequency_field_{"frequency_words", nullptr, nvinfer1::PluginFieldType::kINT32, 0};
  const nvinfer1::PluginFieldCollection fields_{1, &frequency_field_};
  Namespace namespace_{};
  bool valid_ = true;
};

}  // namespace

extern "C" int vlaforge_register_tensorrt_precision_plugins(void) {
  static RoundCreator creator;
  static const bool registered = getPluginRegistry()->registerCreator(creator, "");
  static MatMulCreator matmul_creator;
  static const bool matmul_registered = getPluginRegistry()->registerCreator(matmul_creator, "");
  static ScalarSinCosCreator sincos_creator;
  static const bool sincos_registered = getPluginRegistry()->registerCreator(sincos_creator, "");
  return registered && matmul_registered && sincos_registered ? 1 : 0;
}

namespace {
// trtexec loads this DSO directly; native Sessions also call registration explicitly.
const int kAutoRegistered = vlaforge_register_tensorrt_precision_plugins();
}  // namespace
