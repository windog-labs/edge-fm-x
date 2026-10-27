#include "vlaforge/backends/tensorrt_precision_plugins.h"

#include <NvInfer.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void Cuda(cudaError_t status) { Require(status == cudaSuccess, cudaGetErrorString(status)); }
void Destroy(nvinfer1::IPluginV2* plugin) { if (plugin != nullptr) plugin->destroy(); }
using Plugin = std::unique_ptr<nvinfer1::IPluginV2, decltype(&Destroy)>;

struct DeviceMemory {
  void* value = nullptr;
  ~DeviceMemory() { if (value) (void)cudaFree(value); }
};
struct Stream {
  cudaStream_t value = nullptr;
  ~Stream() { if (value) (void)cudaStreamDestroy(value); }
};
struct Graph {
  cudaGraph_t value = nullptr;
  cudaGraphExec_t executable = nullptr;
  ~Graph() {
    if (executable) (void)cudaGraphExecDestroy(executable);
    if (value) (void)cudaGraphDestroy(value);
  }
};

struct ScalarDeviceMemory {
  void* value = nullptr;
  ~ScalarDeviceMemory() {
    if (value && cudaDeviceSynchronize() == cudaSuccess) (void)cudaFree(value);
  }
};

std::size_t Elements(const std::vector<int>& shape) {
  std::size_t size = 1;
  for (auto dimension : shape) size *= dimension;
  return size;
}

std::size_t BatchOffset(const std::vector<int>& shape, const std::vector<int>& output, std::size_t batch) {
  std::size_t offset = 0, stride = 1;
  for (int axis = static_cast<int>(output.size()) - 3; axis >= 0; --axis) {
    const auto coordinate = batch % output[axis];
    batch /= output[axis];
    const int source = axis - static_cast<int>(output.size() - shape.size());
    if (source >= 0) {
      if (shape[source] != 1) offset += coordinate * stride;
      stride *= shape[source];
    }
  }
  return offset * shape[shape.size() - 2] * shape.back();
}

float BFloat(float input) {
  return __bfloat162float(__float2bfloat16_rn(input));
}

void TestMatMul() {
  auto* creator = getPluginRegistry()->getPluginCreator("VLAForgeBFloat16MatMul", "1", "");
  Require(creator != nullptr, "BF16 MatMul creator missing");
  Plugin original(creator->createPlugin("mm", nullptr), Destroy);
  Require(original != nullptr, "BF16 MatMul creation failed");
  std::vector<char> serialized(original->getSerializationSize());
  original->serialize(serialized.data());
  Require(creator->deserializePlugin("truncated", serialized.data(), serialized.size() - 1) == nullptr,
          "truncated MatMul serialization accepted");
  Plugin restored(creator->deserializePlugin("restored", serialized.data(), serialized.size()), Destroy);
  Require(restored != nullptr, "MatMul deserialize failed");
  serialized[0] ^= 1;
  Require(creator->deserializePlugin("corrupt", serialized.data(), serialized.size()) == nullptr,
          "corrupt MatMul serialization accepted");
  const std::vector<std::array<std::vector<int>, 2>> cases = {
      {{{32, 64}, {64, 32}}}, {{{3, 5, 8}, {8, 7}}},
      {{{2, 1, 3, 8}, {1, 4, 8, 5}}}, {{{2, 3, 4, 8}, {2, 3, 8, 5}}},
      {{{1, 1, 1, 1, 2, 1, 2, 4}, {1, 1, 1, 1, 1, 3, 4, 3}}},
      {{{4, 8}, {2, 8, 5}}}};
  for (const auto& shapes : cases) {
    const auto &a_shape = shapes[0], &b_shape = shapes[1];
    const int rank = static_cast<int>(std::max(a_shape.size(), b_shape.size()));
    std::vector<int> out_shape(rank, 1);
    for (int axis = 0; axis < rank - 2; ++axis) {
      const int ai = axis - (rank - static_cast<int>(a_shape.size()));
      const int bi = axis - (rank - static_cast<int>(b_shape.size()));
      out_shape[axis] = std::max(ai < 0 ? 1 : a_shape[ai], bi < 0 ? 1 : b_shape[bi]);
    }
    const int m = a_shape[a_shape.size() - 2], k = a_shape.back(), n = b_shape.back();
    out_shape[rank - 2] = m; out_shape[rank - 1] = n;
    std::vector<float> a(Elements(a_shape)), b(Elements(b_shape)), expected(Elements(out_shape));
    std::uint32_t state = 9137;
    for (auto* values : {&a, &b}) for (auto& value : *values) {
      state = state * 1664525u + 1013904223u;
      value = BFloat(static_cast<float>(static_cast<int>((state >> 8u) % 513u) - 256) / 128.0F);
    }
    for (std::size_t batch = 0; batch < expected.size() / (m * n); ++batch) {
      const auto ai = BatchOffset(a_shape, out_shape, batch), bi = BatchOffset(b_shape, out_shape, batch);
      for (int row = 0; row < m; ++row) for (int column = 0; column < n; ++column) {
        double sum = 0;
        for (int inner = 0; inner < k; ++inner)
          sum += static_cast<double>(a[ai + row * k + inner]) * b[bi + inner * n + column];
        expected[batch * m * n + row * n + column] = BFloat(static_cast<float>(sum));
      }
    }
    Plugin instance(restored->clone(), Destroy);
    Require(instance != nullptr, "MatMul clone failed");
    auto* plugin = static_cast<nvinfer1::IPluginV2DynamicExt*>(instance.get());
    Require(plugin->initialize() == 0, "MatMul initialization failed");
    nvinfer1::PluginTensorDesc io[3]{};
    const std::vector<int> all_shapes[] = {a_shape, b_shape, out_shape};
    for (int slot = 0; slot < 3; ++slot) {
      io[slot].dims.nbDims = static_cast<int>(all_shapes[slot].size());
      for (int axis = 0; axis < io[slot].dims.nbDims; ++axis) io[slot].dims.d[axis] = all_shapes[slot][axis];
      io[slot].type = nvinfer1::DataType::kFLOAT;
      io[slot].format = nvinfer1::TensorFormat::kLINEAR;
      Require(plugin->supportsFormatCombination(slot, io, 2, 1), "MatMul FLOAT carrier rejected");
    }
    Require(!plugin->supportsFormatCombination(3, io, 2, 1), "MatMul invalid position accepted");
    io[0].type = nvinfer1::DataType::kHALF;
    Require(!plugin->supportsFormatCombination(0, io, 2, 1), "MatMul HALF carrier accepted");
    io[0].type = nvinfer1::DataType::kFLOAT;
    const auto workspace_bytes = plugin->getWorkspaceSize(io, 2, io + 2, 1);
    Require(workspace_bytes >= (4u << 20u), "MatMul workspace is not bounded/declarative");
    DeviceMemory da, db, dc, workspace;
    Stream stream;
    Cuda(cudaStreamCreate(&stream.value));
    Cuda(cudaMalloc(&da.value, a.size() * 4u)); Cuda(cudaMalloc(&db.value, b.size() * 4u));
    Cuda(cudaMalloc(&dc.value, expected.size() * 4u)); Cuda(cudaMalloc(&workspace.value, workspace_bytes));
    Cuda(cudaMemcpy(da.value, a.data(), a.size() * 4u, cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(db.value, b.data(), b.size() * 4u, cudaMemcpyHostToDevice));
    const void* inputs[] = {da.value, db.value};
    void* outputs[] = {dc.value};
    Require(plugin->enqueue(io, io + 2, inputs, outputs, nullptr, stream.value) != 0, "MatMul missing workspace accepted");
    Require(plugin->enqueue(io, io + 2, inputs, outputs, workspace.value, stream.value) == 0, "MatMul enqueue failed");
    Cuda(cudaStreamSynchronize(stream.value));
    std::vector<float> actual(expected.size());
    Cuda(cudaMemcpy(actual.data(), dc.value, actual.size() * 4u, cudaMemcpyDeviceToHost));
    Require(std::memcmp(actual.data(), expected.data(), actual.size() * 4u) == 0, "BF16 matrix values differ");
    Graph graph;
    Cuda(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    Require(plugin->enqueue(io, io + 2, inputs, outputs, workspace.value, stream.value) == 0, "MatMul capture enqueue failed");
    Cuda(cudaStreamEndCapture(stream.value, &graph.value));
    Cuda(cudaGraphInstantiate(&graph.executable, graph.value, nullptr, nullptr, 0));
    Cuda(cudaMemset(dc.value, 0, actual.size() * 4u));
    Cuda(cudaGraphLaunch(graph.executable, stream.value)); Cuda(cudaStreamSynchronize(stream.value));
    Cuda(cudaMemcpy(actual.data(), dc.value, actual.size() * 4u, cudaMemcpyDeviceToHost));
    Require(std::memcmp(actual.data(), expected.data(), actual.size() * 4u) == 0, "MatMul replay changed values");
    io[2].dims.d[io[2].dims.nbDims - 1] += 1;
    Require(plugin->getWorkspaceSize(io, 2, io + 2, 1) == 0, "MatMul wrong output shape accepted");
    Require(plugin->enqueue(io, io + 2, inputs, outputs, workspace.value, stream.value) != 0, "MatMul invalid output enqueued");
    plugin->detachFromContext(); plugin->terminate();
    Require(plugin->enqueue(io, io + 2, inputs, outputs, workspace.value, stream.value) != 0, "MatMul detached handle accepted");
  }
  std::printf("Native BF16 MatMul: %zu static/batched/broadcast cases, full values, clone/serialization and replay passed\n", cases.size());
}

std::vector<int32_t> FrequencyWords(const std::vector<double>& frequencies) {
  std::vector<int32_t> words(2 * frequencies.size());
  for (std::size_t index = 0; index < frequencies.size(); ++index) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &frequencies[index], sizeof(bits));
    const std::uint32_t low = static_cast<std::uint32_t>(bits), high = static_cast<std::uint32_t>(bits >> 32u);
    std::memcpy(&words[2 * index], &low, sizeof(low));
    std::memcpy(&words[2 * index + 1], &high, sizeof(high));
  }
  return words;
}

Plugin CreateSinCos(nvinfer1::IPluginCreator* creator, const std::vector<double>& frequencies) {
  const auto words = FrequencyWords(frequencies);
  const nvinfer1::PluginField field{"frequency_words", words.data(), nvinfer1::PluginFieldType::kINT32,
      static_cast<int32_t>(words.size())};
  const nvinfer1::PluginFieldCollection fields{1, &field};
  return Plugin(creator->createPlugin("scalar", &fields), Destroy);
}

std::vector<char> Serialized(const Plugin& plugin) {
  Require(plugin != nullptr, "scalar sin/cos creation failed");
  std::vector<char> bytes(plugin->getSerializationSize());
  plugin->serialize(bytes.data());
  return bytes;
}

std::array<nvinfer1::PluginTensorDesc, 2> ScalarDescriptors(int rank, std::size_t count) {
  std::array<nvinfer1::PluginTensorDesc, 2> io{};
  for (auto& desc : io) {
    desc.type = nvinfer1::DataType::kFLOAT;
    desc.format = nvinfer1::TensorFormat::kLINEAR;
  }
  io[0].dims.nbDims = rank;
  for (int axis = 0; axis < rank; ++axis) io[0].dims.d[axis] = 1;
  io[1].dims = nvinfer1::Dims2{1, static_cast<int32_t>(2 * count)};
  return io;
}

std::array<nvinfer1::DynamicPluginTensorDesc, 2> ScalarProfile(
    const std::array<nvinfer1::PluginTensorDesc, 2>& io) {
  std::array<nvinfer1::DynamicPluginTensorDesc, 2> profile{};
  for (int slot = 0; slot < 2; ++slot) {
    profile[slot].desc = io[slot];
    profile[slot].min = profile[slot].max = io[slot].dims;
  }
  return profile;
}

void TestScalarSinCosContract() {
  auto* creator = getPluginRegistry()->getPluginCreator("VLAForgeScalarSinCosF64", "1", "");
  Require(creator != nullptr, "scalar sin/cos creator missing");
  const auto* schema = creator->getFieldNames();
  Require(schema != nullptr && schema->nbFields == 1 && schema->fields != nullptr
      && std::strcmp(schema->fields[0].name, "frequency_words") == 0
      && schema->fields[0].type == nvinfer1::PluginFieldType::kINT32, "scalar creator field schema differs");
  const std::vector<double> frequencies{0.0, -0.0, -0.123456789012345, 100000000.125,
      std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::max()};
  Plugin original = CreateSinCos(creator, frequencies);
  const auto bytes = Serialized(original);
  Require(std::strcmp(original->getPluginNamespace(), "") == 0, "scalar plugin namespace differs");
  Require(bytes.size() == 16 + 8 * frequencies.size(), "scalar serialization size differs");
  Plugin restored(creator->deserializePlugin("restored", bytes.data(), bytes.size()), Destroy);
  Require(Serialized(restored) == bytes, "scalar serialization changed frequency bits");
  Plugin clone(restored->clone(), Destroy);
  Require(Serialized(clone) == bytes, "scalar clone changed frequency bits");
  for (std::size_t length = 0; length < bytes.size(); ++length) {
    Require(creator->deserializePlugin("truncated", bytes.data(), length) == nullptr,
        "truncated scalar serialization accepted");
  }
  Require(creator->deserializePlugin("null", nullptr, bytes.size()) == nullptr, "null scalar serialization accepted");
  auto corrupt = bytes;
  corrupt.push_back(0);
  Require(creator->deserializePlugin("extra", corrupt.data(), corrupt.size()) == nullptr,
      "trailing scalar serialization accepted");
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    for (unsigned int bit = 0; bit < 8; ++bit) {
      corrupt = bytes;
      corrupt[index] ^= static_cast<char>(1u << bit);
      Require(creator->deserializePlugin("corrupt", corrupt.data(), corrupt.size()) == nullptr,
          "single-bit scalar serialization corruption accepted");
    }
  }
  Require(creator->deserializePlugin("overflow", bytes.data(), std::numeric_limits<std::size_t>::max()) == nullptr,
      "overflow scalar serialization length accepted");
  Require(creator->createPlugin("missing", nullptr) == nullptr, "missing scalar field accepted");
  auto words = FrequencyWords(frequencies);
  nvinfer1::PluginField field{"frequency_words", words.data(), nvinfer1::PluginFieldType::kINT32,
      static_cast<int32_t>(words.size())};
  nvinfer1::PluginFieldCollection fields{1, &field};
  for (int count : {-1, 0, 2}) {
    fields.nbFields = count;
    Require(creator->createPlugin("bad_count", &fields) == nullptr, "invalid scalar field count accepted");
  }
  fields.nbFields = 1; fields.fields = nullptr;
  Require(creator->createPlugin("null_fields", &fields) == nullptr, "null scalar fields accepted");
  fields.fields = &field;
  for (int length : {-2, 0, 1, 3, 32770, std::numeric_limits<int32_t>::max()}) {
    field.length = length;
    Require(creator->createPlugin("bad_length", &fields) == nullptr, "invalid scalar frequency count accepted");
  }
  field.length = static_cast<int32_t>(words.size());
  field.name = nullptr;
  Require(creator->createPlugin("null_name", &fields) == nullptr, "null scalar field name accepted");
  field.name = "frequencies";
  Require(creator->createPlugin("unknown", &fields) == nullptr, "unknown scalar field accepted");
  field.name = "frequency_words"; field.data = nullptr;
  Require(creator->createPlugin("null_data", &fields) == nullptr, "null scalar field data accepted");
  field.data = words.data(); field.type = nvinfer1::PluginFieldType::kFLOAT32;
  Require(creator->createPlugin("float_field", &fields) == nullptr, "FLOAT scalar field accepted");
  for (double bad : {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN()}) {
    Require(CreateSinCos(creator, {bad}) == nullptr, "nonfinite scalar frequency accepted");
  }
  for (std::size_t count : {std::size_t{1}, std::size_t{16384}}) {
    Plugin bounded = CreateSinCos(creator, std::vector<double>(count, 0.125));
    const auto bounded_bytes = Serialized(bounded);
    Plugin bounded_restored(creator->deserializePlugin("bounded", bounded_bytes.data(), bounded_bytes.size()), Destroy);
    Require(Serialized(bounded_restored) == bounded_bytes, "scalar count boundary did not roundtrip");
  }
  std::vector<unsigned char> unaligned(1 + words.size() * sizeof(int32_t));
  std::memcpy(unaligned.data() + 1, words.data(), words.size() * sizeof(int32_t));
  field.data = unaligned.data() + 1; field.type = nvinfer1::PluginFieldType::kINT32;
  Plugin unaligned_plugin(creator->createPlugin("unaligned", &fields), Destroy);
  Require(Serialized(unaligned_plugin) == bytes, "unaligned INT32 storage changed frequency bits");
  std::fill(unaligned.begin(), unaligned.end(), 0);
  Require(Serialized(unaligned_plugin) == bytes, "scalar plugin retained borrowed creator storage");
  auto* plugin = static_cast<nvinfer1::IPluginV2DynamicExt*>(clone.get());
  const nvinfer1::DataType input_type = nvinfer1::DataType::kFLOAT;
  Require(plugin->getNbOutputs() == 1 && plugin->getOutputDataType(0, &input_type, 1) == input_type,
      "scalar output type/count differs");
  for (int rank = 0; rank <= nvinfer1::Dims::MAX_DIMS; ++rank) {
    auto io = ScalarDescriptors(rank, frequencies.size());
    auto profile = ScalarProfile(io);
    plugin->configurePlugin(profile.data(), 1, profile.data() + 1, 1);
    Require(plugin->supportsFormatCombination(0, io.data(), 1, 1)
        && plugin->supportsFormatCombination(1, io.data(), 1, 1), "static scalar shape rejected");
    Require(plugin->getWorkspaceSize(io.data(), 1, io.data() + 1, 1) == 0, "scalar workspace is nonzero");
  }
  const auto valid_io = ScalarDescriptors(1, frequencies.size());
  Require(!plugin->supportsFormatCombination(-1, valid_io.data(), 1, 1)
      && !plugin->supportsFormatCombination(2, valid_io.data(), 1, 1)
      && !plugin->supportsFormatCombination(0, nullptr, 1, 1)
      && !plugin->supportsFormatCombination(0, valid_io.data(), 2, 1)
      && !plugin->supportsFormatCombination(0, valid_io.data(), 1, 2), "scalar invalid format arguments accepted");
  for (int test = 0; test < 12; ++test) {
    auto io = valid_io;
    if (test < 5) {
      if (test == 0) io[0].dims.d[0] = 2;
      if (test == 1) io[0].dims.d[0] = 0;
      if (test == 2) io[0].dims.d[0] = -1;
      if (test == 3) io[0].dims.nbDims = -1;
      if (test == 4) io[0].dims.nbDims = nvinfer1::Dims::MAX_DIMS + 1;
    }
    if (test == 5) io[1].dims.d[0] = 2;
    if (test == 6) io[1].dims.d[1] -= 1;
    if (test == 7) io[1].dims.nbDims = 1;
    if (test == 8) io[0].type = nvinfer1::DataType::kHALF;
    if (test == 9) io[1].type = nvinfer1::DataType::kHALF;
    if (test == 10) io[0].format = nvinfer1::TensorFormat::kCHW2;
    if (test == 11) io[1].format = nvinfer1::TensorFormat::kCHW2;
    Require(!plugin->supportsFormatCombination(1, io.data(), 1, 1), "invalid scalar shape/type/format accepted");
    Plugin rejected(original->clone(), Destroy);
    auto* dynamic = static_cast<nvinfer1::IPluginV2DynamicExt*>(rejected.get());
    const auto profile = ScalarProfile(io);
    dynamic->configurePlugin(profile.data(), 1, profile.data() + 1, 1);
    Require(dynamic->initialize() != 0, "invalid scalar configuration initialized");
    Plugin rejected_clone(rejected->clone(), Destroy);
    Require(rejected_clone != nullptr && rejected_clone->initialize() != 0, "scalar clone lost invalid configuration");
  }
  for (int test = 0; test < 6; ++test) {
    auto profile = ScalarProfile(valid_io);
    if (test == 0) profile[0].min.d[0] = 0;
    if (test == 1) profile[0].max.d[0] = 2;
    if (test == 2) profile[0].min.nbDims = 0;
    if (test == 3) profile[0].max.nbDims = 0;
    if (test == 4) profile[1].min.d[1] -= 1;
    if (test == 5) profile[1].max.d[1] += 1;
    Plugin rejected(original->clone(), Destroy);
    auto* dynamic = static_cast<nvinfer1::IPluginV2DynamicExt*>(rejected.get());
    dynamic->configurePlugin(profile.data(), 1, profile.data() + 1, 1);
    Require(dynamic->initialize() != 0, "dynamic scalar profile initialized");
  }
  Require(plugin->enqueue(valid_io.data(), valid_io.data() + 1, nullptr, nullptr, nullptr, nullptr) != 0,
      "uninitialized scalar plugin enqueued");
  std::printf("Scalar F64 sin/cos host contract: fields, scalar ranks, bounds, corruption, clone and serialization passed\n");
}

void CheckSinCos(const std::vector<float>& actual, float time, const std::vector<double>& frequencies) {
  const float infinity = std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < frequencies.size(); ++index) {
    const double angle = static_cast<double>(time) * frequencies[index];
    const float expected[] = {static_cast<float>(std::sin(angle)), static_cast<float>(std::cos(angle))};
    for (int half = 0; half < 2; ++half) {
      const float value = actual[index + half * frequencies.size()];
      Require(std::isfinite(value) && value >= std::nextafter(expected[half], -infinity)
          && value <= std::nextafter(expected[half], infinity), "scalar F64 sin/cos differs by more than one FLOAT ULP");
      if (expected[half] == 0.0F) {
        Require(std::memcmp(&value, &expected[half], sizeof(float)) == 0, "scalar sin/cos changed signed zero");
      }
    }
  }
}

void TestScalarSinCosDevice() {
  auto* creator = getPluginRegistry()->getPluginCreator("VLAForgeScalarSinCosF64", "1", "");
  std::vector<double> frequencies_a(257);
  for (std::size_t index = 0; index < frequencies_a.size(); ++index) {
    frequencies_a[index] = index % 2 ? -0.123456789012345 * index : 100000000.125 + index * 0.125;
  }
  frequencies_a[0] = 0.0; frequencies_a[1] = -0.0;
  const std::vector<double> frequencies_b{1.000000000001, -3.141592653589793, 0.000000123456789};
  Plugin original = CreateSinCos(creator, frequencies_a);
  Require(original != nullptr && original->initialize() == 0, "scalar original initialization failed");
  Plugin instance_a(original->clone(), Destroy);
  Require(instance_a != nullptr && instance_a->initialize() == 0, "scalar initialized clone failed");
  original->terminate(); original.reset();
  Plugin instance_b = CreateSinCos(creator, frequencies_b);
  Require(instance_b != nullptr && instance_b->initialize() == 0, "scalar second frequency initialization failed");
  const auto b_bytes = Serialized(instance_b);
  instance_b.reset(creator->deserializePlugin("restored_b", b_bytes.data(), b_bytes.size()));
  Require(instance_b != nullptr, "scalar second deserialize failed");
  auto* a = static_cast<nvinfer1::IPluginV2DynamicExt*>(instance_a.get());
  auto* b = static_cast<nvinfer1::IPluginV2DynamicExt*>(instance_b.get());
  b->attachToContext(nullptr, nullptr, nullptr);
  ScalarDeviceMemory time_a, time_b, output_a, output_b;
  Stream stream_a, stream_b;
  Cuda(cudaStreamCreateWithFlags(&stream_a.value, cudaStreamNonBlocking));
  Cuda(cudaStreamCreateWithFlags(&stream_b.value, cudaStreamNonBlocking));
  Cuda(cudaMalloc(&time_a.value, sizeof(float))); Cuda(cudaMalloc(&time_b.value, sizeof(float)));
  Cuda(cudaMalloc(&output_a.value, frequencies_a.size() * 2 * sizeof(float)));
  Cuda(cudaMalloc(&output_b.value, frequencies_b.size() * 2 * sizeof(float)));
  const void* input_a[] = {time_a.value}; const void* input_b[] = {time_b.value};
  void* outputs_a[] = {output_a.value}; void* outputs_b[] = {output_b.value};
  std::vector<float> actual_a(2 * frequencies_a.size()), actual_b(2 * frequencies_b.size());
  for (int rank = 0; rank <= nvinfer1::Dims::MAX_DIMS; ++rank) {
    auto io_a = ScalarDescriptors(rank, frequencies_a.size());
    auto io_b = ScalarDescriptors(rank, frequencies_b.size());
    const auto profile_a = ScalarProfile(io_a), profile_b = ScalarProfile(io_b);
    a->configurePlugin(profile_a.data(), 1, profile_a.data() + 1, 1);
    b->configurePlugin(profile_b.data(), 1, profile_b.data() + 1, 1);
    Graph graph_a, graph_b;
    for (int iteration = 0; iteration < 2; ++iteration) {
      const float ta = iteration == 0 ? -0.125F : 123.5F;
      const float tb = iteration == 0 ? 17.25F : -0.0F;
      Cuda(cudaMemcpyAsync(time_a.value, &ta, sizeof(float), cudaMemcpyHostToDevice, stream_a.value));
      Cuda(cudaMemcpyAsync(time_b.value, &tb, sizeof(float), cudaMemcpyHostToDevice, stream_b.value));
      if (iteration == 0) {
        Require(a->enqueue(io_a.data(), io_a.data() + 1, input_a, outputs_a, nullptr, stream_a.value) == 0,
            "scalar first nonblocking enqueue failed");
        Require(b->enqueue(io_b.data(), io_b.data() + 1, input_b, outputs_b, nullptr, stream_b.value) == 0,
            "scalar second nonblocking enqueue failed");
      } else {
        Cuda(cudaMemsetAsync(output_a.value, 0, actual_a.size() * sizeof(float), stream_a.value));
        Cuda(cudaMemsetAsync(output_b.value, 0, actual_b.size() * sizeof(float), stream_b.value));
        Cuda(cudaGraphLaunch(graph_a.executable, stream_a.value));
        Cuda(cudaGraphLaunch(graph_b.executable, stream_b.value));
      }
      Cuda(cudaStreamSynchronize(stream_a.value)); Cuda(cudaStreamSynchronize(stream_b.value));
      Cuda(cudaMemcpy(actual_a.data(), output_a.value, actual_a.size() * sizeof(float), cudaMemcpyDeviceToHost));
      Cuda(cudaMemcpy(actual_b.data(), output_b.value, actual_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
      CheckSinCos(actual_a, ta, frequencies_a); CheckSinCos(actual_b, tb, frequencies_b);
      if (iteration == 0) {
        Cuda(cudaStreamBeginCapture(stream_a.value, cudaStreamCaptureModeThreadLocal));
        Require(a->enqueue(io_a.data(), io_a.data() + 1, input_a, outputs_a, nullptr, stream_a.value) == 0,
            "scalar first capture failed");
        Cuda(cudaStreamEndCapture(stream_a.value, &graph_a.value));
        Cuda(cudaGraphInstantiate(&graph_a.executable, graph_a.value, nullptr, nullptr, 0));
        Cuda(cudaStreamBeginCapture(stream_b.value, cudaStreamCaptureModeThreadLocal));
        Require(b->enqueue(io_b.data(), io_b.data() + 1, input_b, outputs_b, nullptr, stream_b.value) == 0,
            "scalar second capture failed");
        Cuda(cudaStreamEndCapture(stream_b.value, &graph_b.value));
        Cuda(cudaGraphInstantiate(&graph_b.executable, graph_b.value, nullptr, nullptr, 0));
      }
    }
    io_a[0].dims.nbDims = 1; io_a[0].dims.d[0] = 2;
    Require(a->enqueue(io_a.data(), io_a.data() + 1, input_a, outputs_a, nullptr, stream_a.value) != 0,
        "nonscalar runtime input enqueued");
    io_a = ScalarDescriptors(rank, frequencies_a.size()); io_a[1].dims.d[1] -= 1;
    Require(a->enqueue(io_a.data(), io_a.data() + 1, input_a, outputs_a, nullptr, stream_a.value) != 0,
        "wrong scalar runtime output enqueued");
  }
  a->terminate(); b->detachFromContext();
  const auto io = ScalarDescriptors(0, frequencies_a.size());
  Require(a->enqueue(io.data(), io.data() + 1, input_a, outputs_a, nullptr, stream_a.value) != 0,
      "terminated scalar plugin enqueued");
  std::printf("Scalar F64 sin/cos device: independent N=257/3, ranks 0..8, clone/deserialize, double references and two-stream capture/replay passed\n");
}

__global__ void OfficialRound(const float* input, float* output, std::size_t count) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = __bfloat162float(__float2bfloat16_rn(input[index]));
  }
}
}  // namespace

int main(int argc, char** argv) {
  float *device_input = nullptr, *device_output = nullptr, *device_expected = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  try {
    Require(vlaforge_register_tensorrt_precision_plugins() == 1, "registration failed");
    Require(vlaforge_register_tensorrt_precision_plugins() == 1, "registration not idempotent");
    TestScalarSinCosContract();
    if (argc == 2 && std::strcmp(argv[1], "--scalar-host-only") == 0) return 0;
    Require(argc == 1, "usage: vlaforge_tensorrt_precision_plugin_smoke [--scalar-host-only]");
    TestScalarSinCosDevice();
    TestMatMul();
    auto* creator = getPluginRegistry()->getPluginCreator("VLAForgeBFloat16Round", "1", "");
    Require(creator != nullptr, "creator missing");
    Plugin plugin(creator->createPlugin("round", nullptr), Destroy);
    Require(plugin != nullptr, "creation failed");
    std::vector<char> bytes(plugin->getSerializationSize());
    plugin->serialize(bytes.data());
    Require(creator->deserializePlugin("bad", bytes.data(), bytes.size() - 1) == nullptr,
        "truncated serialization accepted");
    Plugin restored(creator->deserializePlugin("restored", bytes.data(), bytes.size()), Destroy);
    Require(restored != nullptr, "deserialization failed");
    bytes[0] ^= 1;
    Require(creator->deserializePlugin("bad", bytes.data(), bytes.size()) == nullptr,
        "corrupt serialization accepted");
    auto* dynamic = static_cast<nvinfer1::IPluginV2DynamicExt*>(restored.get());
    Require(dynamic->initialize() == 0, "initialization failed");
    std::vector<float> input;
    for (std::uint32_t high = 0; high < 65536; ++high) {
      for (std::uint32_t low : {0u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu}) {
        const std::uint32_t bits = (high << 16) | low;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        input.push_back(value);
      }
    }
    const std::size_t count = input.size(), size = count * sizeof(float);
    std::vector<float> actual(count), expected(count);
    Cuda(cudaStreamCreate(&stream));
    Cuda(cudaMalloc(&device_input, size));
    Cuda(cudaMalloc(&device_output, size));
    Cuda(cudaMalloc(&device_expected, size));
    Cuda(cudaMemcpy(device_input, input.data(), size, cudaMemcpyHostToDevice));
    nvinfer1::PluginTensorDesc io[2]{};
    for (auto& item : io) {
      item.dims = nvinfer1::Dims2{2, static_cast<int32_t>(count / 2)};
      item.type = nvinfer1::DataType::kFLOAT;
      item.format = nvinfer1::TensorFormat::kLINEAR;
    }
    Require(dynamic->supportsFormatCombination(0, io, 1, 1), "FLOAT input rejected");
    Require(dynamic->supportsFormatCombination(1, io, 1, 1), "FLOAT output rejected");
    Require(!dynamic->supportsFormatCombination(2, io, 1, 1), "invalid position accepted");
    io[1].type = nvinfer1::DataType::kHALF;
    Require(!dynamic->supportsFormatCombination(1, io, 1, 1), "HALF carrier accepted");
    io[1].type = nvinfer1::DataType::kFLOAT;
    const void* inputs[] = {device_input};
    void* outputs[] = {device_output};
    Require(dynamic->enqueue(io, io + 1, inputs, outputs, nullptr, stream) == 0, "enqueue failed");
    OfficialRound<<<128, 256, 0, stream>>>(device_input, device_expected, count);
    Cuda(cudaStreamSynchronize(stream));
    Cuda(cudaMemcpy(actual.data(), device_output, size, cudaMemcpyDeviceToHost));
    Cuda(cudaMemcpy(expected.data(), device_expected, size, cudaMemcpyDeviceToHost));
    for (std::size_t index = 0; index < count; ++index) {
      if (std::isnan(expected[index])) {
        Require(std::isnan(actual[index]), "NaN became a finite value or infinity");
      } else {
        Require(std::memcmp(&actual[index], &expected[index], sizeof(float)) == 0,
            "rounding differs from CUDA BF16 RNE");
      }
    }
    Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    Require(dynamic->enqueue(io, io + 1, inputs, outputs, nullptr, stream) == 0, "capture enqueue failed");
    Cuda(cudaStreamEndCapture(stream, &graph));
    Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    Cuda(cudaMemset(device_output, 0, size));
    Cuda(cudaGraphLaunch(executable, stream));
    Cuda(cudaStreamSynchronize(stream));
    Cuda(cudaMemcpy(expected.data(), device_output, size, cudaMemcpyDeviceToHost));
    Require(std::memcmp(actual.data(), expected.data(), size) == 0, "replay changed output bits");
    io[1].dims.d[0] = 3;
    Require(dynamic->enqueue(io, io + 1, inputs, outputs, nullptr, stream) != 0,
        "shape mismatch accepted");
    io[1].dims = io[0].dims;
    io[0].dims.d[0] = io[1].dims.d[0] = 0;
    Require(dynamic->enqueue(io, io + 1, inputs, outputs, nullptr, stream) == 0,
        "empty tensor rejected");
    dynamic->terminate();
    Cuda(cudaGraphExecDestroy(executable)); executable = nullptr;
    Cuda(cudaGraphDestroy(graph)); graph = nullptr;
    Cuda(cudaFree(device_expected)); device_expected = nullptr;
    Cuda(cudaFree(device_output)); device_output = nullptr;
    Cuda(cudaFree(device_input)); device_input = nullptr;
    Cuda(cudaStreamDestroy(stream)); stream = nullptr;
    std::printf("BF16 FLOAT-carrier plugin: %zu values, finite bits/CUDA RNE, NaN class, serialization and replay passed\n", count);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    if (device_expected) cudaFree(device_expected);
    if (device_output) cudaFree(device_output);
    if (device_input) cudaFree(device_input);
    if (stream) cudaStreamDestroy(stream);
    return 1;
  }
}
