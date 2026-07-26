#include "aoti_sequence_runner.h"

#include "aoti_callable.h"

#include <c10/core/ScalarType.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vlaforge/runtime/artifact_verifier.h"

namespace vlaforge::backends {
namespace {

constexpr std::size_t kMaximumValues = 4096u;
constexpr std::size_t kMaximumArtifacts = 512u;
constexpr std::size_t kMaximumNodes = 8192u;
constexpr std::size_t kMaximumRank = 16u;
constexpr std::size_t kMaximumBindings = 128u;

enum class ValueRole {
  kInput,
  kOutput,
  kTemporary,
};

struct ValueSpec {
  ValueRole role = ValueRole::kTemporary;
  std::optional<std::size_t> binding;
  c10::ScalarType dtype = c10::ScalarType::Undefined;
  std::vector<std::int64_t> shape;
};

struct ArtifactSpec {
  std::string path;
  std::string sha256;
  std::uint64_t size_bytes = 0u;
  std::string resolved_path;
};

struct NodeSpec {
  std::size_t artifact_id = 0u;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
};

template <typename T>
T Read(std::istream& stream, const char* field) {
  T value{};
  if (!(stream >> value)) {
    throw std::runtime_error(
        std::string("invalid AOTI sequence ") + field);
  }
  return value;
}

void Expect(std::istream& stream, const char* expected) {
  const auto value = Read<std::string>(stream, expected);
  if (value != expected) {
    throw std::runtime_error(
        std::string("expected AOTI sequence token ") + expected);
  }
}

std::size_t ReadCount(
    std::istream& stream, const char* field, std::size_t maximum) {
  const auto value = Read<std::uint64_t>(stream, field);
  if (value > maximum) {
    throw std::runtime_error(
        std::string("AOTI sequence ") + field + " exceeds limit");
  }
  return static_cast<std::size_t>(value);
}

ValueRole ParseRole(const std::string& value) {
  if (value == "input") {
    return ValueRole::kInput;
  }
  if (value == "output") {
    return ValueRole::kOutput;
  }
  if (value == "temporary") {
    return ValueRole::kTemporary;
  }
  throw std::runtime_error("unsupported AOTI sequence value role");
}

c10::ScalarType ParseDType(const std::string& value) {
  if (value == "bool") {
    return c10::ScalarType::Bool;
  }
  if (value == "i32") {
    return c10::ScalarType::Int;
  }
  if (value == "i64") {
    return c10::ScalarType::Long;
  }
  if (value == "f16") {
    return c10::ScalarType::Half;
  }
  if (value == "bf16") {
    return c10::ScalarType::BFloat16;
  }
  if (value == "f32") {
    return c10::ScalarType::Float;
  }
  if (value == "f64") {
    return c10::ScalarType::Double;
  }
  if (value == "u64") {
    return c10::ScalarType::UInt64;
  }
  if (value == "u8") {
    return c10::ScalarType::Byte;
  }
  throw std::runtime_error("unsupported AOTI sequence value dtype");
}

bool TensorMatches(
    const at::Tensor& tensor, const ValueSpec& spec,
    VLAForgeDeviceKind device_kind, int device_ordinal) {
  if (!tensor.defined() || !tensor.is_contiguous() ||
      tensor.scalar_type() != spec.dtype ||
      tensor.dim() != static_cast<std::int64_t>(spec.shape.size())) {
    return false;
  }
  if (device_kind == VLAFORGE_DEVICE_CUDA) {
    if (!tensor.is_cuda() || tensor.get_device() != device_ordinal) {
      return false;
    }
  } else if (!tensor.device().is_cpu()) {
    return false;
  }
  for (std::size_t index = 0; index < spec.shape.size(); ++index) {
    if (tensor.size(static_cast<std::int64_t>(index)) != spec.shape[index]) {
      return false;
    }
  }
  return true;
}

std::string DeviceName(
    VLAForgeDeviceKind device_kind, int device_ordinal) {
  if (device_kind == VLAFORGE_DEVICE_CPU) {
    return "cpu";
  }
  if (device_kind == VLAFORGE_DEVICE_CUDA) {
    return "cuda:" + std::to_string(device_ordinal);
  }
  throw std::runtime_error("AOTI sequence requires CPU or CUDA device");
}

}  // namespace

struct AotiSequenceRunner::Impl {
  VLAForgeDeviceKind device_kind = VLAFORGE_DEVICE_CPU;
  int device_ordinal = 0;
  bool is_loaded = false;
  std::vector<ValueSpec> values;
  std::vector<ArtifactSpec> artifact_specs;
  std::vector<std::unique_ptr<AotiCallable>> artifacts;
  std::vector<NodeSpec> nodes;
  std::vector<std::size_t> value_use_counts;
  std::size_t input_count = 0u;
  std::size_t output_count = 0u;
};

AotiSequenceRunner::AotiSequenceRunner(
    VLAForgeDeviceKind device_kind, int device_ordinal)
    : impl_(std::make_unique<Impl>()) {
  impl_->device_kind = device_kind;
  impl_->device_ordinal = device_ordinal;
}

AotiSequenceRunner::~AotiSequenceRunner() = default;

void AotiSequenceRunner::Load(
    const std::string& manifest_path, const std::string& target) {
  if (impl_->is_loaded) {
    throw std::runtime_error("AOTI sequence is already loaded");
  }
  std::ifstream stream(manifest_path);
  if (!stream) {
    throw std::runtime_error("AOTI sequence manifest cannot be opened");
  }
  Expect(stream, "VLAFORGE_AOTI_SEQUENCE");
  if (Read<std::uint32_t>(stream, "version") != 1u) {
    throw std::runtime_error("unsupported AOTI sequence version");
  }
  Expect(stream, "region");
  const auto region_name = Read<std::string>(stream, "Region name");
  if (region_name.empty()) {
    throw std::runtime_error("AOTI sequence Region name is empty");
  }
  Expect(stream, "target");
  if (Read<std::string>(stream, "target") != target) {
    throw std::runtime_error("AOTI sequence target mismatch");
  }
  Expect(stream, "device");
  if (Read<std::string>(stream, "device") !=
      DeviceName(impl_->device_kind, impl_->device_ordinal)) {
    throw std::runtime_error("AOTI sequence device mismatch");
  }

  Expect(stream, "values");
  const auto value_count = ReadCount(stream, "value count", kMaximumValues);
  if (value_count == 0u) {
    throw std::runtime_error("AOTI sequence has no values");
  }
  impl_->values.reserve(value_count);
  std::vector<std::size_t> input_bindings;
  std::vector<std::size_t> output_bindings;
  for (std::size_t expected_id = 0u; expected_id < value_count;
       ++expected_id) {
    Expect(stream, "value");
    if (ReadCount(stream, "value id", kMaximumValues) != expected_id) {
      throw std::runtime_error(
          "AOTI sequence value ids must be dense and ordered");
    }
    ValueSpec spec;
    spec.role = ParseRole(Read<std::string>(stream, "value role"));
    const auto raw_binding = Read<std::int64_t>(stream, "value binding");
    if (spec.role == ValueRole::kTemporary) {
      if (raw_binding != -1) {
        throw std::runtime_error(
            "AOTI sequence temporary has an external binding");
      }
    } else {
      if (raw_binding < 0 ||
          static_cast<std::uint64_t>(raw_binding) >= kMaximumBindings) {
        throw std::runtime_error(
            "AOTI sequence external binding is invalid");
      }
      spec.binding = static_cast<std::size_t>(raw_binding);
      (spec.role == ValueRole::kInput
           ? input_bindings
           : output_bindings)
          .push_back(*spec.binding);
    }
    spec.dtype = ParseDType(Read<std::string>(stream, "value dtype"));
    const auto rank = ReadCount(stream, "value rank", kMaximumRank);
    spec.shape.reserve(rank);
    std::uint64_t element_count = 1u;
    for (std::size_t index = 0u; index < rank; ++index) {
      const auto dimension = Read<std::int64_t>(stream, "dimension");
      if (dimension < 0) {
        throw std::runtime_error(
            "AOTI sequence dimension must be non-negative");
      }
      const auto unsigned_dimension =
          static_cast<std::uint64_t>(dimension);
      if (unsigned_dimension != 0u &&
          element_count >
              std::numeric_limits<std::uint64_t>::max() /
                  unsigned_dimension) {
        throw std::runtime_error("AOTI sequence tensor size overflows");
      }
      element_count *= unsigned_dimension;
      spec.shape.push_back(dimension);
    }
    impl_->values.push_back(std::move(spec));
  }

  auto validate_bindings = [](std::vector<std::size_t> bindings,
                              const char* kind) {
    std::sort(bindings.begin(), bindings.end());
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
      if (bindings[index] != index) {
        throw std::runtime_error(
            std::string("AOTI sequence ") + kind +
            " bindings must be dense and unique");
      }
    }
  };
  validate_bindings(input_bindings, "input");
  validate_bindings(output_bindings, "output");
  impl_->input_count = input_bindings.size();
  impl_->output_count = output_bindings.size();

  Expect(stream, "artifacts");
  const auto artifact_count =
      ReadCount(stream, "artifact count", kMaximumArtifacts);
  if (artifact_count == 0u) {
    throw std::runtime_error("AOTI sequence has no artifacts");
  }
  impl_->artifact_specs.reserve(artifact_count);
  const auto root =
      std::filesystem::path(manifest_path).parent_path().string();
  for (std::size_t expected_id = 0u; expected_id < artifact_count;
       ++expected_id) {
    Expect(stream, "artifact");
    if (ReadCount(stream, "artifact id", kMaximumArtifacts) != expected_id) {
      throw std::runtime_error(
          "AOTI sequence artifact ids must be dense and ordered");
    }
    ArtifactSpec spec;
    spec.path = Read<std::string>(stream, "artifact path");
    spec.sha256 = Read<std::string>(stream, "artifact SHA-256");
    spec.size_bytes = Read<std::uint64_t>(stream, "artifact size");
    const char* resolved_path = nullptr;
    const char* error_message = nullptr;
    const auto status = vlaforge_verify_artifact_file_abi(
        root.data(), root.size(), spec.path.data(), spec.path.size(),
        spec.sha256.data(), spec.sha256.size(), spec.size_bytes,
        &resolved_path, &error_message);
    if (status != static_cast<std::uint32_t>(
                      vlaforge::runtime::StatusCode::kOk)) {
      throw std::runtime_error(
          error_message == nullptr
              ? "AOTI sequence artifact verification failed"
              : error_message);
    }
    spec.resolved_path = resolved_path;
    impl_->artifact_specs.push_back(std::move(spec));
  }

  Expect(stream, "nodes");
  const auto node_count = ReadCount(stream, "node count", kMaximumNodes);
  if (node_count == 0u) {
    throw std::runtime_error("AOTI sequence has no nodes");
  }
  impl_->nodes.reserve(node_count);
  impl_->value_use_counts.assign(value_count, 0u);
  std::vector<bool> defined(value_count, false);
  for (std::size_t index = 0u; index < value_count; ++index) {
    defined[index] = impl_->values[index].role == ValueRole::kInput;
  }
  std::vector<bool> used_artifact(artifact_count, false);
  for (std::size_t node_index = 0u; node_index < node_count; ++node_index) {
    Expect(stream, "node");
    NodeSpec node;
    node.artifact_id =
        ReadCount(stream, "node artifact id", kMaximumArtifacts);
    if (node.artifact_id >= artifact_count) {
      throw std::runtime_error(
          "AOTI sequence node references unknown artifact");
    }
    used_artifact[node.artifact_id] = true;
    const auto input_count =
        ReadCount(stream, "node input count", kMaximumBindings);
    if (input_count == 0u) {
      throw std::runtime_error("AOTI sequence node has no inputs");
    }
    node.inputs.reserve(input_count);
    for (std::size_t index = 0u; index < input_count; ++index) {
      const auto value_id =
          ReadCount(stream, "node input value", kMaximumValues);
      if (value_id >= value_count || !defined[value_id]) {
        throw std::runtime_error(
            "AOTI sequence node reads an undefined value");
      }
      ++impl_->value_use_counts[value_id];
      node.inputs.push_back(value_id);
    }
    const auto output_count =
        ReadCount(stream, "node output count", kMaximumBindings);
    if (output_count == 0u) {
      throw std::runtime_error("AOTI sequence node has no outputs");
    }
    node.outputs.reserve(output_count);
    for (std::size_t index = 0u; index < output_count; ++index) {
      const auto value_id =
          ReadCount(stream, "node output value", kMaximumValues);
      if (value_id >= value_count ||
          impl_->values[value_id].role == ValueRole::kInput ||
          defined[value_id]) {
        throw std::runtime_error(
            "AOTI sequence node redefines an unavailable value");
      }
      defined[value_id] = true;
      node.outputs.push_back(value_id);
    }
    impl_->nodes.push_back(std::move(node));
  }
  Expect(stream, "end");
  std::string trailing;
  if (stream >> trailing) {
    throw std::runtime_error("AOTI sequence has trailing tokens");
  }
  for (const bool used : used_artifact) {
    if (!used) {
      throw std::runtime_error("AOTI sequence contains an unused artifact");
    }
  }
  for (std::size_t index = 0u; index < value_count; ++index) {
    if (impl_->values[index].role == ValueRole::kOutput &&
        !defined[index]) {
      throw std::runtime_error(
          "AOTI sequence does not produce every external output");
    }
    if (impl_->values[index].role == ValueRole::kTemporary &&
        (!defined[index] || impl_->value_use_counts[index] == 0u)) {
      throw std::runtime_error(
          "AOTI sequence contains an unused temporary value");
    }
  }

  impl_->artifacts.reserve(artifact_count);
  for (const auto& spec : impl_->artifact_specs) {
    auto callable =
        std::make_unique<AotiCallable>(
            impl_->device_kind, impl_->device_ordinal);
    callable->Load(spec.resolved_path);
    impl_->artifacts.push_back(std::move(callable));
  }
  impl_->is_loaded = true;
}

bool AotiSequenceRunner::loaded() const noexcept {
  return impl_->is_loaded;
}

std::vector<at::Tensor> AotiSequenceRunner::Run(
    std::vector<at::Tensor>& inputs) {
  if (!impl_->is_loaded) {
    throw std::runtime_error("AOTI sequence is not loaded");
  }
  if (inputs.size() != impl_->input_count) {
    throw std::runtime_error("AOTI sequence input count mismatch");
  }
  std::vector<at::Tensor> values(impl_->values.size());
  auto remaining_uses = impl_->value_use_counts;
  for (std::size_t value_id = 0u; value_id < impl_->values.size();
       ++value_id) {
    const auto& spec = impl_->values[value_id];
    if (spec.role != ValueRole::kInput) {
      continue;
    }
    const auto binding = *spec.binding;
    if (!TensorMatches(
            inputs[binding], spec, impl_->device_kind,
            impl_->device_ordinal)) {
      throw std::runtime_error(
          "AOTI sequence external input metadata mismatch");
    }
    values[value_id] = inputs[binding];
  }

  for (std::size_t node_index = 0u; node_index < impl_->nodes.size();
       ++node_index) {
    const auto& node = impl_->nodes[node_index];
    std::vector<at::Tensor> arguments;
    arguments.reserve(node.inputs.size());
    for (const auto value_id : node.inputs) {
      if (!values[value_id].defined()) {
        throw std::runtime_error(
            "AOTI sequence runtime value is undefined");
      }
      arguments.push_back(values[value_id]);
    }
    auto outputs = impl_->artifacts[node.artifact_id]->Run(arguments);
    if (outputs.size() != node.outputs.size()) {
      throw std::runtime_error(
          "AOTI sequence physical output count mismatch");
    }
    for (std::size_t index = 0u; index < outputs.size(); ++index) {
      const auto value_id = node.outputs[index];
      // Sequence values have a canonical dense ABI.  AOTI is allowed to
      // return a padded/view tensor even when the captured logical value is
      // dense, so materialize that boundary explicitly before it becomes a
      // loop-carried or downstream Region value.
      if (outputs[index].defined() && !outputs[index].is_contiguous()) {
        outputs[index] = outputs[index].contiguous();
      }
      if (!TensorMatches(
              outputs[index], impl_->values[value_id],
              impl_->device_kind, impl_->device_ordinal)) {
        throw std::runtime_error(
            "AOTI sequence physical output metadata mismatch at node " +
            std::to_string(node_index) + ", artifact " +
            std::to_string(node.artifact_id) + ", output " +
            std::to_string(index) + ", value " +
            std::to_string(value_id));
      }
      values[value_id] = std::move(outputs[index]);
    }
    for (const auto value_id : node.inputs) {
      if (remaining_uses[value_id] == 0u) {
        throw std::runtime_error(
            "AOTI sequence value liveness underflow");
      }
      --remaining_uses[value_id];
      if (remaining_uses[value_id] == 0u &&
          impl_->values[value_id].role == ValueRole::kTemporary) {
        values[value_id] = at::Tensor();
      }
    }
  }

  std::vector<at::Tensor> outputs(impl_->output_count);
  for (std::size_t value_id = 0u; value_id < impl_->values.size();
       ++value_id) {
    const auto& spec = impl_->values[value_id];
    if (spec.role == ValueRole::kOutput) {
      outputs[*spec.binding] = values[value_id];
    }
  }
  return outputs;
}

}  // namespace vlaforge::backends
