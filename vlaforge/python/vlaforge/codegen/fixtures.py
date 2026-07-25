"""C++ CPU artifact bodies and runner for deterministic offline fixtures."""

from vlaforge.codegen.model import (
    CppRegionDefinition,
    CppValidatorDefinition,
)


def openvla_fixture_regions() -> dict[str, CppRegionDefinition]:
    return {
        "encode_context": CppRegionDefinition(
            "encode_context",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 2u) ||
    !CheckTensor(executable->inputs[1], VLAFORGE_DTYPE_I64, 3u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 2u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "encode_context binding mismatch");
}
const auto* image = Input<float>(executable, 0u);
const auto* instruction = Input<std::int64_t>(executable, 1u);
auto* output = Output<float>(executable, 0u);
const float language =
    static_cast<float>(instruction[0] + instruction[1] + instruction[2]) /
    100.0f;
output[0] = image[0] + language;
output[1] = image[1] - language;
return vlaforge_status_ok();
""".strip(),
        ),
        "initial_action_token": CppRegionDefinition(
            "initial_action_token",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 2u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_I64, 1u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "initial token binding mismatch");
}
const auto* context = Input<float>(executable, 0u);
auto* output = Output<std::int64_t>(executable, 0u);
const auto rounded = static_cast<std::int64_t>(
    std::llrint((context[0] - context[1]) * 10.0f));
output[0] = ((rounded % 64) + 64) % 64;
return vlaforge_status_ok();
""".strip(),
        ),
        "next_action_token": CppRegionDefinition(
            "next_action_token",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 2u) ||
    !CheckTensor(executable->inputs[1], VLAFORGE_DTYPE_I64, 1u) ||
    !CheckTensor(executable->inputs[2], VLAFORGE_DTYPE_I64, 1u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_I64, 1u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "next token binding mismatch");
}
const auto* context = Input<float>(executable, 0u);
const auto token = Input<std::int64_t>(executable, 1u)[0];
const auto step = Input<std::int64_t>(executable, 2u)[0];
auto* output = Output<std::int64_t>(executable, 0u);
const auto bias = static_cast<std::int64_t>(
    std::llrint((context[0] + context[1]) * 5.0f));
output[0] = (token * 7 + bias + step + 3) % 64;
return vlaforge_status_ok();
""".strip(),
        ),
        "detokenize_action": CppRegionDefinition(
            "detokenize_action",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_I64, 1u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 2u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "detokenize binding mismatch");
}
const auto token = Input<std::int64_t>(executable, 0u)[0];
auto* output = Output<float>(executable, 0u);
const float first = static_cast<float>(token) / 63.0f * 2.0f - 1.0f;
output[0] = first;
output[1] = -0.5f * first;
return vlaforge_status_ok();
""".strip(),
        ),
    }


def openvla_fixture_validators() -> dict[str, CppValidatorDefinition]:
    return {
        "bounded_action": CppValidatorDefinition(
            "bounded_action",
            """
if (size_bytes != 2u * sizeof(float) || data == nullptr) {
  return false;
}
const auto* action = static_cast<const float*>(data);
return std::isfinite(action[0]) && std::isfinite(action[1]) &&
       action[0] >= -1.0f && action[0] <= 1.0f &&
       action[1] >= -1.0f && action[1] <= 1.0f;
""".strip(),
        )
    }


def openvla_fixture_runner_source() -> str:
    return r"""
#include "session_generated.h"

#include <cstdint>
#include <cstdio>

namespace {

VLAForgeBoundTensor Tensor(void* data, std::uint64_t bytes,
                           const std::int64_t* shape, std::uint32_t rank,
                           VLAForgeDType dtype) {
  return VLAForgeBoundTensor{
      sizeof(VLAForgeBoundTensor),
      {data, bytes, shape, rank, dtype, {VLAFORGE_DEVICE_CPU, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS,
      1u};
}

VLAForgeInputStamp Stamp(std::uint64_t revision) {
  VLAForgeInputStamp stamp{};
  stamp.struct_size = sizeof(VLAForgeInputStamp);
  stamp.has_revision = 1u;
  stamp.revision = revision;
  return stamp;
}

}  // namespace

int main() {
  vlaforge_generated::ModelSession typed;
  VLAForgeSession* generic = nullptr;
  if (vlaforge_model_session_create(&generic).code != VLAFORGE_STATUS_OK) {
    return 1;
  }
  const auto* api = vlaforge_model_session_api();
  if (vlaforge_session_api_validate(
          api, vlaforge_generated::kSchemaDigest,
          VLAFORGE_SCHEMA_DIGEST_HEX_SIZE).code != VLAFORGE_STATUS_OK) {
    return 2;
  }
  const std::int64_t image_shape[] = {2};
  const std::int64_t instruction_shape[] = {3};
  for (std::uint64_t index = 0; index < 3; ++index) {
    float image[] = {
        static_cast<float>(0.1 * static_cast<double>(index)),
        static_cast<float>(0.4 - 0.05 * static_cast<double>(index))};
    std::int64_t instruction[] = {
        1, static_cast<std::int64_t>(2 + index), 3};
    auto image_view =
        Tensor(image, sizeof(image), image_shape, 1u, VLAFORGE_DTYPE_F32);
    auto instruction_view = Tensor(
        instruction, sizeof(instruction), instruction_shape, 1u,
        VLAFORGE_DTYPE_I64);
    auto stamp = Stamp(index);

    vlaforge_generated::ModelInputs inputs{};
    inputs.image = image_view;
    inputs.image_stamp = stamp;
    inputs.instruction = instruction_view;
    inputs.instruction_stamp = stamp;
    vlaforge_generated::ModelOutputs outputs{};
    if (!typed.Run(inputs, &outputs).ok()) {
      return 3;
    }
    const auto* typed_action =
        static_cast<const float*>(outputs.action.tensor.data);

    if (api->bind_tensor(generic, 0u, &image_view, &stamp).code !=
            VLAFORGE_STATUS_OK ||
        api->bind_tensor(generic, 1u, &instruction_view, &stamp).code !=
            VLAFORGE_STATUS_OK ||
        api->run(generic).code != VLAFORGE_STATUS_OK) {
      return 4;
    }
    VLAForgeBoundTensor generic_output{};
    if (api->read_output_tensor(generic, 0u, &generic_output).code !=
        VLAFORGE_STATUS_OK) {
      return 5;
    }
    const auto* generic_action =
        static_cast<const float*>(generic_output.tensor.data);
    std::printf("OUTPUT,%llu,%.9g,%.9g,%.9g,%.9g\n",
                static_cast<unsigned long long>(index),
                static_cast<double>(typed_action[0]),
                static_cast<double>(typed_action[1]),
                static_cast<double>(generic_action[0]),
                static_cast<double>(generic_action[1]));
  }
  api->destroy(generic);
  return 0;
}
""".lstrip()


def smolvla_fixture_regions() -> dict[str, CppRegionDefinition]:
    return {
        "encode_observation": CppRegionDefinition(
            "encode_observation",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 2u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 2u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "encode");
}
const auto* image = Input<float>(executable, 0u);
auto* output = Output<float>(executable, 0u);
output[0] = image[0] * 2.0f;
output[1] = image[1] * 2.0f;
return vlaforge_status_ok();
""".strip(),
        ),
        "sample_noise": CppRegionDefinition(
            "sample_noise",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_I64, 1u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 8u) ||
    !CheckTensor(executable->outputs[1], VLAFORGE_DTYPE_I64, 1u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "noise");
}
const auto rng = Input<std::int64_t>(executable, 0u)[0];
auto* chunk = Output<float>(executable, 0u);
for (std::int64_t index = 0; index < 4; ++index) {
  chunk[index * 2] =
      static_cast<float>((rng * 17 + 11 + index * 5) % 101) / 100.0f;
  chunk[index * 2 + 1] =
      static_cast<float>((rng * 29 + 7 + index * 3) % 103) / 100.0f;
}
Output<std::int64_t>(executable, 1u)[0] = rng + 1;
return vlaforge_status_ok();
""".strip(),
        ),
        "solver_step": CppRegionDefinition(
            "solver_step",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 2u) ||
    !CheckTensor(executable->inputs[1], VLAFORGE_DTYPE_F32, 8u) ||
    !CheckTensor(executable->inputs[2], VLAFORGE_DTYPE_I64, 1u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 8u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "solver");
}
const auto* prefix = Input<float>(executable, 0u);
const auto* sample = Input<float>(executable, 1u);
const auto step = Input<std::int64_t>(executable, 2u)[0];
auto* output = Output<float>(executable, 0u);
for (std::size_t index = 0; index < 8u; ++index) {
  output[index] =
      sample[index] + 0.05f * prefix[index % 2u] + 0.01f * step;
}
return vlaforge_status_ok();
""".strip(),
        ),
        "decode_action_chunk": CppRegionDefinition(
            "decode_action_chunk",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 8u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 8u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "decode");
}
const auto* input = Input<float>(executable, 0u);
auto* output = Output<float>(executable, 0u);
for (std::size_t index = 0; index < 8u; ++index) {
  output[index] = std::max(-1.0f, std::min(1.0f, input[index]));
}
return vlaforge_status_ok();
""".strip(),
        ),
        "queue_is_empty": CppRegionDefinition(
            "queue_is_empty",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_I32, 1u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_BOOL, 1u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "empty");
}
Output<std::uint8_t>(executable, 0u)[0] =
    Input<std::int32_t>(executable, 0u)[0] >= 4 ? 1u : 0u;
return vlaforge_status_ok();
""".strip(),
        ),
        "queue_select": CppRegionDefinition(
            "queue_select",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 8u) ||
    !CheckTensor(executable->inputs[1], VLAFORGE_DTYPE_I32, 1u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 2u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "select");
}
const auto* queue = Input<float>(executable, 0u);
const auto cursor = Input<std::int32_t>(executable, 1u)[0];
auto* output = Output<float>(executable, 0u);
output[0] = queue[cursor * 2];
output[1] = queue[cursor * 2 + 1];
return vlaforge_status_ok();
""".strip(),
        ),
        "queue_advance": CppRegionDefinition(
            "queue_advance",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_I32, 1u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_I32, 1u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "advance");
}
Output<std::int32_t>(executable, 0u)[0] =
    Input<std::int32_t>(executable, 0u)[0] + 1;
return vlaforge_status_ok();
""".strip(),
        ),
        "queue_zero": CppRegionDefinition(
            "queue_zero",
            """
if (!CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_I32, 1u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION, "zero");
}
Output<std::int32_t>(executable, 0u)[0] = 0;
return vlaforge_status_ok();
""".strip(),
        ),
    }


def smolvla_fixture_validators() -> dict[str, CppValidatorDefinition]:
    return {
        "finite_action": CppValidatorDefinition(
            "finite_action",
            """
if (data == nullptr || size_bytes != 2u * sizeof(float)) {
  return false;
}
const auto* action = static_cast<const float*>(data);
return std::isfinite(action[0]) && std::isfinite(action[1]);
""".strip(),
        )
    }


def smolvla_fixture_runner_source() -> str:
    return r"""
#include "session_generated.h"

#include <cstdint>
#include <cstdio>

namespace {

VLAForgeBoundTensor Tensor(void* data, std::uint64_t bytes,
                           const std::int64_t* shape) {
  return VLAForgeBoundTensor{
      sizeof(VLAForgeBoundTensor),
      {data, bytes, shape, 1u, VLAFORGE_DTYPE_F32,
       {VLAFORGE_DEVICE_CPU, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS, 1u};
}

VLAForgeInputStamp Stamp(std::uint64_t revision) {
  VLAForgeInputStamp stamp{};
  stamp.struct_size = sizeof(VLAForgeInputStamp);
  stamp.has_revision = 1u;
  stamp.revision = revision;
  return stamp;
}

}  // namespace

int main() {
  vlaforge_generated::ModelSession typed;
  VLAForgeSession* generic = nullptr;
  if (vlaforge_model_session_create(&generic).code != VLAFORGE_STATUS_OK) {
    return 1;
  }
  const auto* api = vlaforge_model_session_api();
  if (vlaforge_session_api_validate(
          api, vlaforge_generated::kSchemaDigest,
          VLAFORGE_SCHEMA_DIGEST_HEX_SIZE).code != VLAFORGE_STATUS_OK) {
    return 2;
  }
  const std::int64_t image_shape[] = {2};
  for (std::uint64_t index = 0; index < 7; ++index) {
    const auto run_index = index == 6u ? 0u : index;
    if (index == 6u) {
      if (!typed.ResetEpisode(1u).ok() ||
          api->reset_episode(generic, 1u).code != VLAFORGE_STATUS_OK) {
        return 3;
      }
    }
    float image[] = {
        static_cast<float>(0.25 + (run_index / 4) * 0.1), -0.5f};
    auto view = Tensor(image, sizeof(image), image_shape);
    auto stamp = Stamp(100u + run_index / 4u);
    vlaforge_generated::ModelInputs inputs{};
    inputs.image = view;
    inputs.image_stamp = stamp;
    vlaforge_generated::ModelOutputs outputs{};
    if (!typed.Run(inputs, &outputs).ok()) {
      return 4;
    }
    if (api->bind_tensor(generic, 0u, &view, &stamp).code !=
            VLAFORGE_STATUS_OK ||
        api->run(generic).code != VLAFORGE_STATUS_OK) {
      return 5;
    }
    VLAForgeBoundTensor generic_output{};
    if (api->read_output_tensor(generic, 0u, &generic_output).code !=
        VLAFORGE_STATUS_OK) {
      return 6;
    }
    const auto* typed_action =
        static_cast<const float*>(outputs.action.tensor.data);
    const auto* generic_action =
        static_cast<const float*>(generic_output.tensor.data);
    std::printf("OUTPUT,%llu,%.9g,%.9g,%.9g,%.9g\n",
                static_cast<unsigned long long>(index),
                static_cast<double>(typed_action[0]),
                static_cast<double>(typed_action[1]),
                static_cast<double>(generic_action[0]),
                static_cast<double>(generic_action[1]));
  }
  api->destroy(generic);
  return 0;
}
""".lstrip()


def driving_diffusion_regions() -> dict[str, CppRegionDefinition]:
    """CPU Regions for the two-step K-candidate driving fixture."""

    return {
        "diffusion_condition": CppRegionDefinition(
            "diffusion_condition",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 4u) ||
    !CheckTensor(executable->inputs[1], VLAFORGE_DTYPE_F32, 32u) ||
    !CheckTensor(executable->inputs[2], VLAFORGE_DTYPE_I32, 1u) ||
    !CheckTensor(executable->inputs[3], VLAFORGE_DTYPE_F32, 3u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 4u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "diffusion condition binding mismatch");
}
const auto* scene = Input<float>(executable, 0u);
const auto* agents = Input<float>(executable, 1u);
const auto count = Input<std::int32_t>(executable, 2u)[0];
const auto* route = Input<float>(executable, 3u);
auto* condition = Output<float>(executable, 0u);
float crowd = 0.0f;
for (std::int32_t index = 0; index < count; ++index) {
  crowd += agents[static_cast<std::size_t>(index) * 4u];
}
condition[0] = scene[0];
condition[1] = route[0];
condition[2] = route[1];
condition[3] = crowd;
return vlaforge_status_ok();
""".strip(),
        ),
        "initialize_candidates": CppRegionDefinition(
            "initialize_candidates",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 4u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 36u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "candidate initialization binding mismatch");
}
const auto* condition = Input<float>(executable, 0u);
auto* candidates = Output<float>(executable, 0u);
for (std::size_t candidate = 0; candidate < 3u; ++candidate) {
  for (std::size_t step = 0; step < 6u; ++step) {
    const auto offset = candidate * 12u + step * 2u;
    candidates[offset] = static_cast<float>(step) * 0.5f;
    candidates[offset + 1u] =
        condition[2] + static_cast<float>(candidate) * 0.2f;
  }
}
return vlaforge_status_ok();
""".strip(),
        ),
        "denoise_candidates": CppRegionDefinition(
            "denoise_candidates",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 4u) ||
    !CheckTensor(executable->inputs[1], VLAFORGE_DTYPE_F32, 36u) ||
    !CheckTensor(executable->inputs[2], VLAFORGE_DTYPE_I64, 1u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 36u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "candidate denoise binding mismatch");
}
const auto* condition = Input<float>(executable, 0u);
const auto* candidates = Input<float>(executable, 1u);
const auto step = Input<std::int64_t>(executable, 2u)[0];
auto* output = Output<float>(executable, 0u);
const float scale = 0.5f / static_cast<float>(step + 1);
for (std::size_t index = 0; index < 36u; index += 2u) {
  output[index] = candidates[index] + scale * condition[0];
  output[index + 1u] = candidates[index + 1u] - scale * condition[3];
}
return vlaforge_status_ok();
""".strip(),
        ),
        "score_candidates": CppRegionDefinition(
            "score_candidates",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 36u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 3u) ||
    !CheckTensor(executable->outputs[1], VLAFORGE_DTYPE_F32, 12u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "candidate scoring binding mismatch");
}
const auto* candidates = Input<float>(executable, 0u);
auto* scores = Output<float>(executable, 0u);
auto* trajectory = Output<float>(executable, 1u);
std::size_t best = 0u;
for (std::size_t candidate = 0; candidate < 3u; ++candidate) {
  float score = 0.0f;
  for (std::size_t step = 0; step < 6u; ++step) {
    score -= std::fabs(candidates[candidate * 12u + step * 2u + 1u]);
  }
  scores[candidate] = score;
  if (candidate == 0u || score > scores[best]) {
    best = candidate;
  }
}
for (std::size_t index = 0; index < 12u; ++index) {
  trajectory[index] = candidates[best * 12u + index];
}
return vlaforge_status_ok();
""".strip(),
        ),
    }


def driving_diffusion_validators() -> dict[str, CppValidatorDefinition]:
    return {
        "finite_trajectory": CppValidatorDefinition(
            "finite_trajectory",
            """
if (data == nullptr || size_bytes != 12u * sizeof(float)) {
  return false;
}
const auto* trajectory = static_cast<const float*>(data);
for (std::size_t index = 0; index < 12u; ++index) {
  if (!std::isfinite(trajectory[index])) {
    return false;
  }
}
return true;
""".strip(),
        )
    }


def driving_diffusion_runner_source() -> str:
    """No-Python typed/generic runner for candidates, scores and trajectory."""

    return r"""
#include "session_generated.h"

#include <cstdint>
#include <cstdio>

namespace {

VLAForgeBoundTensor Tensor(void* data, std::uint64_t bytes,
                           const std::int64_t* shape, std::uint32_t rank) {
  return VLAForgeBoundTensor{
      sizeof(VLAForgeBoundTensor),
      {data, bytes, shape, rank, VLAFORGE_DTYPE_F32,
       {VLAFORGE_DEVICE_CPU, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS, 1u};
}

VLAForgeScalarValue I32(std::int32_t value) {
  VLAForgeScalarValue scalar{};
  scalar.struct_size = sizeof(VLAForgeScalarValue);
  scalar.dtype = VLAFORGE_DTYPE_I32;
  scalar.value.i32 = value;
  return scalar;
}

VLAForgeInputStamp Stamp(std::uint64_t revision) {
  VLAForgeInputStamp stamp{};
  stamp.struct_size = sizeof(VLAForgeInputStamp);
  stamp.has_revision = 1u;
  stamp.revision = revision;
  return stamp;
}

void Trace(void*, const vlaforge::runtime::TraceEvent* event) {
  std::printf(
      "TRACE,%u,%u,%u,%llu,%llu,%llu,%llu,%llu\n",
      static_cast<unsigned>(event->kind),
      static_cast<unsigned>(event->task_id),
      static_cast<unsigned>(event->subject_id),
      static_cast<unsigned long long>(event->logical_version),
      static_cast<unsigned long long>(event->transaction_id),
      static_cast<unsigned long long>(event->episode),
      static_cast<unsigned long long>(event->run),
      static_cast<unsigned long long>(event->revision));
}

void Emit(std::uint64_t run, std::uint32_t output,
          const float* typed, const float* generic, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    std::printf("VALUE,%llu,%u,%zu,%.9g,%.9g\n",
                static_cast<unsigned long long>(run),
                static_cast<unsigned>(output), index,
                static_cast<double>(typed[index]),
                static_cast<double>(generic[index]));
  }
}

}  // namespace

int main() {
  vlaforge_generated::ModelSession typed;
  typed.SetTraceSink({nullptr, &Trace});
  VLAForgeSession* generic = nullptr;
  if (vlaforge_model_session_create(&generic).code != VLAFORGE_STATUS_OK) {
    return 1;
  }
  const auto* api = vlaforge_model_session_api();
  const std::int64_t scene_shape[] = {4};
  const std::int64_t agent_shape[] = {8, 4};
  const std::int64_t route_shape[] = {3};
  float scene[] = {0.2f, 0.1f, 0.0f, 0.3f};
  float route[] = {1.0f, 0.2f, 0.0f};
  float agents[32]{};
  for (std::size_t index = 0; index < 8u; ++index) {
    agents[index * 4u] = static_cast<float>(index) * 0.1f;
    agents[index * 4u + 3u] = 1.0f;
  }
  auto scene_view = Tensor(
      scene, sizeof(scene), scene_shape, 1u);
  auto agent_view = Tensor(
      agents, sizeof(agents), agent_shape, 2u);
  auto route_view = Tensor(
      route, sizeof(route), route_shape, 1u);
  auto count = I32(3);

  for (std::uint64_t run = 0; run < 3u; ++run) {
    auto stamp = Stamp(run < 2u ? 30u : 31u);
    vlaforge_generated::ModelInputs inputs{};
    inputs.scene_feature = scene_view;
    inputs.scene_feature_stamp = stamp;
    inputs.agent_features = agent_view;
    inputs.agent_features_stamp = stamp;
    inputs.agent_valid_count = count;
    inputs.agent_valid_count_stamp = stamp;
    inputs.route_command = route_view;
    inputs.route_command_stamp = stamp;
    vlaforge_generated::ModelOutputs outputs{};
    if (!typed.Run(inputs, &outputs).ok()) {
      return 2;
    }
    if (api->bind_tensor(generic, 0u, &scene_view, &stamp).code !=
            VLAFORGE_STATUS_OK ||
        api->bind_tensor(generic, 1u, &agent_view, &stamp).code !=
            VLAFORGE_STATUS_OK ||
        api->bind_scalar(generic, 2u, &count, &stamp).code !=
            VLAFORGE_STATUS_OK ||
        api->bind_tensor(generic, 3u, &route_view, &stamp).code !=
            VLAFORGE_STATUS_OK ||
        api->run(generic).code != VLAFORGE_STATUS_OK) {
      return 3;
    }
    VLAForgeBoundTensor generic_candidates{};
    VLAForgeBoundTensor generic_scores{};
    VLAForgeBoundTensor generic_trajectory{};
    if (api->read_output_tensor(
            generic, 0u, &generic_candidates).code != VLAFORGE_STATUS_OK ||
        api->read_output_tensor(
            generic, 1u, &generic_scores).code != VLAFORGE_STATUS_OK ||
        api->read_output_tensor(
            generic, 2u, &generic_trajectory).code != VLAFORGE_STATUS_OK) {
      return 4;
    }
    Emit(run, 0u,
         static_cast<const float*>(
             outputs.candidate_trajectories.tensor.data),
         static_cast<const float*>(generic_candidates.tensor.data), 36u);
    Emit(run, 1u,
         static_cast<const float*>(outputs.candidate_scores.tensor.data),
         static_cast<const float*>(generic_scores.tensor.data), 3u);
    Emit(run, 2u,
         static_cast<const float*>(outputs.trajectory.tensor.data),
         static_cast<const float*>(generic_trajectory.tensor.data), 12u);
  }
  api->destroy(generic);
  return 0;
}
""".lstrip()


def hybrid_external_feature_regions() -> dict[str, CppRegionDefinition]:
    """CPU C++ implementations for the HybridExternalFeature fixture.

    The first Region is intentionally customer-style preprocessing: its
    Tensor-only boundary is the same contract a dynamically loaded Region
    executable implements, while this fixture links it directly so the L4
    runner remains deterministic and dependency-free.
    """

    return {
        "external_bev_preprocess": CppRegionDefinition(
            "external_bev_preprocess",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 16u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 4u)) {
  return vlaforge_status_error(
      VLAFORGE_STATUS_FAILED_PRECONDITION,
      "external BEV preprocessing binding mismatch");
}
const auto* bev = Input<float>(executable, 0u);
auto* token = Output<float>(executable, 0u);
for (std::size_t column = 0; column < 4u; ++column) {
  token[column] = 0.0f;
  for (std::size_t row = 0; row < 4u; ++row) {
    token[column] += bev[row * 4u + column];
  }
}
return vlaforge_status_ok();
""".strip(),
        ),
        "hybrid_planner": CppRegionDefinition(
            "hybrid_planner",
            """
if (!CheckTensor(executable->inputs[0], VLAFORGE_DTYPE_F32, 4u) ||
    !CheckTensor(executable->inputs[1], VLAFORGE_DTYPE_F32, 18u) ||
    !CheckTensor(executable->inputs[2], VLAFORGE_DTYPE_I32, 1u) ||
    !CheckTensor(executable->inputs[3], VLAFORGE_DTYPE_F32, 3u) ||
    !CheckTensor(executable->outputs[0], VLAFORGE_DTYPE_F32, 12u) ||
    !CheckTensor(executable->outputs[1], VLAFORGE_DTYPE_F32, 12u) ||
    !CheckTensor(executable->outputs[2], VLAFORGE_DTYPE_I64, 1u)) {
  return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                               "hybrid planner binding mismatch");
}
const auto* bev = Input<float>(executable, 0u);
const auto* agents = Input<float>(executable, 1u);
const auto count = Input<std::int32_t>(executable, 2u)[0];
const auto* route = Input<float>(executable, 3u);
auto* trajectory = Output<float>(executable, 0u);
auto* prediction = Output<float>(executable, 1u);
for (std::size_t step = 0; step < 6u; ++step) {
  trajectory[step * 2u] =
      static_cast<float>(step) * 0.4f + bev[0] * 0.01f;
  trajectory[step * 2u + 1u] = route[1] + bev[1] * 0.01f;
  if (count > 0) {
    const auto agent = step % static_cast<std::size_t>(count);
    prediction[step * 2u] =
        agents[agent * 3u] + static_cast<float>(step) * 0.1f;
    prediction[step * 2u + 1u] = agents[agent * 3u + 1u];
  } else {
    prediction[step * 2u] = static_cast<float>(step) * 0.1f;
    prediction[step * 2u + 1u] = 0.0f;
  }
}
const auto token_value =
    static_cast<std::int64_t>(std::fabs(
        bev[0] + bev[1] + bev[2] + bev[3] + route[0]) * 10.0f);
Output<std::int64_t>(executable, 2u)[0] = token_value % 1024;
return vlaforge_status_ok();
""".strip(),
        ),
    }


def hybrid_external_feature_validators(
) -> dict[str, CppValidatorDefinition]:
    return {
        "finite_trajectory": CppValidatorDefinition(
            "finite_trajectory",
            """
if (data == nullptr || size_bytes != 12u * sizeof(float)) {
  return false;
}
const auto* trajectory = static_cast<const float*>(data);
for (std::size_t index = 0; index < 12u; ++index) {
  if (!std::isfinite(trajectory[index])) {
    return false;
  }
}
return true;
""".strip(),
        )
    }


def hybrid_external_feature_runner_source() -> str:
    """No-Python C++ runner covering typed/generic input contracts."""

    return r"""
#include "session_generated.h"

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

VLAForgeBoundTensor Tensor(void* data, std::uint64_t bytes,
                           const std::int64_t* shape, std::uint32_t rank,
                           VLAForgeDType dtype,
                           VLAForgeDeviceKind device =
                               VLAFORGE_DEVICE_CPU,
                           VLAForgeLayout layout =
                               VLAFORGE_LAYOUT_CONTIGUOUS) {
  return VLAForgeBoundTensor{
      sizeof(VLAForgeBoundTensor),
      {data, bytes, shape, rank, dtype, {device, 0}},
      layout,
      1u};
}

VLAForgeScalarValue I32(std::int32_t value) {
  VLAForgeScalarValue scalar{};
  scalar.struct_size = sizeof(VLAForgeScalarValue);
  scalar.dtype = VLAFORGE_DTYPE_I32;
  scalar.value.i32 = value;
  return scalar;
}

VLAForgeInputStamp Stamp(std::uint64_t revision) {
  VLAForgeInputStamp stamp{};
  stamp.struct_size = sizeof(VLAForgeInputStamp);
  stamp.has_revision = 1u;
  stamp.revision = revision;
  return stamp;
}

struct TraceCounts {
  std::uint32_t hit = 0;
  std::uint32_t miss = 0;
};

void CountTrace(void* context,
                const vlaforge::runtime::TraceEvent* event) {
  auto* counts = static_cast<TraceCounts*>(context);
  if (event->kind == vlaforge::runtime::TraceKind::kCacheHit) {
    ++counts->hit;
  } else if (event->kind == vlaforge::runtime::TraceKind::kCacheMiss) {
    ++counts->miss;
  }
  std::printf(
      "TRACE,%u,%u,%u,%llu,%llu,%llu,%llu,%llu\n",
      static_cast<unsigned>(event->kind),
      static_cast<unsigned>(event->task_id),
      static_cast<unsigned>(event->subject_id),
      static_cast<unsigned long long>(event->logical_version),
      static_cast<unsigned long long>(event->transaction_id),
      static_cast<unsigned long long>(event->episode),
      static_cast<unsigned long long>(event->run),
      static_cast<unsigned long long>(event->revision));
}

bool CheckFailure(VLAForgeStatus status, VLAForgeStatusCode expected) {
  return status.code == expected;
}

}  // namespace

int main() {
  vlaforge_generated::ModelSession typed;
  TraceCounts trace_counts{};
  typed.SetTraceSink({&trace_counts, &CountTrace});

  VLAForgeSession* generic = nullptr;
  if (vlaforge_model_session_create(&generic).code != VLAFORGE_STATUS_OK) {
    return 1;
  }
  const auto* api = vlaforge_model_session_api();
  if (vlaforge_session_api_validate(
          api, vlaforge_generated::kSchemaDigest,
          VLAFORGE_SCHEMA_DIGEST_HEX_SIZE).code != VLAFORGE_STATUS_OK) {
    return 2;
  }
  std::array<char, VLAFORGE_SCHEMA_DIGEST_HEX_SIZE> wrong_digest{};
  wrong_digest.fill('f');
  if (!CheckFailure(
          vlaforge_session_api_validate(
              api, wrong_digest.data(), wrong_digest.size()),
          VLAFORGE_STATUS_FAILED_PRECONDITION)) {
    return 3;
  }

  const std::int64_t bev_shape[] = {4, 4};
  const std::int64_t route_shape[] = {3};
  const std::int64_t agent_shape[] = {6, 3};
  const std::int64_t wrong_shape[] = {2, 8};
  float bev[16];
  for (std::size_t row = 0; row < 4u; ++row) {
    for (std::size_t column = 0; column < 4u; ++column) {
      bev[row * 4u + column] =
          static_cast<float>(row + column) * 0.1f;
    }
  }
  float route[] = {1.0f, 0.3f, 0.0f};
  float agents[18] = {
      0.5f, 0.1f, 0.0f, 1.5f, -0.2f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  auto bev_view = Tensor(
      bev, sizeof(bev), bev_shape, 2u, VLAFORGE_DTYPE_F32);
  auto route_view = Tensor(
      route, sizeof(route), route_shape, 1u, VLAFORGE_DTYPE_F32);
  auto agent_view = Tensor(
      agents, sizeof(agents), agent_shape, 2u, VLAFORGE_DTYPE_F32);
  auto stamp = Stamp(40u);

  auto bad_shape = Tensor(
      bev, sizeof(bev), wrong_shape, 2u, VLAFORGE_DTYPE_F32);
  auto bad_dtype = Tensor(
      bev, sizeof(bev), bev_shape, 2u, VLAFORGE_DTYPE_I32);
  auto bad_device = Tensor(
      bev, sizeof(bev), bev_shape, 2u, VLAFORGE_DTYPE_F32,
      VLAFORGE_DEVICE_CUDA);
  auto bad_layout = Tensor(
      bev, sizeof(bev), bev_shape, 2u, VLAFORGE_DTYPE_F32,
      VLAFORGE_DEVICE_CPU, VLAFORGE_LAYOUT_NHWC);
  if (!CheckFailure(api->bind_tensor(generic, 0u, &bad_shape, &stamp),
                    VLAFORGE_STATUS_INVALID_ARGUMENT) ||
      !CheckFailure(api->bind_tensor(generic, 0u, &bad_dtype, &stamp),
                    VLAFORGE_STATUS_INVALID_ARGUMENT) ||
      !CheckFailure(api->bind_tensor(generic, 0u, &bad_device, &stamp),
                    VLAFORGE_STATUS_INVALID_ARGUMENT) ||
      !CheckFailure(api->bind_tensor(generic, 0u, &bad_layout, &stamp),
                    VLAFORGE_STATUS_INVALID_ARGUMENT) ||
      !CheckFailure(api->bind_tensor(generic, 99u, &bev_view, &stamp),
                    VLAFORGE_STATUS_INVALID_ARGUMENT) ||
      !CheckFailure(api->bind_tensor(generic, 2u, &bev_view, &stamp),
                    VLAFORGE_STATUS_INVALID_ARGUMENT)) {
    return 4;
  }
  auto out_of_range = I32(7);
  if (!CheckFailure(
          api->bind_scalar(generic, 2u, &out_of_range, &stamp),
          VLAFORGE_STATUS_INVALID_ARGUMENT)) {
    return 5;
  }

  VLAForgeSession* missing_required = nullptr;
  if (vlaforge_model_session_create(&missing_required).code !=
          VLAFORGE_STATUS_OK ||
      api->bind_tensor(
          missing_required, 3u, &route_view, &stamp).code !=
          VLAFORGE_STATUS_OK ||
      !CheckFailure(api->run(missing_required),
                    VLAFORGE_STATUS_FAILED_PRECONDITION)) {
    return 6;
  }
  api->destroy(missing_required);

  for (std::uint64_t index = 0; index < 4u; ++index) {
    const auto revision =
        index < 2u ? 40u : (index == 2u ? 41u : 42u);
    stamp = Stamp(revision);
    vlaforge_generated::ModelInputs inputs{};
    inputs.external_bev = bev_view;
    inputs.external_bev_stamp = stamp;
    inputs.route_command = route_view;
    inputs.route_command_stamp = stamp;
    if (index == 3u) {
      inputs.has_agent_features = true;
      inputs.agent_features = agent_view;
      inputs.agent_features_stamp = stamp;
      inputs.has_agent_valid_count = true;
      inputs.agent_valid_count = I32(2);
      inputs.agent_valid_count_stamp = stamp;
    }
    vlaforge_generated::ModelOutputs outputs{};
    if (!typed.Run(inputs, &outputs).ok()) {
      return 7;
    }

    if (api->bind_tensor(generic, 0u, &bev_view, &stamp).code !=
            VLAFORGE_STATUS_OK ||
        api->bind_tensor(generic, 3u, &route_view, &stamp).code !=
            VLAFORGE_STATUS_OK) {
      return 8;
    }
    if (index == 3u &&
        (api->bind_tensor(generic, 1u, &agent_view, &stamp).code !=
             VLAFORGE_STATUS_OK ||
         api->bind_scalar(
             generic, 2u, &inputs.agent_valid_count, &stamp).code !=
             VLAFORGE_STATUS_OK)) {
      return 9;
    }
    if (api->run(generic).code != VLAFORGE_STATUS_OK) {
      return 10;
    }
    VLAForgeBoundTensor generic_trajectory{};
    VLAForgeBoundTensor generic_prediction{};
    VLAForgeScalarValue generic_token{};
    if (api->read_output_tensor(
            generic, 0u, &generic_trajectory).code !=
            VLAFORGE_STATUS_OK ||
        api->read_output_tensor(
            generic, 1u, &generic_prediction).code !=
            VLAFORGE_STATUS_OK ||
        api->read_output_scalar(
            generic, 2u, &generic_token).code !=
            VLAFORGE_STATUS_OK) {
      return 11;
    }
    const auto* typed_trajectory =
        static_cast<const float*>(outputs.trajectory.tensor.data);
    const auto* typed_prediction =
        static_cast<const float*>(outputs.agent_prediction.tensor.data);
    const auto* generic_trajectory_data =
        static_cast<const float*>(generic_trajectory.tensor.data);
    const auto* generic_prediction_data =
        static_cast<const float*>(generic_prediction.tensor.data);
    std::printf(
        "OUTPUT,%llu,%.9g,%.9g,%.9g,%.9g,%lld,"
        "%.9g,%.9g,%.9g,%.9g,%lld\n",
        static_cast<unsigned long long>(index),
        static_cast<double>(typed_trajectory[0]),
        static_cast<double>(typed_trajectory[1]),
        static_cast<double>(typed_prediction[0]),
        static_cast<double>(typed_prediction[1]),
        static_cast<long long>(outputs.vqa_token.value.i64),
        static_cast<double>(generic_trajectory_data[0]),
        static_cast<double>(generic_trajectory_data[1]),
        static_cast<double>(generic_prediction_data[0]),
        static_cast<double>(generic_prediction_data[1]),
        static_cast<long long>(generic_token.value.i64));
  }

  // A successful Run consumes all push bindings. This is the concrete
  // borrowed-until-Run-returns contract: the Session neither retains nor frees
  // the caller-owned input buffers.
  if (!CheckFailure(api->run(generic),
                    VLAFORGE_STATUS_FAILED_PRECONDITION)) {
    return 12;
  }
  if (trace_counts.hit != 1u || trace_counts.miss != 3u) {
    return 13;
  }
  std::printf("CACHE,%u,%u\n", trace_counts.hit, trace_counts.miss);
  api->destroy(generic);
  return 0;
}
""".lstrip()
