"""C++ CPU artifact bodies and runner for deterministic offline fixtures."""

from __future__ import annotations

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
    return """
#include "session_generated.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

struct TraceCollector {
  std::array<vlaforge::runtime::TraceEvent, 256> events{};
  std::size_t count = 0;
};

void Collect(void* context, const vlaforge::runtime::TraceEvent* event) {
  auto* collector = static_cast<TraceCollector*>(context);
  if (collector->count < collector->events.size()) {
    collector->events[collector->count++] = *event;
  }
}

}  // namespace

int main() {
  vlaforge_generated::GeneratedSession session;
  TraceCollector trace;
  session.SetTraceSink({&trace, &Collect});
  const std::int64_t image_shape[] = {2};
  const std::int64_t instruction_shape[] = {3};

  for (std::uint64_t index = 0; index < 3; ++index) {
    float image[] = {
        static_cast<float>(0.1 * static_cast<double>(index)),
        static_cast<float>(0.4 - 0.05 * static_cast<double>(index)),
    };
    std::int64_t instruction[] = {
        1, static_cast<std::int64_t>(2 + index), 3};
    const vlaforge::runtime::Epoch observation{
        0u, index, index * 50000000u, 0u};
    const vlaforge::runtime::Epoch tick{
        1u, index, index * 50000000u, 0u};
    const vlaforge::runtime::TensorView image_view{
        image, image_shape, 1u, vlaforge::runtime::ScalarType::kF32,
        vlaforge::runtime::DeviceType::kCpu, 0u, sizeof(image)};
    const vlaforge::runtime::TensorView instruction_view{
        instruction, instruction_shape, 1u,
        vlaforge::runtime::ScalarType::kI64,
        vlaforge::runtime::DeviceType::kCpu, 0u, sizeof(instruction)};
    if (!session.BindInput(0u, image_view, observation).ok() ||
        !session.BindInput(1u, instruction_view, observation).ok() ||
        !session.RunTick(tick).ok()) {
      return 1;
    }
    vlaforge::runtime::CommittedAction action;
    if (!session.ReadCommittedAction(&action).ok() ||
        action.size_bytes != 2u * sizeof(float)) {
      return 2;
    }
    const auto* values = static_cast<const float*>(action.data);
    std::printf("ACTION,%llu,%.9g,%.9g\\n",
                static_cast<unsigned long long>(index),
                static_cast<double>(values[0]),
                static_cast<double>(values[1]));
  }

  for (std::size_t index = 0; index < trace.count; ++index) {
    const auto& event = trace.events[index];
    std::printf(
        "TRACE,%u,%u,%u,%llu,%llu,%u,%llu,%llu,%llu\\n",
        static_cast<unsigned>(event.kind), event.task_id, event.state_id,
        static_cast<unsigned long long>(event.logical_version),
        static_cast<unsigned long long>(event.transaction_id),
        event.epoch.clock_id,
        static_cast<unsigned long long>(event.epoch.sequence),
        static_cast<unsigned long long>(event.epoch.timestamp_ns),
        static_cast<unsigned long long>(event.epoch.episode));
  }
  return 0;
}
""".lstrip()
