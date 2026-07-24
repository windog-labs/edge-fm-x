"""Generated no-Python OpenVLA runner for a shared TorchScript archive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vlaforge.adapters import build_real_openvla_action_program
from vlaforge.codegen.model import GeneratedSources
from vlaforge.codegen.real_aoti import AotiTensorSpec
from vlaforge.plan import lower_to_plan, physicalize_plan


@dataclass(frozen=True, slots=True)
class OpenVLATorchScriptSpec:
    prefill_inputs: tuple[AotiTensorSpec, ...]
    prefill_outputs: tuple[AotiTensorSpec, ...]
    decode_inputs: tuple[AotiTensorSpec, ...]
    decode_outputs: tuple[AotiTensorSpec, ...]
    action_tokens: AotiTensorSpec
    action: AotiTensorSpec

    def __post_init__(self) -> None:
        if len(self.prefill_inputs) != 3:
            raise ValueError("OpenVLA prefill requires three inputs")
        if len(self.prefill_outputs) != 65:
            raise ValueError("OpenVLA prefill requires logits plus 64 KV tensors")
        if len(self.decode_inputs) != 65 or len(self.decode_outputs) != 65:
            raise ValueError("OpenVLA decode requires token plus 64 KV tensors")
        if self.action_tokens.dtype != "i64":
            raise ValueError("OpenVLA action tokens must be i64")
        if self.action.dtype != "f64" or len(self.action.shape) != 1:
            raise ValueError("OpenVLA action must be a one-dimensional f64 tensor")
        if self.action_tokens.shape != (1, self.action.shape[0]):
            raise ValueError("OpenVLA token and action dimensions must agree")
        if self.action.shape[0] < 2:
            raise ValueError("OpenVLA requires at least two action tokens")
        if self.prefill_outputs[1].shape[2] + 1 != self.decode_outputs[1].shape[2]:
            raise ValueError("OpenVLA decode must append exactly one KV position")
        cache_shape = self.prefill_outputs[1].shape
        if any(item.shape != cache_shape for item in self.prefill_outputs[1:]):
            raise ValueError("OpenVLA prefill KV tensors must share one shape")
        decode_shape = self.decode_outputs[1].shape
        if any(item.shape != decode_shape for item in self.decode_outputs[1:]):
            raise ValueError("OpenVLA decode KV tensors must share one shape")

    @property
    def action_dim(self) -> int:
        return self.action.shape[0]

    @property
    def decode_steps(self) -> int:
        return self.action_dim - 1


def openvla_spec_from_capture_reports(
    prefill: Mapping[str, Any],
    decode: Mapping[str, Any],
    detokenize: Mapping[str, Any],
) -> OpenVLATorchScriptSpec:
    return OpenVLATorchScriptSpec(
        prefill_inputs=tuple(_capture_spec(item) for item in prefill["inputs"]),
        prefill_outputs=tuple(_capture_spec(item) for item in prefill["outputs"]),
        decode_inputs=tuple(_capture_spec(item) for item in decode["inputs"]),
        decode_outputs=tuple(_capture_spec(item) for item in decode["outputs"]),
        action_tokens=_capture_spec(detokenize["inputs"][0]),
        action=_capture_spec(detokenize["outputs"][0]),
    )


def generate_real_openvla_torchscript_runner(
    spec: OpenVLATorchScriptSpec,
) -> GeneratedSources:
    module = build_real_openvla_action_program(action_dim=spec.action_dim)
    plan = physicalize_plan(lower_to_plan(module))
    tasks = {
        "image": _task_id(plan, "vla.sample_input", stream="image"),
        "tokens": _task_id(
            plan, "vla.sample_input", stream="instruction_tokens"
        ),
        "mask": _task_id(
            plan, "vla.sample_input", stream="instruction_mask"
        ),
        "begin": _task_id(plan, "vla.txn.begin"),
        "prefill": _task_id(
            plan, "vla.invoke", region="generate_action_tokens_prefill"
        ),
        "decode": _task_id(
            plan,
            "vla.invoke",
            region="generate_action_tokens_decode_step",
        ),
        "extract": _task_id(
            plan, "vla.invoke", region="extract_action_tokens"
        ),
        "detokenize": _task_id(
            plan, "vla.invoke", region="detokenize_action"
        ),
        "validate": _task_id(plan, "vla.validate"),
        "action": _task_id(plan, "vla.action.create"),
        "commit": _task_id(plan, "vla.txn.commit"),
        "publish": _task_id(plan, "vla.action.publish"),
    }
    return GeneratedSources(
        (
            ("CMakeLists.txt", _cmake()),
            ("runner.cpp", _source(spec, tasks)),
        )
    )


def _source(spec: OpenVLATorchScriptSpec, tasks: dict[str, int]) -> str:
    constants = "\n".join(
        f"constexpr std::uint32_t kTask{_camel(name)} = {value}u;"
        for name, value in tasks.items()
    )
    prefix_allocations = "\n".join(
        f"  prefix_outputs.push_back(at::empty({_shape(item.shape)}, "
        f"options.dtype({_at_dtype(item.dtype)})));"
        for item in spec.prefill_outputs
    )
    input_allocations = "\n".join(
        f"  at::Tensor input_{index} = at::empty({_shape(item.shape)}, "
        f"options.dtype({_at_dtype(item.dtype)}));"
        for index, item in enumerate(spec.prefill_inputs)
    )
    input_reads = "\n".join(
        f'  if (!ReadRaw(input_root + "/input_{index}.bin", input_{index})) '
        f"return 4;"
        for index in range(len(spec.prefill_inputs))
    )
    input_ptrs = ", ".join(
        f"&input_{index}" for index in range(len(spec.prefill_inputs))
    )
    cache_count = len(spec.prefill_outputs) - 1
    cache_shape = spec.decode_outputs[1].shape
    prefix_ptrs = "\n".join(
        f"  prefix_output_ptrs.push_back(&prefix_outputs[{index}]);"
        for index in range(len(spec.prefill_outputs))
    )
    task_constants = constants
    return f"""#include "vlaforge/backends/torchscript_region_executable.h"
#include "vlaforge/runtime/action_queue.h"
#include "vlaforge/runtime/state_store.h"
#include "vlaforge/runtime/static_arena.h"
#include "vlaforge/runtime/transaction.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {{

{task_constants}
constexpr std::uint32_t kControlClock = 1u;
constexpr std::uint64_t kControlPeriodNs = 50000000u;
constexpr std::size_t kCacheCount = {cache_count}u;
constexpr std::size_t kDecodeSteps = {spec.decode_steps}u;
constexpr std::size_t kActionDim = {spec.action_dim}u;

struct TraceCollector {{
  std::array<vlaforge::runtime::TraceEvent, 256> events{{}};
  std::size_t count = 0u;
}};

void CollectTrace(void* context,
                  const vlaforge::runtime::TraceEvent* event) {{
  auto* collector = static_cast<TraceCollector*>(context);
  if (collector->count < collector->events.size()) {{
    collector->events[collector->count++] = *event;
  }}
}}

bool Check(VLAForgeStatus status, const char* operation) {{
  if (status.code == VLAFORGE_STATUS_OK) return true;
  std::fprintf(stderr, "%s failed: %.*s\\n", operation,
               static_cast<int>(status.message_size), status.message);
  return false;
}}

bool Check(const vlaforge::runtime::Status& status,
           const char* operation) {{
  if (status.ok()) return true;
  std::fprintf(stderr, "%s failed: %s (subject=%u)\\n", operation,
               status.message, status.subject_id);
  return false;
}}

VLAForgeDType DType(const at::Tensor& tensor) {{
  switch (tensor.scalar_type()) {{
    case at::kBool: return VLAFORGE_DTYPE_BOOL;
    case at::kInt: return VLAFORGE_DTYPE_I32;
    case at::kLong: return VLAFORGE_DTYPE_I64;
    case at::kHalf: return VLAFORGE_DTYPE_F16;
    case at::kBFloat16: return VLAFORGE_DTYPE_BF16;
    case at::kFloat: return VLAFORGE_DTYPE_F32;
    case at::kDouble: return VLAFORGE_DTYPE_F64;
    default: return VLAFORGE_DTYPE_INVALID;
  }}
}}

VLAForgeTensorView View(at::Tensor& tensor) {{
  return VLAForgeTensorView{{
      tensor.data_ptr(), static_cast<std::uint64_t>(tensor.nbytes()),
      tensor.sizes().data(), static_cast<std::uint32_t>(tensor.dim()),
      DType(tensor), {{VLAFORGE_DEVICE_CPU, 0}}}};
}}

class Region final {{
 public:
  ~Region() {{
    if (executable_ != nullptr) api_->destroy(executable_);
  }}
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;
  Region() = default;

  bool Load(std::uint32_t region_id, const std::string& archive,
            const char* method) {{
    api_ = vlaforge_torchscript_region_executable_api();
    if (!Check(vlaforge_region_executable_api_validate(api_),
               "validate TorchScript API")) return false;
    const VLAForgeRegionCreateOptions options{{
        sizeof(VLAForgeRegionCreateOptions),
        VLAFORGE_REGION_EXECUTABLE_ABI_VERSION, region_id,
        {{VLAFORGE_DEVICE_CPU, 0}}}};
    if (!Check(api_->create(&options, &executable_), "create region")) {{
      return false;
    }}
    const std::string artifact_spec = archive + "#" + method;
    const VLAForgeArtifactDescriptor artifact{{
        sizeof(VLAForgeArtifactDescriptor),
        VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
        artifact_spec.data(), artifact_spec.size(), nullptr, 0u}};
    return Check(api_->load(executable_, &artifact), "load region");
  }}

  bool Run(const std::vector<at::Tensor*>& inputs,
           const std::vector<at::Tensor*>& outputs) {{
    for (std::size_t index = 0; index < inputs.size(); ++index) {{
      auto view = View(*inputs[index]);
      if (!Check(api_->bind_input(
                     executable_, static_cast<std::uint32_t>(index), &view),
                 "bind input")) return false;
    }}
    for (std::size_t index = 0; index < outputs.size(); ++index) {{
      auto view = View(*outputs[index]);
      if (!Check(api_->bind_output(
                     executable_, static_cast<std::uint32_t>(index), &view),
                 "bind output")) return false;
    }}
    return Check(api_->run(executable_), "run region") &&
           Check(api_->synchronize(executable_), "synchronize region");
  }}

 private:
  const VLAForgeRegionExecutableApi* api_ = nullptr;
  VLAForgeRegionExecutable* executable_ = nullptr;
}};

void Emit(vlaforge::runtime::TraceSink trace,
          vlaforge::runtime::TraceKind kind,
          std::uint32_t task_id,
          std::uint64_t transaction_id,
          const vlaforge::runtime::Epoch& tick) {{
  vlaforge::runtime::EmitTrace(
      trace, vlaforge::runtime::TraceEvent{{
                 kind, task_id, 0u, 0u, transaction_id, tick}});
}}

bool ReadRaw(const std::string& path, at::Tensor& tensor) {{
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  input.read(static_cast<char*>(tensor.data_ptr()),
             static_cast<std::streamsize>(tensor.nbytes()));
  return input.good() &&
         input.peek() == std::ifstream::traits_type::eof();
}}

bool DumpTensor(std::ofstream& output, const at::Tensor& tensor) {{
  const at::Tensor contiguous = tensor.contiguous();
  output.write(static_cast<const char*>(contiguous.const_data_ptr()),
               static_cast<std::streamsize>(contiguous.nbytes()));
  return output.good();
}}

}}  // namespace

int main(int argc, char** argv) {{
  if (argc != 4) {{
    std::fprintf(
        stderr, "usage: %s OPENVLA.pt INPUT_DIR EVIDENCE.bin\\n", argv[0]);
    return 2;
  }}
  at::set_num_threads(16);
  const auto options = at::TensorOptions().device(at::kCPU);
{input_allocations}
  const std::string input_root(argv[2]);
{input_reads}

  std::vector<at::Tensor> prefix_outputs;
  prefix_outputs.reserve({len(spec.prefill_outputs)}u);
{prefix_allocations}
  std::vector<at::Tensor*> prefix_inputs{{{input_ptrs}}};
  std::vector<at::Tensor*> prefix_output_ptrs;
  prefix_output_ptrs.reserve(prefix_outputs.size());
{prefix_ptrs}

  std::array<at::Tensor, kActionDim> token_steps;
  for (auto& token : token_steps) {{
    token = at::empty({{1, 1}}, options.dtype(at::kLong));
  }}
  std::array<std::vector<at::Tensor>, kDecodeSteps> decode_outputs;
  std::array<std::vector<at::Tensor*>, kDecodeSteps> decode_output_ptrs;
  std::array<std::vector<at::Tensor*>, kDecodeSteps> decode_input_ptrs;
  for (std::size_t step = 0; step < kDecodeSteps; ++step) {{
    auto& bank = decode_outputs[step];
    bank.reserve(kCacheCount + 1u);
    bank.push_back(at::empty(
        {_shape(spec.decode_outputs[0].shape)},
        options.dtype({_at_dtype(spec.decode_outputs[0].dtype)})));
    for (std::size_t index = 0; index < kCacheCount; ++index) {{
      bank.push_back(at::empty(
          {{{cache_shape[0]}, {cache_shape[1]},
            static_cast<std::int64_t>({cache_shape[2]}) +
                static_cast<std::int64_t>(step),
            {cache_shape[3]}}},
          options.dtype({_at_dtype(spec.decode_outputs[1].dtype)})));
    }}
    auto& outputs = decode_output_ptrs[step];
    outputs.reserve(bank.size());
    for (auto& tensor : bank) outputs.push_back(&tensor);
    auto& inputs = decode_input_ptrs[step];
    inputs.reserve(kCacheCount + 1u);
    inputs.push_back(&token_steps[step]);
    const auto& source = step == 0u
        ? prefix_outputs : decode_outputs[step - 1u];
    for (std::size_t index = 1u; index <= kCacheCount; ++index) {{
      inputs.push_back(const_cast<at::Tensor*>(&source[index]));
    }}
  }}
  at::Tensor action_tokens =
      at::empty({{1, static_cast<std::int64_t>(kActionDim)}},
                options.dtype(at::kLong));
  at::Tensor action =
      at::empty({{static_cast<std::int64_t>(kActionDim)}},
                options.dtype(at::kDouble));

  Region prefill;
  Region decode;
  Region detokenize;
  const std::string archive(argv[1]);
  if (!prefill.Load(0u, archive, "prefill") ||
      !decode.Load(1u, archive, "decode") ||
      !detokenize.Load(2u, archive, "detokenize")) return 3;
  std::ofstream evidence(argv[3], std::ios::binary | std::ios::trunc);
  if (!evidence) return 4;

  TraceCollector collector;
  const vlaforge::runtime::TraceSink trace{{&collector, &CollectTrace}};
  vlaforge::runtime::StaticArena arena(0u, 64u);
  vlaforge::runtime::StateStore state_store(arena, nullptr, 0u, trace);
  vlaforge::runtime::Transaction transaction(0u);
  vlaforge::runtime::ActionQueue action_queue(nullptr, nullptr, trace);
  for (std::uint64_t sequence = 0u; sequence < 3u; ++sequence) {{
    const vlaforge::runtime::Epoch tick{{
        kControlClock, sequence, sequence * kControlPeriodNs, 0u}};
    Emit(trace, vlaforge::runtime::TraceKind::kInput,
         kTaskImage, 0u, tick);
    Emit(trace, vlaforge::runtime::TraceKind::kInput,
         kTaskTokens, 0u, tick);
    Emit(trace, vlaforge::runtime::TraceKind::kInput,
         kTaskMask, 0u, tick);
    if (!Check(state_store.Begin(&transaction, tick, kTaskBegin),
               "begin transaction")) return 5;
    if (!prefill.Run(prefix_inputs, prefix_output_ptrs)) return 6;
    Emit(trace, vlaforge::runtime::TraceKind::kRegion,
         kTaskPrefill, transaction.id(), tick);
    token_steps[0].copy_(at::argmax(prefix_outputs[0], -1, true));
    action_tokens.select(1, 0).copy_(token_steps[0].view({{1}}));
    std::printf("TOKEN,%llu,0,%lld\\n",
                static_cast<unsigned long long>(sequence),
                static_cast<long long>(token_steps[0].item<std::int64_t>()));
    if (sequence == 0u) {{
      for (const auto& output : prefix_outputs) {{
        if (!DumpTensor(evidence, output)) return 7;
      }}
    }}
    for (std::size_t step = 0; step < kDecodeSteps; ++step) {{
      if (!decode.Run(
              decode_input_ptrs[step], decode_output_ptrs[step])) return 8;
      Emit(trace, vlaforge::runtime::TraceKind::kRegion,
           kTaskDecode, transaction.id(), tick);
      token_steps[step + 1u].copy_(
          at::argmax(decode_outputs[step][0], -1, true));
      action_tokens.select(1, static_cast<std::int64_t>(step + 1u))
          .copy_(token_steps[step + 1u].view({{1}}));
      std::printf("TOKEN,%llu,%zu,%lld\\n",
                  static_cast<unsigned long long>(sequence), step + 1u,
                  static_cast<long long>(
                      token_steps[step + 1u].item<std::int64_t>()));
      if (sequence == 0u) {{
        for (const auto& output : decode_outputs[step]) {{
          if (!DumpTensor(evidence, output)) return 9;
        }}
      }}
    }}
    Emit(trace, vlaforge::runtime::TraceKind::kRegion,
         kTaskExtract, transaction.id(), tick);
    std::vector<at::Tensor*> detokenize_inputs{{&action_tokens}};
    std::vector<at::Tensor*> detokenize_outputs{{&action}};
    if (!detokenize.Run(detokenize_inputs, detokenize_outputs)) return 10;
    Emit(trace, vlaforge::runtime::TraceKind::kRegion,
         kTaskDetokenize, transaction.id(), tick);
    if (sequence == 0u &&
        (!DumpTensor(evidence, action_tokens) ||
         !DumpTensor(evidence, action))) return 11;
    const bool valid = bool(at::isfinite(action).all().item<bool>());
    Emit(trace, vlaforge::runtime::TraceKind::kValidation,
         kTaskValidate, transaction.id(), tick);
    Emit(trace, vlaforge::runtime::TraceKind::kActionPending,
         kTaskAction, transaction.id(), tick);
    vlaforge::runtime::CommittedAction committed;
    if (!Check(state_store.Commit(
                   &transaction,
                   vlaforge::runtime::PendingAction{{
                       tick, action.const_data_ptr(), action.nbytes()}},
                   valid, kTaskCommit, &committed),
               "commit") ||
        !Check(action_queue.Publish(committed, kTaskPublish),
               "publish")) return 12;
    if (!DumpTensor(evidence, action)) return 13;
    std::printf("ACTION,%llu",
                static_cast<unsigned long long>(sequence));
    const auto* values = action.const_data_ptr<double>();
    for (std::size_t index = 0; index < kActionDim; ++index) {{
      std::printf(",%.17g", values[index]);
    }}
    std::printf("\\n");
  }}
  for (std::size_t index = 0; index < collector.count; ++index) {{
    const auto& event = collector.events[index];
    std::printf(
        "TRACE,%u,%u,%u,%llu,%llu,%u,%llu,%llu,%llu\\n",
        static_cast<unsigned>(event.kind), event.task_id, event.state_id,
        static_cast<unsigned long long>(event.logical_version),
        static_cast<unsigned long long>(event.transaction_id),
        event.epoch.clock_id,
        static_cast<unsigned long long>(event.epoch.sequence),
        static_cast<unsigned long long>(event.epoch.timestamp_ns),
        static_cast<unsigned long long>(event.epoch.episode));
  }}
  return 0;
}}
"""


def _cmake() -> str:
    return """cmake_minimum_required(VERSION 3.18)
project(vlaforge_real_openvla_runner LANGUAGES C CXX)

if(NOT DEFINED VLAFORGE_RUNTIME_ROOT)
  message(FATAL_ERROR "set VLAFORGE_RUNTIME_ROOT to the VLAForge source root")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(VLAFORGE_BUILD_TORCHSCRIPT_BACKEND ON CACHE BOOL "" FORCE)
find_package(Torch REQUIRED CONFIG)
add_subdirectory("${VLAFORGE_RUNTIME_ROOT}"
                 "${CMAKE_CURRENT_BINARY_DIR}/vlaforge_runtime")
add_executable(vlaforge_real_openvla_runner runner.cpp)
target_link_libraries(vlaforge_real_openvla_runner PRIVATE
    vlaforge_torchscript_backend)
target_compile_options(vlaforge_real_openvla_runner PRIVATE
    ${VLAFORGE_TORCH_CXX_FLAGS}
    -Wall -Wextra -Wpedantic -Werror)
"""


def _capture_spec(data: Mapping[str, Any]) -> AotiTensorSpec:
    tensor = data["type"]
    return AotiTensorSpec(
        tuple(int(value) for value in tensor["shape"]),
        str(tensor["dtype"]),
    )


def _task_id(plan: object, opcode: str, **attributes: object) -> int:
    candidates = [
        task
        for task in plan.tasks
        if task.opcode == opcode
        and all(task.attributes.get(key) == value for key, value in attributes.items())
    ]
    if len(candidates) != 1:
        raise ValueError(f"missing or ambiguous task {opcode}: {attributes}")
    return int(candidates[0].id)


def _shape(shape: tuple[int, ...]) -> str:
    return "{" + ", ".join(str(value) for value in shape) + "}"


def _at_dtype(dtype: str) -> str:
    return {
        "bool": "at::kBool",
        "i32": "at::kInt",
        "i64": "at::kLong",
        "f16": "at::kHalf",
        "bf16": "at::kBFloat16",
        "f32": "at::kFloat",
        "f64": "at::kDouble",
    }[dtype]


def _camel(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_"))
