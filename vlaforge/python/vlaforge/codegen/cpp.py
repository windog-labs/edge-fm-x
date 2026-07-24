"""Deterministic static C++ generation from a verified physical Plan."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from vlaforge.codegen.certificate import render_optimization_certificate_header
from vlaforge.codegen.model import (
    CppRegionDefinition,
    CppValidatorDefinition,
    GeneratedSources,
)
from vlaforge.compiler import CompilationCertificate, CompilationResult
from vlaforge.ir.program import Module
from vlaforge.ir.serializer import module_digest
from vlaforge.ir.types import (
    ActionType,
    CommittedActionType,
    EpochType,
    IRType,
    ScalarType,
    TensorType,
    TransactionType,
)
from vlaforge.plan import PlanModule, Task, emit_memory_constants, verify_plan
from vlaforge.plan.memory import state_arena_sizes, storage_size_bytes
from vlaforge.plan.model import BufferClass


class CodegenUnsupportedError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _EmitContext:
    aliases: Mapping[int, str]


def generate_cpp_session(
    plan: PlanModule,
    semantic_module: Module,
    *,
    regions: Mapping[str, CppRegionDefinition],
    validators: Mapping[str, CppValidatorDefinition],
    runner_source: str | None = None,
    namespace: str = "vlaforge_generated",
    compilation_certificate: CompilationCertificate | None = None,
) -> GeneratedSources:
    verify_plan(plan)
    if plan.arena is None:
        raise ValueError("C++ codegen requires a physicalized plan")
    if plan.semantic_digest != module_digest(semantic_module):
        raise ValueError("C++ codegen source module digest mismatch")
    if len(plan.policies) != 1:
        raise CodegenUnsupportedError(
            "static C++ MVP currently requires exactly one policy"
        )
    if not _IDENTIFIER.fullmatch(namespace):
        raise ValueError("C++ namespace must be an identifier")
    if compilation_certificate is not None:
        if compilation_certificate.plan_digest != plan.digest():
            raise ValueError("compilation certificate plan digest mismatch")
        if (
            compilation_certificate.compiled_semantic_digest
            != module_digest(semantic_module)
        ):
            raise ValueError(
                "compilation certificate semantic digest mismatch"
            )

    required_regions = {artifact.region_name for artifact in plan.artifacts}
    missing_regions = sorted(required_regions - set(regions))
    extra_regions = sorted(set(regions) - required_regions)
    required_validators = {
        str(task.attributes["contract"])
        for task in plan.tasks
        if task.opcode == "vla.validate"
    }
    missing_validators = sorted(required_validators - set(validators))
    if missing_regions or extra_regions or missing_validators:
        raise CodegenUnsupportedError(
            "C++ definitions mismatch: "
            f"missing_regions={missing_regions}, "
            f"extra_regions={extra_regions}, "
            f"missing_validators={missing_validators}"
        )

    emitter = _CppEmitter(
        plan,
        semantic_module,
        regions,
        validators,
        namespace=namespace,
        compilation_certificate=compilation_certificate,
    )
    files = {
        "CMakeLists.txt": _cmake_source(runner_source is not None),
        "memory_constants.h": emit_memory_constants(
            plan, namespace=namespace
        ),
        "session_generated.cpp": emitter.source(),
        "session_generated.h": emitter.header(),
    }
    if compilation_certificate is not None:
        files["optimization_certificate.h"] = (
            render_optimization_certificate_header(
                compilation_certificate,
                namespace=namespace,
            )
        )
    if runner_source is not None:
        files["runner.cpp"] = runner_source
    return GeneratedSources(tuple(sorted(files.items())))


def generate_compiled_cpp_session(
    compilation: CompilationResult,
    *,
    regions: Mapping[str, CppRegionDefinition],
    validators: Mapping[str, CppValidatorDefinition],
    runner_source: str | None = None,
    namespace: str = "vlaforge_generated",
) -> GeneratedSources:
    """Generate the normal Session from one certified compiler result."""

    return generate_cpp_session(
        compilation.plan,
        compilation.module,
        regions=regions,
        validators=validators,
        runner_source=runner_source,
        namespace=namespace,
        compilation_certificate=compilation.certificate,
    )


class _CppEmitter:
    def __init__(
        self,
        plan: PlanModule,
        module: Module,
        regions: Mapping[str, CppRegionDefinition],
        validators: Mapping[str, CppValidatorDefinition],
        *,
        namespace: str,
        compilation_certificate: CompilationCertificate | None,
    ):
        self.plan = plan
        self.module = module
        self.regions = dict(regions)
        self.validators = dict(validators)
        self.namespace = namespace
        self.compilation_certificate = compilation_certificate
        self.cache_certificates = {
            item.task_id: item
            for item in (
                ()
                if compilation_certificate is None
                else compilation_certificate.caches
            )
            if item.enabled
        }
        self.cache_indices = {
            task_id: index
            for index, task_id in enumerate(sorted(self.cache_certificates))
        }
        for task_id, certificate in self.cache_certificates.items():
            task = plan.task(task_id)
            if task.opcode != "vla.invoke":
                raise CodegenUnsupportedError(
                    f"cache certificate task {task_id} is not a region invoke"
                )
            if str(task.attributes.get("region")) != certificate.region:
                raise CodegenUnsupportedError(
                    f"cache certificate region mismatch for task {task_id}"
                )
        self.input_ids = {
            stream.name: index for index, stream in enumerate(module.inputs)
        }
        self.artifact_ids = {
            artifact.region_name: artifact.artifact_id
            for artifact in plan.artifacts
        }
        self.physical = {
            logical_id: physical
            for physical in plan.arena.physical_buffers
            for logical_id in physical.logical_buffers
        }

    def header(self) -> str:
        artifact_count = len(self.plan.artifacts)
        input_count = len(self.module.inputs)
        cache_include = (
            '#include "vlaforge/runtime/epoch_cache.h"\n'
            if self.compilation_certificate is not None
            else ""
        )
        cache_member = (
            "  std::array<vlaforge::runtime::EpochVersionCacheGuard,\n"
            f"             {len(self.cache_certificates)}> cache_guards_{{}};\n"
            if self.compilation_certificate is not None
            else ""
        )
        return f"""#ifndef VLAFORGE_GENERATED_SESSION_H_
#define VLAFORGE_GENERATED_SESSION_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "vlaforge/runtime/action_queue.h"
{cache_include}#include "vlaforge/runtime/region_executable.h"
#include "vlaforge/runtime/session.h"
#include "vlaforge/runtime/state_store.h"
#include "vlaforge/runtime/static_arena.h"
#include "vlaforge/runtime/transaction.h"

namespace {self.namespace} {{

class GeneratedSession final : public vlaforge::runtime::Session {{
 public:
  GeneratedSession();
  ~GeneratedSession() override;

  GeneratedSession(const GeneratedSession&) = delete;
  GeneratedSession& operator=(const GeneratedSession&) = delete;

  vlaforge::runtime::Status ResetEpisode(
      std::uint64_t new_episode) noexcept override;
  vlaforge::runtime::Status BindInput(
      std::uint32_t input_id, const vlaforge::runtime::TensorView& input,
      const vlaforge::runtime::Epoch& epoch) noexcept override;
  vlaforge::runtime::Status RunTick(
      const vlaforge::runtime::Epoch& tick) noexcept override;
  vlaforge::runtime::Status ReadCommittedAction(
      vlaforge::runtime::CommittedAction* action) const noexcept override;
  void SetTraceSink(vlaforge::runtime::TraceSink trace) noexcept override;

 private:
  struct BoundInput {{
    vlaforge::runtime::TensorView view{{}};
    vlaforge::runtime::Epoch epoch{{}};
    bool bound = false;
  }};

  void InitializeBufferObjects() noexcept;
  vlaforge::runtime::Status InitializeArtifacts() noexcept;
  vlaforge::runtime::Status AbortTick(
      vlaforge::runtime::Status cause) noexcept;
  void* BufferData(std::uint32_t logical_id) noexcept;
  VLAForgeTensorView MakeRegionView(
      std::uint32_t logical_id, void* data) const noexcept;

  vlaforge::runtime::StaticArena arena_;
  vlaforge::runtime::StaticArena state_arena_;
  vlaforge::runtime::StateStore state_store_;
  vlaforge::runtime::Transaction transaction_;
  vlaforge::runtime::ActionQueue actions_;
  vlaforge::runtime::TraceSink trace_{{}};
  vlaforge::runtime::Status initialization_status_{{}};
  std::array<BoundInput, {input_count}> inputs_{{}};
  std::array<VLAForgeRegionExecutable*, {artifact_count}> executables_{{}};
{cache_member}}};

}}  // namespace {self.namespace}

#endif  // VLAFORGE_GENERATED_SESSION_H_
"""

    def source(self) -> str:
        sections = [
            '#include "session_generated.h"',
            '#include "memory_constants.h"',
        ]
        if self.compilation_certificate is not None:
            sections.append('#include "optimization_certificate.h"')
        sections.extend([
            "",
            "#include <array>",
            "#include <cmath>",
            "#include <cstddef>",
            "#include <cstdint>",
            "#include <cstring>",
            "#include <new>",
            "#include <utility>",
            "",
            self._fixture_executable_declaration(),
            "",
            "namespace {",
            "",
            self._logical_buffer_tables(),
            "",
            self._state_tables(),
            "",
            self._fixture_backend(),
            "",
            "}  // namespace",
            "",
            f"namespace {self.namespace} {{",
            "",
            self._session_implementation(),
            "",
            f"}}  // namespace {self.namespace}",
            "",
        ])
        return "\n".join(sections)

    def _fixture_executable_declaration(self) -> str:
        maximum_inputs = max(
            (len(region.inputs) for region in self.module.regions), default=1
        )
        maximum_outputs = max(
            (len(region.outputs) for region in self.module.regions), default=1
        )
        return f"""struct VLAForgeRegionExecutable {{
  std::uint32_t region_id = 0;
  bool loaded = false;
  std::array<VLAForgeTensorView, {maximum_inputs}> inputs{{}};
  std::array<VLAForgeTensorView, {maximum_outputs}> outputs{{}};
  void* workspace = nullptr;
  std::uint64_t workspace_size = 0;
}};"""

    def _logical_buffer_tables(self) -> str:
        logical = []
        for buffer in self.plan.buffers:
            physical = self.physical.get(buffer.id)
            if physical is None:
                logical.append(
                    "  {static_cast<std::size_t>(-1), 0u, 1u},"
                )
            else:
                logical.append(
                    f"  {{{physical.offset}u, {physical.size_bytes}u, "
                    f"{physical.alignment}u}},"
                )
        shapes = []
        for buffer in self.plan.buffers:
            if isinstance(buffer.type, TensorType) and buffer.type.shape:
                if any(item is None for item in buffer.type.shape):
                    raise CodegenUnsupportedError(
                        f"buffer {buffer.id} has dynamic region ABI shape"
                    )
                dimensions = ", ".join(
                    str(int(item)) for item in buffer.type.shape
                )
                shapes.append(
                    f"constexpr std::int64_t kShape{buffer.id}[] = "
                    f"{{{dimensions}}};"
                )
        return """struct LogicalBufferDesc {
  std::size_t offset;
  std::size_t size;
  std::size_t alignment;
};

constexpr LogicalBufferDesc kLogicalBuffers[] = {
%s
};

%s""" % ("\n".join(logical), "\n".join(shapes))

    def _state_tables(self) -> str:
        state_sizes = state_arena_sizes(self.plan)
        if len(state_sizes) > 1:
            raise CodegenUnsupportedError(
                "static C++ MVP supports one state arena device"
            )
        state_arena_size = next(iter(state_sizes.values()), 0)
        state_alignment = max(
            (state.alignment or 1 for state in self.plan.states), default=1
        )
        if not self.plan.states:
            slots = (
                "constexpr const vlaforge::runtime::StateSlotDescriptor* "
                "kStateSlots = nullptr;"
            )
        else:
            entries = []
            for state in self.plan.states:
                assert state.slot_capacity is not None
                assert state.slot_size_bytes is not None
                assert state.alignment is not None
                assert state.offset is not None
                reset = (
                    "true"
                    if self.module.state(state.name).reset.value
                    in {"episode_start", "explicit", "error"}
                    else "false"
                )
                entries.append(
                    "  {"
                    f"{state.state_id}u, {state.slot_capacity}u, "
                    f"{state.slot_size_bytes}u, {state.alignment}u, "
                    f"{state.offset}u, {reset}"
                    "},"
                )
            slots = (
                "constexpr vlaforge::runtime::StateSlotDescriptor "
                "kStateSlots[] = {\n"
                + "\n".join(entries)
                + "\n};"
            )
        return f"""constexpr std::size_t kStateArenaSize = {state_arena_size}u;
constexpr std::size_t kStateArenaAlignment = {state_alignment}u;
{slots}"""

    def _fixture_backend(self) -> str:
        region_functions = []
        for artifact in self.plan.artifacts:
            definition = self.regions[artifact.region_name]
            region_functions.append(
                f"""VLAForgeStatus RunRegion{artifact.artifact_id}(
    VLAForgeRegionExecutable* executable) {{
{_indent(definition.body, 2)}
}}"""
            )
        validator_functions = []
        for index, name in enumerate(sorted(self.validators)):
            definition = self.validators[name]
            validator_functions.append(
                f"""bool Validate{index}(const void* data,
               std::size_t size_bytes) noexcept {{
{_indent(definition.body, 2)}
}}"""
            )
        run_cases = "\n".join(
            f"    case {artifact.artifact_id}u:\n"
            f"      return RunRegion{artifact.artifact_id}(executable);"
            for artifact in self.plan.artifacts
        )
        workspace_cases = "\n".join(
            f"    case {artifact.artifact_id}u:\n"
            f"      requirement->size_bytes = "
            f"{artifact.workspace_size_bytes}u;\n"
            f"      requirement->alignment = "
            f"{artifact.workspace_alignment}u;\n"
            "      break;"
            for artifact in self.plan.artifacts
        )
        return """
bool CheckTensor(const VLAForgeTensorView& tensor, VLAForgeDType dtype,
                 std::size_t elements) noexcept {
  const std::size_t element_bytes =
      dtype == VLAFORGE_DTYPE_BOOL ? 1u :
      dtype == VLAFORGE_DTYPE_I32 || dtype == VLAFORGE_DTYPE_F32 ? 4u :
      dtype == VLAFORGE_DTYPE_F16 || dtype == VLAFORGE_DTYPE_BF16 ? 2u :
      dtype == VLAFORGE_DTYPE_I64 || dtype == VLAFORGE_DTYPE_F64 ? 8u : 0u;
  return tensor.data != nullptr && tensor.dtype == dtype &&
         element_bytes != 0u &&
         tensor.size_bytes == elements * element_bytes;
}

template <typename T>
const T* Input(const VLAForgeRegionExecutable* executable,
               std::uint32_t index) noexcept {
  return static_cast<const T*>(executable->inputs[index].data);
}

template <typename T>
T* Output(VLAForgeRegionExecutable* executable,
          std::uint32_t index) noexcept {
  return static_cast<T*>(executable->outputs[index].data);
}

%s

%s

VLAForgeStatus FixtureCreate(
    const VLAForgeRegionCreateOptions* options,
    VLAForgeRegionExecutable** output) {
  if (options == nullptr || output == nullptr ||
      options->struct_size < sizeof(*options) ||
      options->abi_version != VLAFORGE_REGION_EXECUTABLE_ABI_VERSION) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid fixture create options");
  }
  auto* executable = new (std::nothrow) VLAForgeRegionExecutable();
  if (executable == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                 "fixture executable allocation failed");
  }
  executable->region_id = options->region_id;
  *output = executable;
  return vlaforge_status_ok();
}

VLAForgeStatus FixtureLoad(
    VLAForgeRegionExecutable* executable,
    const VLAForgeArtifactDescriptor* artifact) {
  if (executable == nullptr || artifact == nullptr ||
      artifact->struct_size < sizeof(*artifact) ||
      artifact->callable_abi_version !=
          VLAFORGE_REGION_EXECUTABLE_ABI_VERSION) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid embedded fixture artifact");
  }
  executable->loaded = true;
  return vlaforge_status_ok();
}

VLAForgeStatus FixtureQueryWorkspace(
    const VLAForgeRegionExecutable* executable,
    VLAForgeWorkspaceRequirement* requirement) {
  if (executable == nullptr || requirement == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid workspace query");
  }
  requirement->size_bytes = 0u;
  requirement->alignment = 1u;
  requirement->device = {VLAFORGE_DEVICE_CPU, 0};
  switch (executable->region_id) {
%s
    default:
      return vlaforge_status_error(VLAFORGE_STATUS_NOT_FOUND,
                                   "unknown fixture region");
  }
  return vlaforge_status_ok();
}

VLAForgeStatus FixtureBindInput(
    VLAForgeRegionExecutable* executable, std::uint32_t index,
    const VLAForgeTensorView* tensor) {
  if (executable == nullptr || tensor == nullptr ||
      index >= executable->inputs.size()) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid fixture input binding");
  }
  executable->inputs[index] = *tensor;
  return vlaforge_status_ok();
}

VLAForgeStatus FixtureBindOutput(
    VLAForgeRegionExecutable* executable, std::uint32_t index,
    const VLAForgeTensorView* tensor) {
  if (executable == nullptr || tensor == nullptr ||
      index >= executable->outputs.size()) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid fixture output binding");
  }
  executable->outputs[index] = *tensor;
  return vlaforge_status_ok();
}

VLAForgeStatus FixtureBindWorkspace(
    VLAForgeRegionExecutable* executable, void* workspace,
    std::uint64_t workspace_size) {
  if (executable == nullptr ||
      (workspace_size != 0u && workspace == nullptr)) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid fixture workspace binding");
  }
  executable->workspace = workspace;
  executable->workspace_size = workspace_size;
  return vlaforge_status_ok();
}

VLAForgeStatus FixtureRun(VLAForgeRegionExecutable* executable) {
  if (executable == nullptr || !executable->loaded) {
    return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                 "fixture executable is not loaded");
  }
  switch (executable->region_id) {
%s
    default:
      return vlaforge_status_error(VLAFORGE_STATUS_NOT_FOUND,
                                   "unknown fixture region");
  }
}

VLAForgeStatus FixtureSynchronize(
    VLAForgeRegionExecutable* executable) {
  return executable == nullptr
      ? vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                              "fixture executable is null")
      : vlaforge_status_ok();
}

void FixtureDestroy(VLAForgeRegionExecutable* executable) {
  delete executable;
}

constexpr VLAForgeRegionExecutableApi kFixtureApi = {
    sizeof(VLAForgeRegionExecutableApi),
    VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
    &FixtureCreate,
    &FixtureLoad,
    &FixtureQueryWorkspace,
    &FixtureBindInput,
    &FixtureBindOutput,
    &FixtureBindWorkspace,
    &FixtureRun,
    &FixtureSynchronize,
    &FixtureDestroy,
};
""" % (
            "\n\n".join(region_functions),
            "\n\n".join(validator_functions),
            _indent(workspace_cases, 2),
            _indent(run_cases, 2),
        )

    def _session_implementation(self) -> str:
        state_pointer = "kStateSlots" if self.plan.states else "nullptr"
        policy = self.plan.policies[0]
        policy_clock_id = next(
            index
            for index, clock in enumerate(self.module.clocks)
            if clock.name == policy.clock
        )
        run_body = self._emit_block(
            policy.body_block,
            _EmitContext(
                {
                    policy.inputs[0]: "const_cast<vlaforge::runtime::Epoch*>(&tick)"
                }
            ),
            indent=2,
        )
        run_body = run_body.replace(
            "return status;",
            "return AbortTick(status);",
        )
        init_objects = []
        for buffer in self.plan.buffers:
            if buffer.id not in self.physical:
                continue
            type_name = _cpp_type(buffer.type)
            init_objects.append(
                f"  new (BufferData({buffer.id}u)) {type_name}{{}};"
            )
        input_cases = []
        for input_id, stream in enumerate(self.module.inputs):
            input_cases.append(
                self._input_binding_case(input_id, stream.payload)
            )
        make_view_cases = []
        for buffer in self.plan.buffers:
            make_view_cases.append(self._make_view_case(buffer.id, buffer.type))
        cache_reset = (
            "    for (auto& cache : cache_guards_) {\n"
            "      cache.Invalidate();\n"
            "    }\n"
            if self.compilation_certificate is not None
            else ""
        )
        return f"""GeneratedSession::GeneratedSession()
    : arena_(kArenaSize, kArenaAlignment),
      state_arena_(kStateArenaSize, kStateArenaAlignment),
      state_store_(state_arena_, {state_pointer}, {len(self.plan.states)}u),
      transaction_({len(self.plan.states)}u),
      actions_() {{
  InitializeBufferObjects();
  initialization_status_ = InitializeArtifacts();
}}

GeneratedSession::~GeneratedSession() {{
  for (auto* executable : executables_) {{
    if (executable != nullptr) {{
      kFixtureApi.destroy(executable);
    }}
  }}
}}

void GeneratedSession::InitializeBufferObjects() noexcept {{
{chr(10).join(init_objects)}
}}

vlaforge::runtime::Status GeneratedSession::InitializeArtifacts() noexcept {{
  const VLAForgeStatus api_status =
      vlaforge_region_executable_api_validate(&kFixtureApi);
  if (api_status.code != VLAFORGE_STATUS_OK) {{
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kInternal, 0u, api_status.message);
  }}
  constexpr std::uint8_t kDigest[32] = {{0u}};
  for (std::uint32_t region_id = 0; region_id < executables_.size();
       ++region_id) {{
    const VLAForgeRegionCreateOptions options{{
        sizeof(VLAForgeRegionCreateOptions),
        VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
        region_id,
        {{VLAFORGE_DEVICE_CPU, 0}}}};
    VLAForgeStatus status =
        kFixtureApi.create(&options, &executables_[region_id]);
    if (status.code != VLAFORGE_STATUS_OK) {{
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kInternal, region_id,
          status.message);
    }}
    constexpr char kEmbeddedPath[] = "embedded://cpu-fixture";
    const VLAForgeArtifactDescriptor artifact{{
        sizeof(VLAForgeArtifactDescriptor),
        VLAFORGE_REGION_EXECUTABLE_ABI_VERSION,
        kEmbeddedPath,
        sizeof(kEmbeddedPath) - 1u,
        kDigest,
        0u}};
    status = kFixtureApi.load(executables_[region_id], &artifact);
    if (status.code != VLAFORGE_STATUS_OK) {{
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kInternal, region_id,
          status.message);
    }}
  }}
  return vlaforge::runtime::Status::Ok();
}}

vlaforge::runtime::Status GeneratedSession::AbortTick(
    vlaforge::runtime::Status cause) noexcept {{
  if (transaction_.active()) {{
    (void)state_store_.Abort(&transaction_, 0u);
  }}
  return cause;
}}

void* GeneratedSession::BufferData(std::uint32_t logical_id) noexcept {{
  if (logical_id >=
      sizeof(kLogicalBuffers) / sizeof(kLogicalBuffers[0])) {{
    return nullptr;
  }}
  const auto& descriptor = kLogicalBuffers[logical_id];
  if (descriptor.offset == static_cast<std::size_t>(-1)) {{
    return nullptr;
  }}
  return arena_.Resolve(descriptor.offset, descriptor.size,
                        descriptor.alignment);
}}

VLAForgeTensorView GeneratedSession::MakeRegionView(
    std::uint32_t logical_id, void* data) const noexcept {{
  switch (logical_id) {{
{chr(10).join(make_view_cases)}
    default:
      return VLAForgeTensorView{{}};
  }}
}}

vlaforge::runtime::Status GeneratedSession::ResetEpisode(
    std::uint64_t new_episode) noexcept {{
  if (!initialization_status_.ok()) {{
    return initialization_status_;
  }}
  const auto status = state_store_.ResetEpisode(new_episode, 0u);
  if (status.ok()) {{
    actions_.Reset();
{cache_reset}  }}
  return status;
}}

vlaforge::runtime::Status GeneratedSession::BindInput(
    std::uint32_t input_id, const vlaforge::runtime::TensorView& input,
    const vlaforge::runtime::Epoch& epoch) noexcept {{
  if (!initialization_status_.ok()) {{
    return initialization_status_;
  }}
  switch (input_id) {{
{chr(10).join(input_cases)}
    default:
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kOutOfRange, input_id,
          "input id is out of range");
  }}
}}

vlaforge::runtime::Status GeneratedSession::RunTick(
    const vlaforge::runtime::Epoch& tick) noexcept {{
  if (!initialization_status_.ok()) {{
    return initialization_status_;
  }}
  if (tick.clock_id != {policy_clock_id}u ||
      tick.episode != state_store_.episode()) {{
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kFailedPrecondition, 0u,
        "tick clock or episode mismatch");
  }}
  vlaforge::runtime::Status status = vlaforge::runtime::Status::Ok();
{run_body}
  return vlaforge::runtime::Status::Ok();
}}

vlaforge::runtime::Status GeneratedSession::ReadCommittedAction(
    vlaforge::runtime::CommittedAction* action) const noexcept {{
  if (action == nullptr || !actions_.latest().valid) {{
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kNotFound, 0u,
        "no committed action is available");
  }}
  *action = actions_.latest();
  return vlaforge::runtime::Status::Ok();
}}

void GeneratedSession::SetTraceSink(
    vlaforge::runtime::TraceSink trace) noexcept {{
  trace_ = trace;
  state_store_.SetTraceSink(trace);
  actions_.SetTraceSink(trace);
}}"""

    def _emit_block(
        self, block_id: int, context: _EmitContext, *, indent: int
    ) -> str:
        lines: list[str] = []
        aliases = dict(context.aliases)
        block = self.plan.block(block_id)
        for task_id in block.tasks:
            task = self.plan.task(task_id)
            if task.opcode == "vla.sample_input":
                stream = str(task.attributes["stream"])
                input_id = self.input_ids[stream]
                maximum = int(task.attributes.get("max_age_ns", 0))
                lines.extend(
                    [
                        f"if (!inputs_[{input_id}u].bound) {{",
                        "  return vlaforge::runtime::Status::Error(",
                        "      vlaforge::runtime::StatusCode::kFailedPrecondition,",
                        f"      {input_id}u, \"required input is not bound\");",
                        "}",
                        (
                            f"if (inputs_[{input_id}u].epoch.episode != "
                            "tick.episode ||"
                        ),
                        (
                            f"    inputs_[{input_id}u].epoch.timestamp_ns > "
                            "tick.timestamp_ns ||"
                        ),
                        (
                            f"    tick.timestamp_ns - inputs_[{input_id}u]."
                            f"epoch.timestamp_ns > {maximum}u) {{"
                        ),
                        "  return vlaforge::runtime::Status::Error(",
                        "      vlaforge::runtime::StatusCode::kFailedPrecondition,",
                        f"      {task.id}u, \"input freshness guard failed\");",
                        "}",
                        (
                            "vlaforge::runtime::EmitTrace(trace_, "
                            "vlaforge::runtime::TraceEvent{"
                            "vlaforge::runtime::TraceKind::kInput, "
                            f"{task.id}u, 0u, 0u, 0u, tick}});"
                        ),
                    ]
                )
                aliases[task.outputs[0]] = f"inputs_[{input_id}u].view.data"
                aliases[task.outputs[1]] = f"&inputs_[{input_id}u].epoch"
            elif task.opcode == "vla.txn.begin":
                lines.extend(
                    [
                        (
                            "status = state_store_.Begin(&transaction_, tick, "
                            f"{task.id}u);"
                        ),
                        "if (!status.ok()) { return status; }",
                    ]
                )
                aliases[task.outputs[0]] = "&transaction_"
            elif task.opcode == "vla.invoke":
                lines.extend(self._emit_region(task, aliases))
            elif task.opcode == "vla.for":
                lines.extend(self._emit_for(task, aliases))
            elif task.opcode == "vla.validate":
                contract = str(task.attributes["contract"])
                validator_id = sorted(self.validators).index(contract)
                value = self._raw(task.inputs[0], aliases)
                size = _value_bytes(self.plan.buffer(task.inputs[0]).type)
                output = self._lvalue(task.outputs[0], aliases)
                lines.extend(
                    [
                        (
                            f"{output} = Validate{validator_id}({value}, "
                            f"{size}u);"
                        ),
                        (
                            "vlaforge::runtime::EmitTrace(trace_, "
                            "vlaforge::runtime::TraceEvent{"
                            "vlaforge::runtime::TraceKind::kValidation, "
                            f"{task.id}u, 0u, 0u, transaction_.id(), tick}});"
                        ),
                    ]
                )
            elif task.opcode == "vla.action.create":
                value_id = task.inputs[0]
                output = self._lvalue(task.outputs[0], aliases)
                value = self._raw(value_id, aliases)
                size = _value_bytes(self.plan.buffer(value_id).type)
                lines.extend(
                    [
                        (
                            f"{output} = vlaforge::runtime::PendingAction"
                            f"{{tick, {value}, {size}u}};"
                        ),
                        (
                            "vlaforge::runtime::EmitTrace(trace_, "
                            "vlaforge::runtime::TraceEvent{"
                            "vlaforge::runtime::TraceKind::kActionPending, "
                            f"{task.id}u, 0u, 0u, transaction_.id(), tick}});"
                        ),
                    ]
                )
            elif task.opcode == "vla.txn.commit":
                pending = self._lvalue(task.inputs[1], aliases)
                condition = self._lvalue(task.inputs[2], aliases)
                committed = self._raw(task.outputs[0], aliases)
                lines.extend(
                    [
                        (
                            "status = state_store_.Commit(&transaction_, "
                            f"{pending}, {condition}, {task.id}u, "
                            "static_cast<vlaforge::runtime::"
                            f"CommittedAction*>({committed}));"
                        ),
                        "if (!status.ok()) { return status; }",
                    ]
                )
            elif task.opcode == "vla.action.publish":
                committed = self._lvalue(task.inputs[0], aliases)
                lines.extend(
                    [
                        f"status = actions_.Publish({committed}, {task.id}u);",
                        "if (!status.ok()) { return status; }",
                    ]
                )
            elif task.opcode == "vla.return":
                continue
            elif task.opcode == "vla.yield":
                raise CodegenUnsupportedError(
                    "yield may only appear in a structured loop body"
                )
            else:
                raise CodegenUnsupportedError(
                    f"C++ MVP does not yet lower {task.opcode}"
                )
        return _indent("\n".join(lines), indent)

    def _emit_region(
        self, task: Task, aliases: Mapping[int, str]
    ) -> list[str]:
        artifact_id = task.artifact_id
        assert artifact_id is not None
        lines = ["{"]
        for index, buffer_id in enumerate(task.inputs):
            lines.extend(
                [
                    (
                        f"  auto input_{task.id}_{index} = "
                        f"MakeRegionView({buffer_id}u, "
                        f"{self._raw(buffer_id, aliases)});"
                    ),
                    (
                        "  VLAForgeStatus region_status = "
                        f"kFixtureApi.bind_input(executables_[{artifact_id}u], "
                        f"{index}u, &input_{task.id}_{index});"
                    )
                    if index == 0
                    else (
                        f"  region_status = kFixtureApi.bind_input("
                        f"executables_[{artifact_id}u], {index}u, "
                        f"&input_{task.id}_{index});"
                    ),
                    *self._region_status_check(task.id, indent=2),
                ]
            )
        if not task.inputs:
            lines.append(
                "  VLAForgeStatus region_status = vlaforge_status_ok();"
            )
        for index, buffer_id in enumerate(task.outputs):
            lines.extend(
                [
                    (
                        f"  auto output_{task.id}_{index} = "
                        f"MakeRegionView({buffer_id}u, "
                        f"{self._raw(buffer_id, aliases)});"
                    ),
                    (
                        f"  region_status = kFixtureApi.bind_output("
                        f"executables_[{artifact_id}u], {index}u, "
                        f"&output_{task.id}_{index});"
                    ),
                    *self._region_status_check(task.id, indent=2),
                ]
            )
        workspace = (
            "nullptr"
            if task.workspace_buffer is None
            else self._raw(task.workspace_buffer, aliases)
        )
        workspace_size = (
            0
            if task.workspace_buffer is None
            else self.physical[task.workspace_buffer].size_bytes
        )
        lines.extend(
            [
                (
                    f"  region_status = kFixtureApi.bind_workspace("
                    f"executables_[{artifact_id}u], {workspace}, "
                    f"{workspace_size}u);"
                ),
                *self._region_status_check(task.id, indent=2),
                (
                    f"  region_status = kFixtureApi.run("
                    f"executables_[{artifact_id}u]);"
                ),
                *self._region_status_check(task.id, indent=2),
                (
                    f"  region_status = kFixtureApi.synchronize("
                    f"executables_[{artifact_id}u]);"
                ),
                *self._region_status_check(task.id, indent=2),
                (
                    "  vlaforge::runtime::EmitTrace(trace_, "
                    "vlaforge::runtime::TraceEvent{"
                    "vlaforge::runtime::TraceKind::kRegion, "
                    f"{task.id}u, 0u, 0u, transaction_.id(), tick}});"
                ),
                "}",
            ]
        )
        certificate = self.cache_certificates.get(task.id)
        if certificate is not None:
            dependencies = []
            for item in certificate.dependencies:
                if item.kind != "epoch":
                    raise CodegenUnsupportedError(
                        "generated fixture Session needs an explicit "
                        "StateVersion binding before state-version caching"
                    )
                dependency = (
                    "vlaforge::runtime::TemporalDependency{"
                    "vlaforge::runtime::TemporalDependencyKind::kInputEpoch, "
                    f"{item.subject_id}u, "
                    f"inputs_[{item.subject_id}u].epoch.sequence, "
                    f"inputs_[{item.subject_id}u].epoch, "
                    f"kCacheTask{task.id}Dependencies"
                    f"[{len(dependencies)}u].max_age_ns, "
                    f"kCacheTask{task.id}Dependencies"
                    f"[{len(dependencies)}u].max_versions"
                    "}"
                )
                dependencies.append(f"    {dependency},")
            cache_index = self.cache_indices[task.id]
            trace_line = lines[-2]
            region_body = lines[1:-2]
            return [
                "{",
                (
                    "  const vlaforge::runtime::TemporalDependency "
                    f"cache_dependencies_{task.id}[] = {{"
                ),
                *dependencies,
                "  };",
                (
                    f"  const bool cache_hit_{task.id} = "
                    f"cache_guards_[{cache_index}u].Lookup("
                    f"cache_dependencies_{task.id}, "
                    f"kCacheTask{task.id}DependencyCount, tick);"
                ),
                f"  if (!cache_hit_{task.id}) {{",
                *_indent_lines(region_body, 2),
                (
                    f"    status = cache_guards_[{cache_index}u].Update("
                    f"cache_dependencies_{task.id}, "
                    f"kCacheTask{task.id}DependencyCount);"
                ),
                "    if (!status.ok()) { return status; }",
                "  }",
                trace_line,
                "}",
            ]
        return lines

    def _region_status_check(
        self, task_id: int, *, indent: int
    ) -> list[str]:
        prefix = " " * indent
        return [
            f"{prefix}if (region_status.code != VLAFORGE_STATUS_OK) {{",
            (
                f"{prefix}  return AbortTick("
                "vlaforge::runtime::Status::Error("
                "vlaforge::runtime::StatusCode::kInternal, "
                f"{task_id}u, region_status.message));"
            ),
            f"{prefix}}}",
        ]

    def _emit_for(
        self, task: Task, aliases: Mapping[int, str]
    ) -> list[str]:
        body = self.plan.block(task.blocks[0])
        if len(body.arguments) != 2 or len(task.outputs) != 1:
            raise CodegenUnsupportedError(
                "C++ fixed loop requires one carry and two block arguments"
            )
        carry_type = _cpp_type(self.plan.buffer(task.outputs[0]).type)
        carry_name = f"loop_{task.id}_carry"
        index_name = f"loop_{task.id}_index"
        nested_aliases = dict(aliases)
        nested_aliases[body.arguments[0]] = f"&{index_name}"
        nested_aliases[body.arguments[1]] = f"&{carry_name}"
        lines = [
            "{",
            (
                f"  {carry_type} {carry_name} = "
                f"{self._lvalue(task.inputs[0], aliases)};"
            ),
            (
                f"  for (std::int64_t {index_name} = "
                f"{int(task.attributes['lower'])}; "
                f"{index_name} < {int(task.attributes['upper'])}; "
                f"{index_name} += {int(task.attributes['step'])}) {{"
            ),
        ]
        yielded: int | None = None
        for nested_task_id in body.tasks:
            nested_task = self.plan.task(nested_task_id)
            if nested_task.opcode == "vla.yield":
                if len(nested_task.inputs) != 1:
                    raise CodegenUnsupportedError(
                        "C++ fixed loop requires one yielded carry"
                    )
                yielded = nested_task.inputs[0]
                continue
            if nested_task.opcode != "vla.invoke":
                raise CodegenUnsupportedError(
                    "C++ fixed loop body currently supports region tasks only"
                )
            lines.extend(
                _indent_lines(
                    self._emit_region(nested_task, nested_aliases), 2
                )
            )
        if yielded is None:
            raise CodegenUnsupportedError("fixed loop body has no yield")
        lines.extend(
            [
                f"    {carry_name} = {self._lvalue(yielded, nested_aliases)};",
                "  }",
                f"  {self._lvalue(task.outputs[0], aliases)} = {carry_name};",
                "}",
            ]
        )
        return lines

    def _raw(self, buffer_id: int, aliases: Mapping[int, str]) -> str:
        return aliases.get(buffer_id, f"BufferData({buffer_id}u)")

    def _lvalue(self, buffer_id: int, aliases: Mapping[int, str]) -> str:
        type_name = _cpp_type(self.plan.buffer(buffer_id).type)
        return f"*static_cast<{type_name}*>({self._raw(buffer_id, aliases)})"

    def _input_binding_case(self, input_id: int, type: IRType) -> str:
        if not isinstance(type, TensorType):
            raise CodegenUnsupportedError("runtime inputs must be tensors")
        if any(item is None for item in type.shape):
            raise CodegenUnsupportedError(
                "generated fixture input requires static shape"
            )
        checks = [
            "input.data == nullptr",
            f"input.scalar_type != {_runtime_scalar(type.dtype)}",
            f"input.rank != {len(type.shape)}u",
            f"input.bytes != {_value_bytes(type)}u",
        ]
        for index, dimension in enumerate(type.shape):
            checks.append(
                f"input.shape == nullptr || input.shape[{index}] != "
                f"{int(dimension)}"
            )
        condition = " ||\n          ".join(checks)
        return f"""    case {input_id}u:
      if ({condition}) {{
        return vlaforge::runtime::Status::Error(
            vlaforge::runtime::StatusCode::kInvalidArgument, input_id,
            "input tensor contract mismatch");
      }}
      inputs_[{input_id}u] = BoundInput{{input, epoch, true}};
      return vlaforge::runtime::Status::Ok();"""

    def _make_view_case(self, buffer_id: int, type: IRType) -> str:
        if isinstance(type, TensorType):
            shape = (
                "nullptr" if not type.shape else f"kShape{buffer_id}"
            )
            rank = len(type.shape)
            dtype = _region_dtype(type.dtype)
            size = _value_bytes(type)
        elif isinstance(type, ScalarType):
            shape = "nullptr"
            rank = 0
            dtype = _region_dtype(type.name)
            size = _value_bytes(type)
        else:
            return (
                f"    case {buffer_id}u:\n"
                "      return VLAForgeTensorView{};"
            )
        return f"""    case {buffer_id}u:
      return VLAForgeTensorView{{
          data, {size}u, {shape}, {rank}u, {dtype},
          {{VLAFORGE_DEVICE_CPU, 0}}}};"""


def _cpp_type(type: IRType) -> str:
    if isinstance(type, TensorType):
        if any(item is None for item in type.shape):
            raise CodegenUnsupportedError(
                "dynamic internal tensor needs a concrete codegen profile"
            )
        elements = 1
        for dimension in type.shape:
            elements *= int(dimension)
        scalar = _cpp_scalar(type.dtype)
        return scalar if not type.shape else f"std::array<{scalar}, {elements}>"
    if isinstance(type, ScalarType):
        return _cpp_scalar(type.name)
    if isinstance(type, EpochType):
        return "vlaforge::runtime::Epoch"
    if isinstance(type, TransactionType):
        return "vlaforge::runtime::Transaction*"
    if isinstance(type, ActionType):
        return "vlaforge::runtime::PendingAction"
    if isinstance(type, CommittedActionType):
        return "vlaforge::runtime::CommittedAction"
    raise CodegenUnsupportedError(f"unsupported C++ buffer type: {type!r}")


def _cpp_scalar(dtype: str) -> str:
    result = {
        "bool": "bool",
        "i32": "std::int32_t",
        "i64": "std::int64_t",
        "index": "std::int64_t",
        "f32": "float",
        "f64": "double",
        "opaque": "std::uintptr_t",
    }.get(dtype)
    if result is None:
        raise CodegenUnsupportedError(f"unsupported C++ scalar: {dtype}")
    return result


def _region_dtype(dtype: str) -> str:
    result = {
        "bool": "VLAFORGE_DTYPE_BOOL",
        "i32": "VLAFORGE_DTYPE_I32",
        "i64": "VLAFORGE_DTYPE_I64",
        "index": "VLAFORGE_DTYPE_I64",
        "f16": "VLAFORGE_DTYPE_F16",
        "bf16": "VLAFORGE_DTYPE_BF16",
        "f32": "VLAFORGE_DTYPE_F32",
        "f64": "VLAFORGE_DTYPE_F64",
    }.get(dtype)
    if result is None:
        raise CodegenUnsupportedError(
            f"unsupported RegionExecutable dtype: {dtype}"
        )
    return result


def _runtime_scalar(dtype: str) -> str:
    result = {
        "bool": "vlaforge::runtime::ScalarType::kBool",
        "i32": "vlaforge::runtime::ScalarType::kI32",
        "i64": "vlaforge::runtime::ScalarType::kI64",
        "f16": "vlaforge::runtime::ScalarType::kF16",
        "bf16": "vlaforge::runtime::ScalarType::kBF16",
        "f32": "vlaforge::runtime::ScalarType::kF32",
        "f64": "vlaforge::runtime::ScalarType::kF64",
        "u8": "vlaforge::runtime::ScalarType::kU8",
    }.get(dtype)
    if result is None:
        raise CodegenUnsupportedError(f"unsupported runtime scalar: {dtype}")
    return result


def _value_bytes(type: IRType) -> int:
    if isinstance(type, TensorType | ScalarType):
        return storage_size_bytes(type)
    raise CodegenUnsupportedError(
        f"value is not a RegionExecutable ABI tensor: {type!r}"
    )


def _cmake_source(has_runner: bool) -> str:
    runner = """
add_executable(vlaforge_generated_runner runner.cpp)
target_link_libraries(vlaforge_generated_runner PRIVATE
    vlaforge_generated_session)
install(TARGETS vlaforge_generated_runner RUNTIME DESTINATION bin)
""" if has_runner else ""
    return f"""cmake_minimum_required(VERSION 3.18)
project(vlaforge_generated_session LANGUAGES C CXX)

include(GNUInstallDirs)

if(NOT DEFINED VLAFORGE_RUNTIME_ROOT)
  message(FATAL_ERROR "set VLAFORGE_RUNTIME_ROOT to the VLAForge source root")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory("${{VLAFORGE_RUNTIME_ROOT}}"
                 "${{CMAKE_CURRENT_BINARY_DIR}}/vlaforge_runtime")

add_library(vlaforge_generated_session STATIC session_generated.cpp)
target_include_directories(vlaforge_generated_session PUBLIC
    $<BUILD_INTERFACE:${{CMAKE_CURRENT_SOURCE_DIR}}>
    $<INSTALL_INTERFACE:${{CMAKE_INSTALL_INCLUDEDIR}}/vlaforge/generated>)
target_link_libraries(vlaforge_generated_session PUBLIC vlaforge_runtime)
target_compile_options(vlaforge_generated_session PRIVATE
    -Wall -Wextra -Wpedantic -Werror)

install(TARGETS vlaforge_generated_session
    EXPORT VLAForgeRuntimeTargets
    ARCHIVE DESTINATION ${{CMAKE_INSTALL_LIBDIR}})
set(VLAFORGE_GENERATED_HEADERS session_generated.h memory_constants.h)
if(EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/optimization_certificate.h")
  list(APPEND VLAFORGE_GENERATED_HEADERS optimization_certificate.h)
endif()
install(FILES ${{VLAFORGE_GENERATED_HEADERS}}
    DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}}/vlaforge/generated)
{runner}"""


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(
        prefix + line if line else "" for line in text.splitlines()
    )


def _indent_lines(lines: list[str], spaces: int) -> list[str]:
    prefix = " " * spaces
    return [prefix + line if line else "" for line in lines]


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
