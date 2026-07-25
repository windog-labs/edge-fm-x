"""Deterministic C++ Session generation for Invocation IR v0.2."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from vlaforge.codegen.model import (
    CppArtifactRegionDefinition,
    CppRegionDefinition,
    CppValidatorDefinition,
    GeneratedSources,
)
from vlaforge.ir.program import InputPort, Module
from vlaforge.ir.serializer import io_schema_digest, module_digest
from vlaforge.ir.types import (
    InputRevisionType,
    IRType,
    ScalarType,
    SnapshotType,
    TensorType,
)
from vlaforge.plan import (
    BufferClass,
    PlanModule,
    emit_memory_constants,
    verify_plan,
)
from vlaforge.plan.memory import state_arena_sizes, storage_size_bytes


class CodegenUnsupportedError(ValueError):
    pass


def generate_cpp_session(
    plan: PlanModule,
    semantic_module: Module,
    *,
    regions: Mapping[str, CppRegionDefinition] | None = None,
    artifact_regions: Mapping[
        str, CppArtifactRegionDefinition
    ] | None = None,
    validators: Mapping[str, CppValidatorDefinition],
    runner_source: str | None = None,
    namespace: str = "vlaforge_generated",
    compilation_certificate: object | None = None,
    initial_state: Mapping[str, object] | None = None,
) -> GeneratedSources:
    verify_plan(plan)
    if plan.arena is None:
        raise ValueError("C++ codegen requires a physicalized plan")
    if plan.semantic_digest != module_digest(semantic_module):
        raise ValueError("C++ codegen source module digest mismatch")
    if plan.io_schema_digest != io_schema_digest(semantic_module):
        raise ValueError("C++ codegen I/O schema digest mismatch")
    if len(plan.invocations) != 1:
        raise CodegenUnsupportedError(
            "static C++ v0.2 currently requires one invocation"
        )
    if not _IDENTIFIER.fullmatch(namespace):
        raise ValueError("C++ namespace must be an identifier")
    if compilation_certificate is not None:
        certificate_plan = getattr(
            compilation_certificate, "plan_digest", plan.digest()
        )
        if certificate_plan != plan.digest():
            raise ValueError("compilation certificate plan digest mismatch")

    inline_regions = dict(regions or {})
    compiled_regions = dict(artifact_regions or {})
    if inline_regions and compiled_regions:
        raise CodegenUnsupportedError(
            "static C++ v0.2 does not mix inline and artifact Regions"
        )
    required_regions = {item.region_name for item in plan.artifacts}
    required_validators = {
        str(task.attributes["contract"])
        for task in plan.tasks
        if task.opcode == "vla.validate"
    }
    provided_regions = set(inline_regions) | set(compiled_regions)
    missing_regions = sorted(required_regions - provided_regions)
    extra_regions = sorted(provided_regions - required_regions)
    missing_validators = sorted(required_validators - set(validators))
    if missing_regions or extra_regions or missing_validators:
        raise CodegenUnsupportedError(
            "C++ definitions mismatch: "
            f"missing_regions={missing_regions}, "
            f"extra_regions={extra_regions}, "
            f"missing_validators={missing_validators}"
        )
    for name, definition in compiled_regions.items():
        if definition.region_name != name:
            raise CodegenUnsupportedError(
                f"artifact Region key/name mismatch: {name!r} != "
                f"{definition.region_name!r}"
            )
        if definition.io_schema_digest != plan.io_schema_digest:
            raise CodegenUnsupportedError(
                f"artifact Region {name}: I/O schema digest mismatch"
            )

    emitter = _Emitter(
        plan,
        semantic_module,
        inline_regions,
        compiled_regions,
        validators,
        namespace,
        dict(initial_state or {}),
    )
    files = {
        "CMakeLists.txt": _cmake_source(
            runner_source is not None,
            artifact_backend=bool(compiled_regions),
        ),
        "memory_constants.h": emit_memory_constants(
            plan, namespace=namespace
        ),
        "session_generated.cpp": emitter.source(),
        "session_generated.h": emitter.header(),
    }
    if runner_source is not None:
        files["runner.cpp"] = runner_source
    return GeneratedSources(tuple(sorted(files.items())))


def generate_compiled_cpp_session(
    compilation: object,
    *,
    regions: Mapping[str, CppRegionDefinition] | None = None,
    artifact_regions: Mapping[
        str, CppArtifactRegionDefinition
    ] | None = None,
    validators: Mapping[str, CppValidatorDefinition],
    runner_source: str | None = None,
    namespace: str = "vlaforge_generated",
    initial_state: Mapping[str, object] | None = None,
) -> GeneratedSources:
    return generate_cpp_session(
        compilation.plan,
        compilation.module,
        regions=regions,
        artifact_regions=artifact_regions,
        validators=validators,
        runner_source=runner_source,
        namespace=namespace,
        compilation_certificate=getattr(compilation, "certificate", None),
        initial_state=initial_state,
    )


@dataclass(frozen=True, slots=True)
class _Cache:
    task_id: int
    outputs: tuple[int, ...]
    input_ids: tuple[int, ...]
    state_ids: tuple[int, ...]


class _Emitter:
    def __init__(
        self,
        plan: PlanModule,
        module: Module,
        regions: Mapping[str, CppRegionDefinition],
        artifact_regions: Mapping[str, CppArtifactRegionDefinition],
        validators: Mapping[str, CppValidatorDefinition],
        namespace: str,
        initial_state: Mapping[str, object],
    ):
        self.plan = plan
        self.module = module
        self.regions = dict(regions)
        self.artifact_regions = dict(artifact_regions)
        self.artifact_mode = bool(self.artifact_regions)
        self.validators = dict(validators)
        self.namespace = namespace
        self.initial_state = dict(initial_state)
        unknown_initial = sorted(
            set(self.initial_state) - {state.name for state in module.states}
        )
        if unknown_initial:
            raise KeyError(
                f"initial state references unknown slots: {unknown_initial}"
            )
        self.input_ids = {
            port.name: int(port.input_id) for port in module.inputs
        }
        self.output_ids = {
            port.name: int(port.output_id) for port in module.outputs
        }
        self.output_group_ids = {
            name: index
            for index, name in enumerate(
                sorted({port.group for port in module.outputs})
            )
        }
        self.state_ids = {
            state.name: index for index, state in enumerate(module.states)
        }
        assert plan.arena is not None
        self.physical = {
            logical_id: physical
            for physical in plan.arena.physical_buffers
            for logical_id in physical.logical_buffers
        }
        caches = []
        for task in plan.tasks:
            if task.opcode != "vla.invoke":
                continue
            region = module.region(str(task.attributes["region"]))
            if not bool(region.metadata.get("memoize", False)):
                continue
            caches.append(
                _Cache(
                    task.id,
                    task.outputs,
                    tuple(
                        self.input_ids[str(name)]
                        for name in region.metadata.get(
                            "cache_input_ports",
                            tuple(port.name for port in module.inputs),
                        )
                    ),
                    tuple(
                        self.state_ids[str(name)]
                        for name in region.metadata.get(
                            "cache_state_slots",
                            tuple(state.name for state in module.states),
                        )
                    ),
                )
            )
        self.caches = tuple(caches)
        for cache in self.caches:
            for output in cache.outputs:
                physical = self.physical.get(output)
                if (
                    physical is None
                    or physical.buffer_class is not BufferClass.DERIVED_CACHE
                    or len(physical.logical_buffers) != 1
                ):
                    raise CodegenUnsupportedError(
                        "exact cache output requires dedicated persistent "
                        "derived-cache storage"
                    )
        self.max_region_inputs = max(
            (len(region.inputs) for region in module.regions),
            default=1,
        )
        self.max_region_outputs = max(
            (len(region.outputs) for region in module.regions),
            default=1,
        )

    def header(self) -> str:
        input_enum = "\n".join(
            f"  k{_camel(port.name)} = {port.input_id}u,"
            for port in self.module.inputs
        )
        output_enum = "\n".join(
            f"  k{_camel(port.name)} = {port.output_id}u,"
            for port in self.module.outputs
        )
        input_fields = "\n".join(
            self._typed_input_field(port) for port in self.module.inputs
        )
        output_fields = "\n".join(
            self._typed_output_field(port) for port in self.module.outputs
        )
        cache_fields = "\n".join(
            self._cache_field(cache) for cache in self.caches
        )
        artifact_include = "#include <string>" if self.artifact_mode else ""
        constructors = (
            """  explicit ModelSession(const char* bundle_root = ".");
  ~ModelSession() override;"""
            if self.artifact_mode
            else """  ModelSession();
  ~ModelSession() override = default;"""
        )
        artifact_methods = (
            """  vlaforge::runtime::Status InitializeRegions(
      const char* bundle_root) noexcept;
  vlaforge::runtime::Status LoadRegion(std::size_t slot) noexcept;
  vlaforge::runtime::Status FailArtifactRegion(
      vlaforge::runtime::Status status, std::size_t slot) noexcept;
  void DestroyRegion(std::size_t slot) noexcept;
  void DestroyRegions() noexcept;"""
            if self.artifact_mode
            else ""
        )
        artifact_fields = (
            f"""  std::array<VLAForgeRegionExecutable*,
             {len(self.module.regions)}> region_executables_{{}};
  std::array<const VLAForgeRegionExecutableValueApi*,
             {len(self.module.regions)}> region_apis_{{}};
  std::array<std::string, {len(self.module.regions)}> region_paths_{{}};"""
            if self.artifact_mode
            else ""
        )
        bundle_factory = (
            """extern "C" VLAForgeStatus vlaforge_model_session_create_from_bundle(
    const char* bundle_root, size_t bundle_root_size,
    VLAForgeSession** session);"""
            if self.artifact_mode
            else ""
        )
        return f"""#ifndef VLAFORGE_GENERATED_SESSION_H_
#define VLAFORGE_GENERATED_SESSION_H_

#include <array>
#include <cstddef>
#include <cstdint>
{artifact_include}

#include "vlaforge/runtime/session.h"
#include "vlaforge/runtime/state_store.h"
#include "vlaforge/runtime/static_arena.h"
#include "vlaforge/runtime/transaction.h"

namespace {self.namespace} {{

inline constexpr char kSchemaDigest[] =
    "{io_schema_digest(self.module)}";

enum class InputId : std::uint32_t {{
{input_enum}
}};

enum class OutputId : std::uint32_t {{
{output_enum}
}};

struct ModelInputs final {{
{input_fields}
}};

struct ModelOutputs final {{
{output_fields}
}};

class ModelSession final : public vlaforge::runtime::Session {{
 public:
{constructors}

  ModelSession(const ModelSession&) = delete;
  ModelSession& operator=(const ModelSession&) = delete;

  vlaforge::runtime::Status BindTensor(
      std::uint32_t input_id, const VLAForgeBoundTensor& input,
      const VLAForgeInputStamp* stamp) noexcept override;
  vlaforge::runtime::Status BindScalar(
      std::uint32_t input_id, const VLAForgeScalarValue& input,
      const VLAForgeInputStamp* stamp) noexcept override;
  vlaforge::runtime::Status Run() noexcept override;
  vlaforge::runtime::Status Run(
      const ModelInputs& inputs, ModelOutputs* outputs) noexcept;
  vlaforge::runtime::Status ReadOutputTensor(
      std::uint32_t output_id,
      VLAForgeBoundTensor* output) const noexcept override;
  vlaforge::runtime::Status ReadOutputScalar(
      std::uint32_t output_id,
      VLAForgeScalarValue* output) const noexcept override;
  vlaforge::runtime::Status ResetEpisode(
      std::uint64_t new_episode) noexcept override;
  const char* SchemaDigest() const noexcept override {{
    return kSchemaDigest;
  }}
  void SetTraceSink(
      vlaforge::runtime::TraceSink trace) noexcept override;
  [[nodiscard]] vlaforge::runtime::Status initialization_status()
      const noexcept {{
    return initialization_status_;
  }}
  vlaforge::runtime::Status InitializeStateTensor(
      std::uint32_t state_id,
      const VLAForgeBoundTensor& value) noexcept;
  vlaforge::runtime::Status InitializeStateScalar(
      std::uint32_t state_id,
      const VLAForgeScalarValue& value) noexcept;

 private:
  struct BoundInput final {{
    bool bound = false;
    bool tensor = false;
    VLAForgeBoundTensor tensor_value{{}};
    VLAForgeScalarValue scalar_value{{}};
    std::uint64_t revision = 0;
    std::uint64_t timestamp_ns = 0;
  }};

  void InitializeValues() noexcept;
  void ClearBindings() noexcept;
{artifact_methods}
  vlaforge::runtime::Status PrepareInputs() noexcept;
  vlaforge::runtime::Status Fail(
      vlaforge::runtime::Status status) noexcept;
  void* BufferData(std::uint32_t logical_id) noexcept;

  vlaforge::runtime::StaticArena arena_;
  vlaforge::runtime::StaticArena state_arena_;
  vlaforge::runtime::StateStore state_store_;
  vlaforge::runtime::Transaction transaction_;
  vlaforge::runtime::TraceSink trace_{{}};
  vlaforge::runtime::Status initialization_status_{{}};
  std::array<BoundInput, {len(self.module.inputs)}> inputs_{{}};
  std::array<VLAForgeTensorView, {len(self.plan.buffers)}> values_{{}};
  std::array<vlaforge::runtime::StateSnapshot,
             {max(len(self.plan.buffers), 1)}> snapshots_{{}};
  std::array<std::uint64_t, {max(len(self.module.inputs), 1)}>
      input_revisions_{{}};
  std::array<std::uint64_t, {max(len(self.module.states), 1)}>
      state_versions_{{}};
  std::array<VLAForgeBoundTensor, {max(len(self.module.outputs), 1)}>
      tensor_outputs_{{}};
  std::array<VLAForgeScalarValue, {max(len(self.module.outputs), 1)}>
      scalar_outputs_{{}};
  std::array<bool, {max(len(self.module.outputs), 1)}>
      output_valid_{{}};
  std::uint64_t next_auto_revision_ = 1;
  std::uint64_t run_index_ = 0;
{artifact_fields}
{cache_fields}
}};

using GeneratedSession = ModelSession;

}}  // namespace {self.namespace}

extern "C" VLAForgeStatus vlaforge_model_session_create(
    VLAForgeSession** session);
{bundle_factory}
extern "C" const VLAForgeSessionApi* vlaforge_model_session_api(void);

#endif  // VLAFORGE_GENERATED_SESSION_H_
"""

    def source(self) -> str:
        artifact_headers = (
            """#include "vlaforge/backends/aoti_region_executable.h"
#include "vlaforge/runtime/artifact_verifier.h" """
            if self.artifact_mode
            else ""
        )
        return "\n".join(
            (
                '#include "session_generated.h"',
                '#include "memory_constants.h"',
                '#include "vlaforge/runtime/device_copy.h"',
                "",
                "#include <algorithm>",
                "#include <array>",
                "#include <cmath>",
                "#include <cstddef>",
                "#include <cstdint>",
                "#include <cstring>",
                "#include <new>",
                "",
                artifact_headers,
                "",
                self._local_executable(),
                "",
                "namespace {",
                self._support_tables(),
                self._artifact_tables(),
                self._support_functions(),
                self._region_functions(),
                self._validator_functions(),
                "}  // namespace",
                "",
                f"namespace {self.namespace} {{",
                self._session_source(),
                f"}}  // namespace {self.namespace}",
                "",
                self._c_abi_source(),
                "",
            )
        )

    def _local_executable(self) -> str:
        if self.artifact_mode:
            return ""
        return f"""struct VLAForgeRegionExecutable {{
  std::array<VLAForgeTensorView, {self.max_region_inputs}> inputs{{}};
  std::array<VLAForgeTensorView, {self.max_region_outputs}> outputs{{}};
}};"""

    def _artifact_tables(self) -> str:
        if not self.artifact_mode:
            return ""
        definitions = []
        for index, region in enumerate(self.module.regions):
            artifact = self.artifact_regions[region.name]
            digest = ", ".join(
                f"0x{artifact.artifact_sha256[offset:offset + 2]}u"
                for offset in range(0, 64, 2)
            )
            variant = artifact.backend_variant or ""
            definitions.append(
                f"""constexpr char kArtifactPath{index}[] =
    "{artifact.artifact_path}";
constexpr char kArtifactShaHex{index}[] =
    "{artifact.artifact_sha256}";
constexpr std::uint8_t kArtifactSha{index}[] = {{{digest}}};
constexpr char kArtifactTarget{index}[] = "{artifact.target}";
constexpr char kArtifactVariant{index}[] = "{variant}";
constexpr bool kArtifactInvocationResident{index} =
    {str(artifact.residency == "invocation").lower()};"""
            )
        return "\n\n".join(definitions)

    def _support_tables(self) -> str:
        assert self.plan.arena is not None
        logical = []
        shapes = []
        for buffer in self.plan.buffers:
            physical = self.physical.get(buffer.id)
            if physical is None:
                logical.append("  {static_cast<std::size_t>(-1), 0u, 1u},")
            else:
                logical.append(
                    f"  {{{physical.offset}u, {physical.size_bytes}u, "
                    f"{physical.alignment}u}},"
                )
            type_ = buffer.type
            if isinstance(type_, TensorType) and type_.shape:
                dimensions = ", ".join(str(int(item)) for item in type_.shape)
                shapes.append(
                    f"constexpr std::int64_t kShape{buffer.id}[] = "
                    f"{{{dimensions}}};"
                )
        state_sizes = state_arena_sizes(self.plan)
        if len(state_sizes) > 1:
            raise CodegenUnsupportedError("generated session supports one state arena")
        state_device = next(iter(state_sizes), "cpu")
        state_size = next(iter(state_sizes.values()), 0)
        state_alignment = max(
            (state.alignment or 1 for state in self.plan.states),
            default=1,
        )
        states = "\n".join(
            "  {"
            f"{state.state_id}u, {state.slot_capacity}u, "
            f"{_bytes(state.payload)}u, {state.slot_size_bytes}u, "
            f"{state.alignment}u, "
            f"{state.offset}u, "
            f"{str(self.module.states[state.state_id].reset_on_episode).lower()}"
            "},"
            for state in self.plan.states
        )
        state_table = (
            "constexpr vlaforge::runtime::StateSlotDescriptor "
            "kStateSlots[] = {\n"
            f"{states}\n"
            "};"
            if self.plan.states
            else (
                "constexpr const vlaforge::runtime::StateSlotDescriptor* "
                "kStateSlots = nullptr;"
            )
        )
        return f"""
struct LogicalDesc {{
  std::size_t offset;
  std::size_t size;
  std::size_t alignment;
}};

constexpr LogicalDesc kLogical[] = {{
{chr(10).join(logical)}
}};

{chr(10).join(shapes)}

constexpr std::size_t kStateArenaSize = {state_size}u;
constexpr std::size_t kStateArenaAlignment = {state_alignment}u;
constexpr VLAForgeDevice kStateArenaDevice{{
    {_device(state_device)}, {_device_ordinal(state_device)}}};
{state_table}
"""

    def _support_functions(self) -> str:
        inline_helpers = (
            ""
            if self.artifact_mode
            else """
template <typename T>
const T* Input(const VLAForgeRegionExecutable* executable,
               std::size_t index) {
  return static_cast<const T*>(executable->inputs[index].data);
}

template <typename T>
T* Output(VLAForgeRegionExecutable* executable, std::size_t index) {
  return static_cast<T*>(executable->outputs[index].data);
}

bool CheckTensor(const VLAForgeTensorView& view, VLAForgeDType dtype,
                 std::size_t elements) {
  std::size_t element_size = 0;
  switch (dtype) {
    case VLAFORGE_DTYPE_BOOL: element_size = 1; break;
    case VLAFORGE_DTYPE_I32: element_size = 4; break;
    case VLAFORGE_DTYPE_I64:
    case VLAFORGE_DTYPE_U64:
    case VLAFORGE_DTYPE_F64: element_size = 8; break;
    case VLAFORGE_DTYPE_F16:
    case VLAFORGE_DTYPE_BF16: element_size = 2; break;
    case VLAFORGE_DTYPE_F32: element_size = 4; break;
    case VLAFORGE_DTYPE_U8: element_size = 1; break;
    default: return false;
  }
  return view.data != nullptr && view.dtype == dtype &&
         view.size_bytes == elements * element_size;
}
"""
        )
        return inline_helpers + """

vlaforge::runtime::Status FromCStatus(VLAForgeStatus status,
                                     std::uint32_t subject) {
  using vlaforge::runtime::Status;
  using vlaforge::runtime::StatusCode;
  switch (status.code) {
    case VLAFORGE_STATUS_OK:
      return Status::Ok();
    case VLAFORGE_STATUS_INVALID_ARGUMENT:
      return Status::Error(StatusCode::kInvalidArgument, subject,
                           status.message);
    case VLAFORGE_STATUS_NOT_FOUND:
      return Status::Error(StatusCode::kNotFound, subject, status.message);
    case VLAFORGE_STATUS_OUT_OF_MEMORY:
      return Status::Error(StatusCode::kResourceExhausted, subject,
                           status.message);
    case VLAFORGE_STATUS_FAILED_PRECONDITION:
      return Status::Error(StatusCode::kFailedPrecondition, subject,
                           status.message);
    case VLAFORGE_STATUS_UNSUPPORTED_ABI:
    case VLAFORGE_STATUS_IO_ERROR:
    case VLAFORGE_STATUS_BACKEND_ERROR:
    case VLAFORGE_STATUS_INTERNAL:
      return Status::Error(StatusCode::kInternal, subject, status.message);
  }
  return Status::Error(StatusCode::kInternal, subject,
                       "unknown C Region status code");
}

VLAForgeStatus ToCStatus(vlaforge::runtime::Status status) {
  using vlaforge::runtime::StatusCode;
  switch (status.code) {
    case StatusCode::kOk:
      return vlaforge_status_ok();
    case StatusCode::kInvalidArgument:
    case StatusCode::kOutOfRange:
      return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                   status.message);
    case StatusCode::kNotFound:
      return vlaforge_status_error(VLAFORGE_STATUS_NOT_FOUND,
                                   status.message);
    case StatusCode::kAlreadyExists:
    case StatusCode::kFailedPrecondition:
    case StatusCode::kValidationFailed:
      return vlaforge_status_error(VLAFORGE_STATUS_FAILED_PRECONDITION,
                                   status.message);
    case StatusCode::kResourceExhausted:
      return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                   status.message);
    case StatusCode::kInternal:
      return vlaforge_status_error(VLAFORGE_STATUS_INTERNAL,
                                   status.message);
  }
  return vlaforge_status_error(VLAFORGE_STATUS_INTERNAL,
                               "unknown C++ Session status code");
}

void* ScalarData(VLAForgeScalarValue* value) {
  switch (value->dtype) {
    case VLAFORGE_DTYPE_BOOL: return &value->value.boolean;
    case VLAFORGE_DTYPE_I32: return &value->value.i32;
    case VLAFORGE_DTYPE_I64: return &value->value.i64;
    case VLAFORGE_DTYPE_U64: return &value->value.u64;
    case VLAFORGE_DTYPE_F32: return &value->value.f32;
    case VLAFORGE_DTYPE_F64: return &value->value.f64;
    default: return nullptr;
  }
}

std::size_t ScalarBytes(VLAForgeDType dtype) {
  switch (dtype) {
    case VLAFORGE_DTYPE_BOOL: return 1u;
    case VLAFORGE_DTYPE_I32: return 4u;
    case VLAFORGE_DTYPE_I64:
    case VLAFORGE_DTYPE_U64:
    case VLAFORGE_DTYPE_F64: return 8u;
    case VLAFORGE_DTYPE_F32: return 4u;
    default: return 0u;
  }
}

[[maybe_unused]] VLAForgeTensorView ScalarView(
    VLAForgeScalarValue* scalar) {
  return VLAForgeTensorView{ScalarData(scalar), ScalarBytes(scalar->dtype),
                            nullptr, 0u, scalar->dtype,
                            {VLAFORGE_DEVICE_CPU, 0}};
}

vlaforge::runtime::Status ReadBool(
    const VLAForgeTensorView& view, bool* result) {
  if (result == nullptr || view.data == nullptr || view.size_bytes < 1u) {
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kInvalidArgument, 0u,
        "invalid boolean control value");
  }
  std::uint8_t value = 0u;
  auto status = vlaforge::runtime::CopyBytes(
      &value, {VLAFORGE_DEVICE_CPU, 0}, view.data, view.device, 1u);
  if (status.ok()) {
    *result = value != 0u;
  }
  return status;
}

[[maybe_unused]] vlaforge::runtime::Status CopyValue(
    const VLAForgeTensorView& source,
    const VLAForgeTensorView& target) {
  if (source.size_bytes != target.size_bytes) {
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kInvalidArgument, 0u,
        "value copy size mismatch");
  }
  return vlaforge::runtime::CopyBytes(
      target.data, target.device, source.data, source.device,
      source.size_bytes);
}
"""

    def _region_functions(self) -> str:
        if self.artifact_mode:
            return ""
        functions = []
        for index, region in enumerate(self.module.regions):
            body = self.regions[region.name].body
            functions.append(
                f"""VLAForgeStatus RunRegion{index}(
    VLAForgeRegionExecutable* executable) {{
{_indent(body, 2)}
}}"""
            )
        return "\n\n".join(functions)

    def _artifact_session_methods(self) -> str:
        if not self.artifact_mode:
            return ""
        verify = []
        load_cases = []
        for index, region in enumerate(self.module.regions):
            artifact = self.artifact_regions[region.name]
            kind = _device(artifact.device)
            ordinal = _device_ordinal(artifact.device)
            variant_pointer = (
                f"kArtifactVariant{index}"
                if artifact.backend_variant is not None
                else "nullptr"
            )
            variant_size = (
                f"sizeof(kArtifactVariant{index}) - 1u"
                if artifact.backend_variant is not None
                else "0u"
            )
            verify.append(
                f"""  {{
    auto verify_status = vlaforge::runtime::VerifyArtifactFile(
        bundle_root, kArtifactPath{index}, kArtifactShaHex{index},
        {artifact.artifact_size_bytes}u, &region_paths_[{index}u]);
    if (!verify_status.ok()) {{
      DestroyRegions();
      return verify_status;
    }}
    const auto* api = vlaforge_aoti_region_executable_value_api();
    auto c_status = vlaforge_region_executable_value_api_validate(api);
    if (c_status.code != VLAFORGE_STATUS_OK) {{
      DestroyRegions();
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kFailedPrecondition, {index}u,
          "AOTI Region value ABI validation failed");
    }}
    region_apis_[{index}u] = api;
    if (!kArtifactInvocationResident{index}) {{
      auto load_status = LoadRegion({index}u);
      if (!load_status.ok()) {{
        DestroyRegions();
        return load_status;
      }}
    }}
  }}"""
            )
            load_cases.append(
                f"""    case {index}u: {{
      if (region_executables_[{index}u] != nullptr) {{
        return vlaforge::runtime::Status::Ok();
      }}
      const auto* api = region_apis_[{index}u];
      if (api == nullptr || region_paths_[{index}u].empty()) {{
        return vlaforge::runtime::Status::Error(
            vlaforge::runtime::StatusCode::kFailedPrecondition, {index}u,
            "AOTI Region artifact is not verified");
      }}
    const VLAForgeRegionCreateOptions options{{
        sizeof(VLAForgeRegionCreateOptions),
        VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
        {index}u, {{{kind}, {ordinal}}}}};
      auto c_status =
          api->create(&options, &region_executables_[{index}u]);
    if (c_status.code != VLAFORGE_STATUS_OK) {{
        DestroyRegion({index}u);
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kInternal, {index}u,
          "AOTI Region creation failed");
    }}
    const VLAForgeArtifactDescriptor descriptor{{
        sizeof(VLAForgeArtifactDescriptor),
        VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
        region_paths_[{index}u].data(), region_paths_[{index}u].size(),
        kArtifactSha{index}, {artifact.artifact_size_bytes}u,
        kSchemaDigest, VLAFORGE_SCHEMA_DIGEST_HEX_SIZE,
        kArtifactTarget{index}, sizeof(kArtifactTarget{index}) - 1u,
        {variant_pointer}, {variant_size}}};
    c_status = api->load(region_executables_[{index}u], &descriptor);
    if (c_status.code != VLAFORGE_STATUS_OK) {{
        DestroyRegion({index}u);
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kFailedPrecondition, {index}u,
          "AOTI Region load failed");
    }}
    VLAForgeWorkspaceRequirement requirement{{}};
    c_status = api->query_workspace(region_executables_[{index}u],
                                    &requirement);
    if (c_status.code != VLAFORGE_STATUS_OK ||
        requirement.size_bytes != 0u) {{
        DestroyRegion({index}u);
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kFailedPrecondition, {index}u,
          "AOTI Region workspace contract mismatch");
    }}
    c_status = api->bind_workspace(region_executables_[{index}u],
                                   nullptr, 0u);
    if (c_status.code != VLAFORGE_STATUS_OK) {{
        DestroyRegion({index}u);
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kFailedPrecondition, {index}u,
          "AOTI Region workspace binding failed");
    }}
      return vlaforge::runtime::Status::Ok();
    }}"""
            )
        return f"""ModelSession::~ModelSession() {{ DestroyRegions(); }}

void ModelSession::DestroyRegion(std::size_t slot) noexcept {{
  if (slot >= region_executables_.size()) {{
    return;
  }}
  if (region_apis_[slot] != nullptr &&
      region_executables_[slot] != nullptr) {{
    region_apis_[slot]->destroy(region_executables_[slot]);
    region_executables_[slot] = nullptr;
  }}
}}

void ModelSession::DestroyRegions() noexcept {{
  for (std::size_t index = region_executables_.size(); index > 0u; --index) {{
    DestroyRegion(index - 1u);
  }}
}}

vlaforge::runtime::Status ModelSession::FailArtifactRegion(
    vlaforge::runtime::Status status, std::size_t slot) noexcept {{
  DestroyRegion(slot);
  return Fail(status);
}}

vlaforge::runtime::Status ModelSession::LoadRegion(
    std::size_t slot) noexcept {{
  switch (slot) {{
{chr(10).join(load_cases)}
    default:
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kInvalidArgument, 0u,
          "artifact Region slot is out of range");
  }}
}}

vlaforge::runtime::Status ModelSession::InitializeRegions(
    const char* bundle_root) noexcept {{
  if (bundle_root == nullptr || bundle_root[0] == '\\0') {{
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kInvalidArgument, 0u,
        "bundle root is empty");
  }}
{chr(10).join(verify)}
  return vlaforge::runtime::Status::Ok();
}}"""

    def _validator_functions(self) -> str:
        functions = []
        for index, name in enumerate(sorted(self.validators)):
            body = self.validators[name].body
            functions.append(
                f"""bool Validate{index}(const void* data,
               std::size_t size_bytes) {{
{_indent(body, 2)}
}}"""
            )
        return "\n\n".join(functions)

    def _session_source(self) -> str:
        state_pointer = "kStateSlots"
        input_cases_tensor = "\n".join(
            self._bind_tensor_case(port)
            for port in self.module.inputs
            if isinstance(port.payload, TensorType)
        )
        input_cases_scalar = "\n".join(
            self._bind_scalar_case(port)
            for port in self.module.inputs
            if isinstance(port.payload, ScalarType)
        )
        prepare = "\n".join(
            self._prepare_input(port) for port in self.module.inputs
        )
        init_values = "\n".join(
            self._initialize_value(buffer.id, buffer.type)
            for buffer in self.plan.buffers
            if buffer.id in self.physical
        )
        cache_reset = "\n".join(
            f"  cache_{cache.task_id}_valid_ = false;"
            for cache in self.caches
        )
        run_body = self._emit_block(
            self.plan.invocations[0].body_block,
            indent=2,
        )
        typed_bind = "\n".join(
            self._typed_bind(port) for port in self.module.inputs
        )
        typed_read = "\n".join(
            self._typed_read(port) for port in self.module.outputs
        )
        state_init_tensor = self._state_init_cases(tensor=True)
        state_init_scalar = self._state_init_cases(tensor=False)
        initial_state = self._initial_state_source()
        constructor_signature = (
            "ModelSession::ModelSession(const char* bundle_root)"
            if self.artifact_mode
            else "ModelSession::ModelSession()"
        )
        artifact_initialize = (
            """  if (initialization_status_.ok()) {
    initialization_status_ = InitializeRegions(bundle_root);
  }"""
            if self.artifact_mode
            else ""
        )
        artifact_methods = self._artifact_session_methods()
        return f"""
{constructor_signature}
    : arena_(kArenaSize, kArenaAlignment,
             {{{_device(self.plan.arena.device)},
               {_device_ordinal(self.plan.arena.device)}}}),
      state_arena_(kStateArenaSize, kStateArenaAlignment,
                   kStateArenaDevice),
      state_store_(state_arena_, {state_pointer},
                   {len(self.plan.states)}u),
      transaction_({len(self.plan.states)}u) {{
  InitializeValues();
  initialization_status_ = state_store_.initialization_status();
{initial_state}
{artifact_initialize}
}}

{artifact_methods}

void ModelSession::InitializeValues() noexcept {{
{init_values}
}}

void ModelSession::ClearBindings() noexcept {{
  for (auto& input : inputs_) {{
    input.bound = false;
    input.tensor_value.tensor.data = nullptr;
  }}
}}

void* ModelSession::BufferData(std::uint32_t logical_id) noexcept {{
  if (logical_id >= sizeof(kLogical) / sizeof(kLogical[0])) {{
    return nullptr;
  }}
  const auto& item = kLogical[logical_id];
  if (item.offset == static_cast<std::size_t>(-1)) {{
    return nullptr;
  }}
  return arena_.Resolve(item.offset, item.size, item.alignment);
}}

vlaforge::runtime::Status ModelSession::BindTensor(
    std::uint32_t input_id, const VLAForgeBoundTensor& input,
    const VLAForgeInputStamp* stamp) noexcept {{
  (void)input;
  (void)stamp;
  switch (input_id) {{
{input_cases_tensor}
    default:
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kInvalidArgument, input_id,
          "unknown tensor input id or scalar input bound as tensor");
  }}
}}

vlaforge::runtime::Status ModelSession::BindScalar(
    std::uint32_t input_id, const VLAForgeScalarValue& input,
    const VLAForgeInputStamp* stamp) noexcept {{
  (void)input;
  (void)stamp;
  switch (input_id) {{
{input_cases_scalar}
    default:
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kInvalidArgument, input_id,
          "unknown scalar input id or tensor input bound as scalar");
  }}
}}

vlaforge::runtime::Status ModelSession::PrepareInputs() noexcept {{
{prepare}
  return vlaforge::runtime::Status::Ok();
}}

vlaforge::runtime::Status ModelSession::Fail(
    vlaforge::runtime::Status status) noexcept {{
  if (transaction_.active()) {{
    (void)state_store_.Abort(&transaction_, 0u);
  }}
  return status;
}}

vlaforge::runtime::Status ModelSession::Run() noexcept {{
  struct BindingReset final {{
    ModelSession* session;
    ~BindingReset() {{ session->ClearBindings(); }}
  }} reset{{this}};
  if (!initialization_status_.ok()) {{
    return initialization_status_;
  }}
  state_store_.SetRunIndex(run_index_);
  auto status = PrepareInputs();
  if (!status.ok()) {{
    return status;
  }}
{run_body}
  ++run_index_;
  return vlaforge::runtime::Status::Ok();
}}

vlaforge::runtime::Status ModelSession::Run(
    const ModelInputs& inputs, ModelOutputs* outputs) noexcept {{
  if (outputs == nullptr) {{
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kInvalidArgument, 0u,
        "typed outputs pointer is null");
  }}
{typed_bind}
  auto status = Run();
  if (!status.ok()) {{
    return status;
  }}
{typed_read}
  return vlaforge::runtime::Status::Ok();
}}

vlaforge::runtime::Status ModelSession::ReadOutputTensor(
    std::uint32_t output_id,
    VLAForgeBoundTensor* output) const noexcept {{
  if (output == nullptr || output_id >= output_valid_.size() ||
      !output_valid_[output_id]) {{
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kNotFound, output_id,
        "committed tensor output is unavailable");
  }}
  if (tensor_outputs_[output_id].struct_size == 0u) {{
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kInvalidArgument, output_id,
        "output is scalar, not tensor");
  }}
  *output = tensor_outputs_[output_id];
  return vlaforge::runtime::Status::Ok();
}}

vlaforge::runtime::Status ModelSession::ReadOutputScalar(
    std::uint32_t output_id,
    VLAForgeScalarValue* output) const noexcept {{
  if (output == nullptr || output_id >= output_valid_.size() ||
      !output_valid_[output_id]) {{
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kNotFound, output_id,
        "committed scalar output is unavailable");
  }}
  if (scalar_outputs_[output_id].struct_size == 0u) {{
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kInvalidArgument, output_id,
        "output is tensor, not scalar");
  }}
  *output = scalar_outputs_[output_id];
  return vlaforge::runtime::Status::Ok();
}}

vlaforge::runtime::Status ModelSession::ResetEpisode(
    std::uint64_t new_episode) noexcept {{
  auto status = state_store_.ResetEpisode(new_episode, 0u);
  if (status.ok()) {{
    output_valid_.fill(false);
{cache_reset}
  }}
  return status;
}}

void ModelSession::SetTraceSink(
    vlaforge::runtime::TraceSink trace) noexcept {{
  trace_ = trace;
  state_store_.SetTraceSink(trace);
}}

vlaforge::runtime::Status ModelSession::InitializeStateTensor(
    std::uint32_t state_id,
    const VLAForgeBoundTensor& value) noexcept {{
  (void)value;
  switch (state_id) {{
{state_init_tensor}
    default:
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kInvalidArgument, state_id,
          "unknown tensor state");
  }}
}}

vlaforge::runtime::Status ModelSession::InitializeStateScalar(
    std::uint32_t state_id,
    const VLAForgeScalarValue& value) noexcept {{
  (void)value;
  switch (state_id) {{
{state_init_scalar}
    default:
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kInvalidArgument, state_id,
          "unknown scalar state");
  }}
}}
"""

    def _emit_block(
        self,
        block_id: int,
        *,
        indent: int,
        task_ids: tuple[int, ...] | None = None,
    ) -> str:
        lines: list[str] = []
        block = self.plan.block(block_id)
        for task_id in block.tasks if task_ids is None else task_ids:
            task = self.plan.task(task_id)
            opcode = task.opcode
            if opcode == "vla.input.read":
                port_name = str(task.attributes["input"])
                input_id = self.input_ids[port_name]
                port = self.module.inputs[input_id]
                if isinstance(port.payload, TensorType):
                    lines.append(
                        f"values_[{task.outputs[0]}u] = "
                        f"inputs_[{input_id}u].tensor_value.tensor;"
                    )
                else:
                    lines.append(
                        f"values_[{task.outputs[0]}u] = "
                        f"ScalarView(&inputs_[{input_id}u].scalar_value);"
                    )
                lines.extend(
                    (
                        f"input_revisions_[{input_id}u] = "
                        f"inputs_[{input_id}u].revision;",
                        f"values_[{task.outputs[1]}u] = VLAForgeTensorView{{"
                        f"&input_revisions_[{input_id}u], "
                        "sizeof(std::uint64_t), nullptr, 0u, "
                        "VLAFORGE_DTYPE_U64, {VLAFORGE_DEVICE_CPU, 0}};",
                        "vlaforge::runtime::EmitTrace("
                        "trace_, vlaforge::runtime::TraceEvent{"
                        "vlaforge::runtime::TraceKind::kInput, "
                        f"{task.id}u, {input_id}u, 0u, 0u, "
                        "state_store_.episode(), run_index_, "
                        f"inputs_[{input_id}u].revision}});",
                    )
                )
            elif opcode == "vla.txn.begin":
                lines.extend(
                    (
                        f"status = state_store_.Begin(&transaction_, {task.id}u);",
                        "if (!status.ok()) { return Fail(status); }",
                    )
                )
            elif opcode == "vla.state.read_latest":
                state_id = self.state_ids[str(task.attributes["state"])]
                lines.extend(
                    (
                        f"status = state_store_.ReadLatest({state_id}u, "
                        f"{task.id}u, &snapshots_[{task.outputs[0]}u]);",
                        "if (!status.ok()) { return Fail(status); }",
                        f"state_versions_[{state_id}u] = "
                        f"snapshots_[{task.outputs[0]}u].logical_version;",
                    )
                )
            elif opcode == "vla.snapshot.value":
                snapshot_id = task.inputs[0]
                output_id = task.outputs[0]
                lines.extend(
                    (
                        f"values_[{output_id}u].data = "
                        f"const_cast<void*>(snapshots_[{snapshot_id}u].data);",
                        f"values_[{output_id}u].size_bytes = "
                        f"snapshots_[{snapshot_id}u].size_bytes;",
                        f"values_[{output_id}u].device = "
                        f"snapshots_[{snapshot_id}u].device;",
                    )
                )
            elif opcode == "vla.invoke":
                lines.extend(self._emit_region(task))
            elif opcode == "vla.for":
                lines.extend(self._emit_for(task))
            elif opcode == "vla.if":
                lines.extend(self._emit_if(task))
            elif opcode == "vla.yield":
                continue
            elif opcode == "vla.state.stage_write":
                state_id = self.state_ids[str(task.attributes["state"])]
                value = task.inputs[1]
                lines.extend(
                    (
                        f"status = state_store_.Stage(&transaction_, {state_id}u, "
                        f"values_[{value}u].data, values_[{value}u].size_bytes, "
                        f"{task.id}u, values_[{value}u].device);",
                        "if (!status.ok()) { return Fail(status); }",
                    )
                )
            elif opcode == "vla.validate":
                contract = str(task.attributes["contract"])
                validator_id = sorted(self.validators).index(contract)
                value = task.inputs[0]
                output = task.outputs[0]
                lines.extend(
                    (
                        f"std::array<std::byte, "
                        f"{max(_bytes(self.plan.buffer(value).type), 1)}u> "
                        f"validation_input_{task.id}{{}};",
                        "status = vlaforge::runtime::CopyBytes("
                        f"validation_input_{task.id}.data(), "
                        "{VLAFORGE_DEVICE_CPU, 0}, "
                        f"values_[{value}u].data, "
                        f"values_[{value}u].device, "
                        f"values_[{value}u].size_bytes, {task.id}u);",
                        "if (!status.ok()) { return Fail(status); }",
                        f"std::uint8_t validation_result_{task.id} = "
                        f"Validate{validator_id}("
                        f"validation_input_{task.id}.data(), "
                        f"values_[{value}u].size_bytes) ? 1u : 0u;",
                        f"values_[{output}u] = VLAForgeTensorView{{"
                        f"&validation_result_{task.id}, 1u, nullptr, 0u, "
                        "VLAFORGE_DTYPE_BOOL, "
                        "{VLAFORGE_DEVICE_CPU, 0}};",
                        "vlaforge::runtime::EmitTrace("
                        "trace_, vlaforge::runtime::TraceEvent{"
                        "vlaforge::runtime::TraceKind::kValidation, "
                        f"{task.id}u, {validator_id}u, 0u, transaction_.id(), "
                        "state_store_.episode(), run_index_, 0u});",
                    )
                )
            elif opcode == "vla.output.create":
                lines.append(
                    f"values_[{task.outputs[0]}u] = values_[{task.inputs[0]}u];"
                )
            elif opcode == "vla.output.group":
                group_id = self.output_group_ids[
                    str(task.attributes["group"])
                ]
                lines.append(
                    "vlaforge::runtime::EmitTrace("
                    "trace_, vlaforge::runtime::TraceEvent{"
                    "vlaforge::runtime::TraceKind::kOutputGroupPending, "
                    f"{task.id}u, {group_id}u, 0u, transaction_.id(), "
                    "state_store_.episode(), run_index_, 0u});"
                )
            elif opcode == "vla.txn.commit":
                lines.extend(self._emit_commit(task))
            elif opcode in {"vla.return", "vla.txn.abort"}:
                if opcode == "vla.txn.abort":
                    lines.extend(
                        (
                            f"status = state_store_.Abort(&transaction_, "
                            f"{task.id}u);",
                            "if (!status.ok()) { return status; }",
                        )
                    )
            else:
                raise CodegenUnsupportedError(
                    f"unsupported C++ task operation: {opcode}"
                )
        return _indent("\n".join(lines), indent)

    def _emit_region(self, task: Any) -> list[str]:
        region_name = str(task.attributes["region"])
        region_id = next(
            index
            for index, item in enumerate(self.module.regions)
            if item.name == region_name
        )
        if self.artifact_mode:
            return self._emit_artifact_region(task, region_id)
        run = [
            "{",
            "  VLAForgeRegionExecutable executable{};",
        ]
        run.extend(
            f"  executable.inputs[{index}u] = values_[{buffer_id}u];"
            for index, buffer_id in enumerate(task.inputs)
        )
        run.extend(
            f"  executable.outputs[{index}u] = values_[{buffer_id}u];"
            for index, buffer_id in enumerate(task.outputs)
        )
        run.extend(
            (
                f"  const auto region_status = RunRegion{region_id}(&executable);",
                "  status = FromCStatus(region_status, "
                f"{task.id}u);",
                "  if (!status.ok()) { return Fail(status); }",
                "  vlaforge::runtime::EmitTrace("
                "trace_, vlaforge::runtime::TraceEvent{"
                "vlaforge::runtime::TraceKind::kRegion, "
                f"{task.id}u, {region_id}u, 0u, transaction_.id(), "
                "state_store_.episode(), run_index_, 0u});",
                "}",
            )
        )
        cache = next(
            (item for item in self.caches if item.task_id == task.id),
            None,
        )
        if cache is None:
            return run
        equal_inputs = " && ".join(
            f"cache_{task.id}_revisions_[{index}u] == "
            f"input_revisions_[{index}u]"
            for index in cache.input_ids
        ) or "true"
        equal_states = " && ".join(
            f"cache_{task.id}_state_versions_[{index}u] == "
            f"state_versions_[{index}u]"
            for index in cache.state_ids
        ) or "true"
        miss_copy = [
            "vlaforge::runtime::EmitTrace("
            "trace_, vlaforge::runtime::TraceEvent{"
            "vlaforge::runtime::TraceKind::kCacheMiss, "
            f"{task.id}u, {region_id}u, 0u, transaction_.id(), "
            "state_store_.episode(), run_index_, 0u});",
            *run,
            f"cache_{task.id}_revisions_ = input_revisions_;",
            f"cache_{task.id}_state_versions_ = state_versions_;",
            f"cache_{task.id}_episode_ = state_store_.episode();",
            f"cache_{task.id}_valid_ = true;",
        ]
        return [
            f"if (cache_{task.id}_valid_ && "
            f"cache_{task.id}_episode_ == state_store_.episode() && "
            f"({equal_inputs}) && ({equal_states})) {{",
            "  vlaforge::runtime::EmitTrace("
            "trace_, vlaforge::runtime::TraceEvent{"
            "vlaforge::runtime::TraceKind::kCacheHit, "
            f"{task.id}u, {region_id}u, 0u, transaction_.id(), "
            "state_store_.episode(), run_index_, 0u});",
            "} else {",
            *_indent_lines(miss_copy, 2),
            "}",
        ]

    def _emit_artifact_region(
        self, task: Any, region_id: int
    ) -> list[str]:
        definition = self.artifact_regions[
            self.module.regions[region_id].name
        ]
        invocation_resident = definition.residency == "invocation"
        fail_status = (
            f"FailArtifactRegion(status, {region_id}u)"
            if invocation_resident
            else "Fail(status)"
        )
        fail_uninitialized = (
            "FailArtifactRegion("
            "vlaforge::runtime::Status::Error("
            "vlaforge::runtime::StatusCode::kFailedPrecondition, "
            f"{task.id}u, \"artifact Region is not initialized\"), "
            f"{region_id}u)"
            if invocation_resident
            else (
                "Fail(vlaforge::runtime::Status::Error("
                "vlaforge::runtime::StatusCode::kFailedPrecondition, "
                f"{task.id}u, \"artifact Region is not initialized\"))"
            )
        )
        run = [
            "{",
        ]
        if invocation_resident:
            run.extend(
                (
                    f"  status = LoadRegion({region_id}u);",
                    "  if (!status.ok()) { return Fail(status); }",
                )
            )
        run.extend(
            (
            f"  const auto* api = region_apis_[{region_id}u];",
            f"  auto* executable = region_executables_[{region_id}u];",
            "  if (api == nullptr || executable == nullptr) {",
            f"    return {fail_uninitialized};",
            "  }",
            "  VLAForgeStatus region_status{};",
            )
        )
        for category, buffers in (
            ("input", task.inputs),
            ("output", task.outputs),
        ):
            for index, buffer_id in enumerate(buffers):
                buffer = self.plan.buffer(buffer_id)
                layout = (
                    _layout(buffer.type.layout)
                    if isinstance(buffer.type, TensorType)
                    else "VLAFORGE_LAYOUT_CONTIGUOUS"
                )
                physical = self.physical.get(buffer_id)
                alignment = 1 if physical is None else physical.alignment
                variable = f"region_{task.id}_{category}_{index}"
                run.extend(
                    (
                        f"  VLAForgeValueView {variable}{{}};",
                        f"  {variable}.struct_size = "
                        "sizeof(VLAForgeValueView);",
                        f"  {variable}.kind = VLAFORGE_VALUE_TENSOR;",
                        f"  {variable}.value.tensor = VLAForgeBoundTensor{{",
                        "      sizeof(VLAForgeBoundTensor), "
                        f"values_[{buffer_id}u], {layout}, {alignment}u}};",
                        f"  region_status = api->bind_{category}(",
                        f"      executable, {index}u, &{variable});",
                        "  status = FromCStatus(region_status, "
                        f"{task.id}u);",
                        "  if (!status.ok()) { "
                        f"return {fail_status}; }}",
                    )
                )
        run.extend(
            (
                "  region_status = api->run(executable);",
                "  status = FromCStatus(region_status, "
                f"{task.id}u);",
                "  if (!status.ok()) { "
                f"return {fail_status}; }}",
                "  region_status = api->synchronize(executable);",
                "  status = FromCStatus(region_status, "
                f"{task.id}u);",
                "  if (!status.ok()) { "
                f"return {fail_status}; }}",
                "  vlaforge::runtime::EmitTrace("
                "trace_, vlaforge::runtime::TraceEvent{"
                "vlaforge::runtime::TraceKind::kRegion, "
                f"{task.id}u, {region_id}u, 0u, transaction_.id(), "
                "state_store_.episode(), run_index_, 0u});",
            )
        )
        if invocation_resident:
            run.append(f"  DestroyRegion({region_id}u);")
        run.append("}")
        cache = next(
            (item for item in self.caches if item.task_id == task.id),
            None,
        )
        if cache is None:
            return run
        equal_inputs = " && ".join(
            f"cache_{task.id}_revisions_[{index}u] == "
            f"input_revisions_[{index}u]"
            for index in cache.input_ids
        ) or "true"
        equal_states = " && ".join(
            f"cache_{task.id}_state_versions_[{index}u] == "
            f"state_versions_[{index}u]"
            for index in cache.state_ids
        ) or "true"
        return [
            f"if (cache_{task.id}_valid_ && "
            f"cache_{task.id}_episode_ == state_store_.episode() && "
            f"({equal_inputs}) && ({equal_states})) {{",
            "  vlaforge::runtime::EmitTrace("
            "trace_, vlaforge::runtime::TraceEvent{"
            "vlaforge::runtime::TraceKind::kCacheHit, "
            f"{task.id}u, {region_id}u, 0u, transaction_.id(), "
            "state_store_.episode(), run_index_, 0u});",
            "} else {",
            "  vlaforge::runtime::EmitTrace("
            "trace_, vlaforge::runtime::TraceEvent{"
            "vlaforge::runtime::TraceKind::kCacheMiss, "
            f"{task.id}u, {region_id}u, 0u, transaction_.id(), "
            "state_store_.episode(), run_index_, 0u});",
            *_indent_lines(run, 2),
            f"  cache_{task.id}_revisions_ = input_revisions_;",
            f"  cache_{task.id}_state_versions_ = state_versions_;",
            f"  cache_{task.id}_episode_ = state_store_.episode();",
            f"  cache_{task.id}_valid_ = true;",
            "}",
        ]

    def _emit_for(self, task: Any) -> list[str]:
        body = self.plan.block(task.blocks[0])
        if len(body.arguments) != 2 or not body.tasks:
            raise CodegenUnsupportedError("bounded for has invalid body")
        terminal = self.plan.task(body.tasks[-1])
        if terminal.opcode != "vla.yield" or len(terminal.inputs) != 1:
            raise CodegenUnsupportedError("bounded for body must yield one value")
        result = task.outputs[0]
        lines = [
            f"status = CopyValue(values_[{task.inputs[0]}u], "
            f"values_[{result}u]);",
            "if (!status.ok()) { return Fail(status); }",
            f"for (std::int64_t loop_{task.id} = "
            f"{int(task.attributes['lower'])}; "
            f"loop_{task.id} < {int(task.attributes['upper'])}; "
            f"loop_{task.id} += {int(task.attributes['step'])}) {{",
            f"  VLAForgeScalarValue induction_{task.id}{{"
            "sizeof(VLAForgeScalarValue), VLAFORGE_DTYPE_I64, {}};",
            f"  induction_{task.id}.value.i64 = loop_{task.id};",
            f"  values_[{body.arguments[0]}u] = "
            f"ScalarView(&induction_{task.id});",
            f"  values_[{body.arguments[1]}u] = values_[{result}u];",
            self._emit_block_without_terminal(body.id, indent=2),
            f"  status = CopyValue(values_[{terminal.inputs[0]}u], "
            f"values_[{result}u]);",
            "  if (!status.ok()) { return Fail(status); }",
            "}",
        ]
        return lines

    def _emit_if(self, task: Any) -> list[str]:
        lines = [
            f"bool branch_condition_{task.id} = false;",
            f"status = ReadBool(values_[{task.inputs[0]}u], "
            f"&branch_condition_{task.id});",
            "if (!status.ok()) { return Fail(status); }",
            f"if (branch_condition_{task.id}) {{",
        ]
        for branch_index, block_id in enumerate(task.blocks):
            block = self.plan.block(block_id)
            terminal = self.plan.task(block.tasks[-1])
            if terminal.opcode != "vla.yield":
                raise CodegenUnsupportedError("if branch must end in yield")
            if branch_index == 1:
                lines.append("} else {")
            lines.append(self._emit_block_without_terminal(block_id, indent=2))
            for source, target in zip(
                terminal.inputs, task.outputs, strict=True
            ):
                lines.append(
                    f"  status = CopyValue(values_[{source}u], "
                    f"values_[{target}u]);"
                )
                lines.append(
                    "  if (!status.ok()) { return Fail(status); }"
                )
        lines.append("}")
        return lines

    def _emit_block_without_terminal(
        self,
        block_id: int,
        *,
        indent: int,
    ) -> str:
        block = self.plan.block(block_id)
        return self._emit_block(
            block_id,
            indent=indent,
            task_ids=block.tasks[:-1],
        )

    def _emit_commit(self, task: Any) -> list[str]:
        condition = task.inputs[2]
        group_buffer = self.plan.buffer(task.inputs[1])
        producer = self.plan.task(group_buffer.producer_task)
        if producer.opcode != "vla.output.group":
            raise CodegenUnsupportedError("commit input is not output.group")
        group_id = self.output_group_ids[str(producer.attributes["group"])]
        assignments = []
        for pending_buffer in producer.inputs:
            pending = self.plan.buffer(pending_buffer)
            create = self.plan.task(pending.producer_task)
            output_name = str(create.attributes["output"])
            output_id = self.output_ids[output_name]
            value_id = create.inputs[0]
            port = self.module.outputs[output_id]
            if isinstance(port.payload, TensorType):
                assignments.extend(
                    (
                        f"tensor_outputs_[{output_id}u] = "
                        "VLAForgeBoundTensor{"
                        "sizeof(VLAForgeBoundTensor), "
                        f"values_[{value_id}u], {_layout(port.payload.layout)}, "
                        f"{port.alignment}u}};",
                        f"scalar_outputs_[{output_id}u] = "
                        "VLAForgeScalarValue{};",
                    )
                )
            else:
                assignments.extend(
                    (
                        f"scalar_outputs_[{output_id}u] = "
                        f"VLAForgeScalarValue{{sizeof(VLAForgeScalarValue), "
                        f"{_dtype(port.payload.name)}, {{}}}};",
                        "status = vlaforge::runtime::CopyBytes("
                        f"ScalarData(&scalar_outputs_[{output_id}u]), "
                        "{VLAFORGE_DEVICE_CPU, 0}, "
                        f"values_[{value_id}u].data, "
                        f"values_[{value_id}u].device, "
                        f"values_[{value_id}u].size_bytes, {task.id}u);",
                        "if (!status.ok()) { return Fail(status); }",
                        f"tensor_outputs_[{output_id}u] = VLAForgeBoundTensor{{}};",
                    )
                )
            assignments.append(f"output_valid_[{output_id}u] = true;")
        return [
            f"bool commit_condition_{task.id} = false;",
            f"status = ReadBool(values_[{condition}u], "
            f"&commit_condition_{task.id});",
            "if (!status.ok()) { return Fail(status); }",
            f"if (!commit_condition_{task.id}) {{",
            f"  status = state_store_.Abort(&transaction_, {task.id}u);",
            "  if (!status.ok()) { return status; }",
            "  return vlaforge::runtime::Status::Error(",
            "      vlaforge::runtime::StatusCode::kValidationFailed, "
            f"{task.id}u, \"output validation failed\");",
            "}",
            f"status = state_store_.Commit(&transaction_, {task.id}u);",
            "if (!status.ok()) { return Fail(status); }",
            *assignments,
            "vlaforge::runtime::EmitTrace("
            "trace_, vlaforge::runtime::TraceEvent{"
            "vlaforge::runtime::TraceKind::kOutputGroupCommit, "
            f"{task.id}u, {group_id}u, 0u, transaction_.id(), "
            "state_store_.episode(), run_index_, 0u});",
        ]

    def _initialize_value(self, buffer_id: int, type_: IRType) -> str:
        if isinstance(type_, TensorType):
            shape = f"kShape{buffer_id}" if type_.shape else "nullptr"
            return (
                f"  values_[{buffer_id}u] = VLAForgeTensorView{{"
                f"BufferData({buffer_id}u), {_bytes(type_)}u, {shape}, "
                f"{len(type_.shape)}u, {_dtype(type_.dtype)}, "
                f"{{{_device(self.plan.arena.device)}, "
                f"{_device_ordinal(self.plan.arena.device)}}}}};"
            )
        if isinstance(type_, ScalarType | InputRevisionType):
            dtype = (
                _dtype(type_.name)
                if isinstance(type_, ScalarType)
                else "VLAFORGE_DTYPE_U64"
            )
            return (
                f"  values_[{buffer_id}u] = VLAForgeTensorView{{"
                f"BufferData({buffer_id}u), {_bytes(type_)}u, nullptr, 0u, "
                f"{dtype}, {{{_device(self.plan.arena.device)}, "
                f"{_device_ordinal(self.plan.arena.device)}}}}};"
            )
        return ""

    def _bind_tensor_case(self, port: InputPort) -> str:
        assert isinstance(port.payload, TensorType)
        checks = [
            "input.struct_size < sizeof(VLAForgeBoundTensor)",
            "input.tensor.data == nullptr",
            f"input.tensor.dtype != {_dtype(port.payload.dtype)}",
            f"input.tensor.rank != {len(port.payload.shape)}u",
            f"input.tensor.size_bytes != {_bytes(port.payload)}u",
            f"input.tensor.device.kind != {_device(port.device)}",
            f"input.layout != {_layout(port.payload.layout)}",
            f"input.alignment < {port.alignment}u",
            f"reinterpret_cast<std::uintptr_t>(input.tensor.data) % "
            f"{port.alignment}u != 0u",
        ]
        for index, dimension in enumerate(port.payload.shape):
            checks.append(
                f"input.tensor.dimensions == nullptr || "
                f"input.tensor.dimensions[{index}] != {int(dimension)}"
            )
        condition = " ||\n          ".join(checks)
        return f"""    case {port.input_id}u:
      if ({condition}) {{
        return vlaforge::runtime::Status::Error(
            vlaforge::runtime::StatusCode::kInvalidArgument, input_id,
            "tensor input contract mismatch");
      }}
      inputs_[{port.input_id}u].bound = true;
      inputs_[{port.input_id}u].tensor = true;
      inputs_[{port.input_id}u].tensor_value = input;
      inputs_[{port.input_id}u].revision =
          stamp != nullptr && stamp->has_revision
              ? stamp->revision : next_auto_revision_++;
      inputs_[{port.input_id}u].timestamp_ns =
          stamp != nullptr && stamp->has_timestamp
              ? stamp->timestamp_ns : 0u;
      return vlaforge::runtime::Status::Ok();"""

    def _bind_scalar_case(self, port: InputPort) -> str:
        assert isinstance(port.payload, ScalarType)
        range_check = ""
        if port.value_range is not None:
            member = _scalar_member(port.payload.name)
            lower, upper = port.value_range
            range_check = (
                f" || input.value.{member} < {_literal(lower)} || "
                f"input.value.{member} > {_literal(upper)}"
            )
        return f"""    case {port.input_id}u:
      if (input.struct_size < sizeof(VLAForgeScalarValue) ||
          input.dtype != {_dtype(port.payload.name)}{range_check}) {{
        return vlaforge::runtime::Status::Error(
            vlaforge::runtime::StatusCode::kInvalidArgument, input_id,
            "scalar input contract mismatch");
      }}
      inputs_[{port.input_id}u].bound = true;
      inputs_[{port.input_id}u].tensor = false;
      inputs_[{port.input_id}u].scalar_value = input;
      inputs_[{port.input_id}u].revision =
          stamp != nullptr && stamp->has_revision
              ? stamp->revision : next_auto_revision_++;
      inputs_[{port.input_id}u].timestamp_ns =
          stamp != nullptr && stamp->has_timestamp
              ? stamp->timestamp_ns : 0u;
      return vlaforge::runtime::Status::Ok();"""

    def _prepare_input(self, port: InputPort) -> str:
        if port.required:
            return f"""  if (!inputs_[{port.input_id}u].bound) {{
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kFailedPrecondition,
        {port.input_id}u, "required input is not bound");
  }}"""
        return self._default_input(port)

    def _default_input(self, port: InputPort) -> str:
        if isinstance(port.payload, ScalarType):
            member = _scalar_member(port.payload.name)
            value = _literal(port.default)
            return f"""  if (!inputs_[{port.input_id}u].bound) {{
    inputs_[{port.input_id}u].bound = true;
    inputs_[{port.input_id}u].tensor = false;
    inputs_[{port.input_id}u].scalar_value =
        VLAForgeScalarValue{{sizeof(VLAForgeScalarValue),
                            {_dtype(port.payload.name)}, {{}}}};
    inputs_[{port.input_id}u].scalar_value.value.{member} = {value};
    inputs_[{port.input_id}u].revision = 0u;
  }}"""
        assert isinstance(port.payload, TensorType)
        values = ", ".join(
            _literal(item) for item in _flatten(port.default)
        )
        ctype = _cpp_scalar(port.payload.dtype)
        shape = ", ".join(str(int(item)) for item in port.payload.shape)
        return f"""  if (!inputs_[{port.input_id}u].bound) {{
    static {ctype} default_data_{port.input_id}[] = {{{values}}};
    static const std::int64_t default_shape_{port.input_id}[] = {{{shape}}};
    inputs_[{port.input_id}u].bound = true;
    inputs_[{port.input_id}u].tensor = true;
    inputs_[{port.input_id}u].tensor_value = VLAForgeBoundTensor{{
        sizeof(VLAForgeBoundTensor),
        {{default_data_{port.input_id}, {_bytes(port.payload)}u,
          default_shape_{port.input_id}, {len(port.payload.shape)}u,
          {_dtype(port.payload.dtype)}, {{{_device(port.device)}, 0}}}},
        {_layout(port.payload.layout)}, {port.alignment}u}};
    inputs_[{port.input_id}u].revision = 0u;
  }}"""

    def _typed_input_field(self, port: InputPort) -> str:
        type_name = (
            "VLAForgeBoundTensor"
            if isinstance(port.payload, TensorType)
            else "VLAForgeScalarValue"
        )
        optional = (
            f"  bool has_{_identifier(port.name)} = false;\n"
            if not port.required
            else ""
        )
        name = _identifier(port.name)
        return (
            f"{optional}  {type_name} {name}{{}};\n"
            f"  VLAForgeInputStamp {name}_stamp{{}};"
        )

    def _typed_output_field(self, port: Any) -> str:
        type_name = (
            "VLAForgeBoundTensor"
            if isinstance(port.payload, TensorType)
            else "VLAForgeScalarValue"
        )
        return f"  {type_name} {_identifier(port.name)}{{}};"

    def _typed_bind(self, port: InputPort) -> str:
        name = _identifier(port.name)
        call = "BindTensor" if isinstance(port.payload, TensorType) else "BindScalar"
        condition = f"if (inputs.has_{name}) " if not port.required else ""
        opening = condition + "{"
        return f"""  {opening}
    const auto* stamp = inputs.{name}_stamp.struct_size == 0u
        ? nullptr : &inputs.{name}_stamp;
    auto status = {call}({port.input_id}u, inputs.{name}, stamp);
    if (!status.ok()) {{ return status; }}
  }}"""

    def _typed_read(self, port: Any) -> str:
        name = _identifier(port.name)
        call = (
            "ReadOutputTensor"
            if isinstance(port.payload, TensorType)
            else "ReadOutputScalar"
        )
        return f"""  status = {call}({port.output_id}u, &outputs->{name});
  if (!status.ok()) {{ return status; }}"""

    def _cache_field(self, cache: _Cache) -> str:
        return f"""  bool cache_{cache.task_id}_valid_ = false;
  std::uint64_t cache_{cache.task_id}_episode_ = 0;
  std::array<std::uint64_t, {max(len(self.module.inputs), 1)}>
      cache_{cache.task_id}_revisions_{{}};
  std::array<std::uint64_t, {max(len(self.module.states), 1)}>
      cache_{cache.task_id}_state_versions_{{}};"""

    def _state_init_cases(self, *, tensor: bool) -> str:
        cases = []
        for state_id, state in enumerate(self.module.states):
            if tensor != isinstance(state.payload, TensorType):
                continue
            if tensor:
                payload = state.payload
                assert isinstance(payload, TensorType)
                checks = (
                    f"value.tensor.dtype != {_dtype(payload.dtype)} || "
                    f"value.tensor.size_bytes != {_bytes(payload)}u"
                )
                data = "value.tensor.data"
                size = "value.tensor.size_bytes"
                source_device = "value.tensor.device"
            else:
                payload = state.payload
                assert isinstance(payload, ScalarType)
                checks = f"value.dtype != {_dtype(payload.name)}"
                data = "ScalarData(const_cast<VLAForgeScalarValue*>(&value))"
                size = f"{_bytes(payload)}u"
                source_device = "{VLAFORGE_DEVICE_CPU, 0}"
            cases.append(
                f"""    case {state_id}u:
      if ({checks}) {{
        return vlaforge::runtime::Status::Error(
            vlaforge::runtime::StatusCode::kInvalidArgument, state_id,
            "state initializer contract mismatch");
      }}
      return state_store_.Initialize(
          state_id, {data}, {size}, {source_device});"""
            )
        return "\n".join(cases)

    def _initial_state_source(self) -> str:
        lines = []
        for state_id, state in enumerate(self.module.states):
            if state.name not in self.initial_state:
                continue
            value = self.initial_state[state.name]
            if isinstance(state.payload, TensorType):
                ctype = _cpp_scalar(state.payload.dtype)
                values = ", ".join(
                    _literal(item) for item in _flatten(value)
                )
                lines.extend(
                    (
                        "  if (initialization_status_.ok()) {",
                        f"    const {ctype} initial_state_{state_id}[] = "
                        f"{{{values}}};",
                        "    initialization_status_ = state_store_.Initialize(",
                        f"        {state_id}u, initial_state_{state_id}, "
                        f"sizeof(initial_state_{state_id}));",
                        "  }",
                    )
                )
            elif isinstance(state.payload, ScalarType):
                ctype = _cpp_scalar(state.payload.name)
                lines.extend(
                    (
                        "  if (initialization_status_.ok()) {",
                        f"    const {ctype} initial_state_{state_id} = "
                        f"{_literal(value)};",
                        "    initialization_status_ = state_store_.Initialize(",
                        f"        {state_id}u, &initial_state_{state_id}, "
                        f"sizeof(initial_state_{state_id}));",
                        "  }",
                    )
                )
        return "\n".join(lines)

    def _c_abi_source(self) -> str:
        ns = self.namespace
        session_struct = (
            f"""struct VLAForgeSession {{
  explicit VLAForgeSession(const char* bundle_root)
      : implementation(bundle_root) {{}}
  {ns}::ModelSession implementation;
}};"""
            if self.artifact_mode
            else f"""struct VLAForgeSession {{
  {ns}::ModelSession implementation;
}};"""
        )
        allocation = (
            '*session = new (std::nothrow) VLAForgeSession(".");'
            if self.artifact_mode
            else "*session = new (std::nothrow) VLAForgeSession;"
        )
        bundle_factory = (
            """
extern "C" VLAForgeStatus vlaforge_model_session_create_from_bundle(
    const char* bundle_root, size_t bundle_root_size,
    VLAForgeSession** session) {
  if (bundle_root == nullptr || bundle_root_size == 0u ||
      session == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "invalid bundle Session create arguments");
  }
  try {
    const std::string root(bundle_root, bundle_root_size);
    *session = new (std::nothrow) VLAForgeSession(root.c_str());
  } catch (const std::bad_alloc&) {
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                 "bundle Session allocation failed");
  } catch (...) {
    return vlaforge_status_error(VLAFORGE_STATUS_INTERNAL,
                                 "bundle Session construction failed");
  }
  if (*session == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                 "session allocation failed");
  }
  const auto status = (*session)->implementation.initialization_status();
  if (!status.ok()) {
    const auto result = ToCStatus(status);
    delete *session;
    *session = nullptr;
    return result;
  }
  return vlaforge_status_ok();
}
"""
            if self.artifact_mode
            else ""
        )
        return f"""
{session_struct}

namespace {{

VLAForgeStatus CBindTensor(VLAForgeSession* session, std::uint32_t input_id,
                           const VLAForgeBoundTensor* tensor,
                           const VLAForgeInputStamp* stamp) {{
  if (session == nullptr || tensor == nullptr) {{
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "null tensor binding");
  }}
  return ToCStatus(
      session->implementation.BindTensor(input_id, *tensor, stamp));
}}

VLAForgeStatus CBindScalar(VLAForgeSession* session, std::uint32_t input_id,
                           const VLAForgeScalarValue* scalar,
                           const VLAForgeInputStamp* stamp) {{
  if (session == nullptr || scalar == nullptr) {{
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "null scalar binding");
  }}
  return ToCStatus(
      session->implementation.BindScalar(input_id, *scalar, stamp));
}}

VLAForgeStatus CRun(VLAForgeSession* session) {{
  if (session == nullptr) {{
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "null session");
  }}
  return ToCStatus(session->implementation.Run());
}}

VLAForgeStatus CReadTensor(const VLAForgeSession* session,
                           std::uint32_t output_id,
                           VLAForgeBoundTensor* output) {{
  if (session == nullptr) {{
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "null session");
  }}
  return ToCStatus(
      session->implementation.ReadOutputTensor(output_id, output));
}}

VLAForgeStatus CReadScalar(const VLAForgeSession* session,
                           std::uint32_t output_id,
                           VLAForgeScalarValue* output) {{
  if (session == nullptr) {{
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "null session");
  }}
  return ToCStatus(
      session->implementation.ReadOutputScalar(output_id, output));
}}

VLAForgeStatus CReset(VLAForgeSession* session, std::uint64_t episode) {{
  if (session == nullptr) {{
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "null session");
  }}
  return ToCStatus(session->implementation.ResetEpisode(episode));
}}

void CDestroy(VLAForgeSession* session) {{ delete session; }}

const VLAForgeSessionApi kApi = {{
    sizeof(VLAForgeSessionApi),
    VLAFORGE_SESSION_ABI_VERSION,
    {ns}::kSchemaDigest,
    VLAFORGE_SCHEMA_DIGEST_HEX_SIZE,
    &CBindTensor,
    &CBindScalar,
    &CRun,
    &CReadTensor,
    &CReadScalar,
    &CReset,
    &CDestroy,
}};

}}  // namespace

extern "C" VLAForgeStatus vlaforge_model_session_create(
    VLAForgeSession** session) {{
  if (session == nullptr) {{
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT,
                                 "session output is null");
  }}
  try {{
    {allocation}
  }} catch (const std::bad_alloc&) {{
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                 "session allocation failed");
  }} catch (...) {{
    return vlaforge_status_error(VLAFORGE_STATUS_INTERNAL,
                                 "session construction failed");
  }}
  if (*session == nullptr) {{
    return vlaforge_status_error(VLAFORGE_STATUS_OUT_OF_MEMORY,
                                 "session allocation failed");
  }}
  const auto status = (*session)->implementation.initialization_status();
  if (!status.ok()) {{
    const auto result = ToCStatus(status);
    delete *session;
    *session = nullptr;
    return result;
  }}
  return vlaforge_status_ok();
}}

{bundle_factory}
extern "C" const VLAForgeSessionApi* vlaforge_model_session_api(void) {{
  return &kApi;
}}
"""

def _bytes(type_: IRType) -> int:
    return storage_size_bytes(type_)


def _dtype(name: str) -> str:
    mapping = {
        "bool": "VLAFORGE_DTYPE_BOOL",
        "i32": "VLAFORGE_DTYPE_I32",
        "index": "VLAFORGE_DTYPE_I64",
        "i64": "VLAFORGE_DTYPE_I64",
        "u64": "VLAFORGE_DTYPE_U64",
        "f16": "VLAFORGE_DTYPE_F16",
        "bf16": "VLAFORGE_DTYPE_BF16",
        "f32": "VLAFORGE_DTYPE_F32",
        "f64": "VLAFORGE_DTYPE_F64",
        "u8": "VLAFORGE_DTYPE_U8",
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise CodegenUnsupportedError(f"unsupported C ABI dtype {name}") from error


def _device(name: str) -> str:
    if name == "cpu":
        return "VLAFORGE_DEVICE_CPU"
    if name.startswith("cuda"):
        return "VLAFORGE_DEVICE_CUDA"
    return "VLAFORGE_DEVICE_EXTERNAL"


def _device_ordinal(name: str) -> int:
    if name == "cpu":
        return 0
    if name.startswith("cuda:") and name[5:].isdigit():
        return int(name[5:])
    raise CodegenUnsupportedError(
        f"generated Session requires an explicit device ordinal: {name!r}"
    )


def _layout(name: str) -> str:
    return {
        "contiguous": "VLAFORGE_LAYOUT_CONTIGUOUS",
        "nchw": "VLAFORGE_LAYOUT_NCHW",
        "nhwc": "VLAFORGE_LAYOUT_NHWC",
    }.get(name.lower(), "VLAFORGE_LAYOUT_CUSTOM")


def _cpp_scalar(name: str) -> str:
    return {
        "bool": "std::uint8_t",
        "i32": "std::int32_t",
        "index": "std::int64_t",
        "i64": "std::int64_t",
        "u64": "std::uint64_t",
        "f32": "float",
        "f64": "double",
    }.get(name, "std::byte")


def _scalar_member(name: str) -> str:
    return {
        "bool": "boolean",
        "i32": "i32",
        "index": "i64",
        "i64": "i64",
        "u64": "u64",
        "f32": "f32",
        "f64": "f64",
    }[name]


def _literal(value: object) -> str:
    if isinstance(value, bool):
        return "1u" if value else "0u"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CodegenUnsupportedError("non-finite defaults are unsupported")
        return repr(value)
    raise CodegenUnsupportedError(f"unsupported C++ default literal {value!r}")


def _flatten(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple | list):
        return tuple(item for nested in value for item in _flatten(nested))
    return (value,)


def _identifier(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not result or result[0].isdigit():
        result = f"value_{result}"
    return result


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in _identifier(name).split("_"))


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(
        prefix + line if line else "" for line in text.splitlines()
    )


def _indent_lines(lines: list[str], spaces: int) -> list[str]:
    prefix = " " * spaces
    return [prefix + line if line else "" for line in lines]


def _cmake_source(has_runner: bool, *, artifact_backend: bool = False) -> str:
    runner = """
add_executable(vlaforge_generated_runner runner.cpp)
target_link_libraries(vlaforge_generated_runner PRIVATE
    vlaforge_generated_session)
""" if has_runner else ""
    backend_option = (
        'set(VLAFORGE_BUILD_AOTI_BACKEND ON CACHE BOOL "" FORCE)'
        if artifact_backend
        else ""
    )
    backend_find = (
        "find_package(Torch REQUIRED CONFIG)" if artifact_backend else ""
    )
    backend_library = " vlaforge_aoti_backend" if artifact_backend else ""
    return f"""cmake_minimum_required(VERSION 3.18)
project(vlaforge_generated_session LANGUAGES C CXX)

if(NOT DEFINED VLAFORGE_RUNTIME_ROOT)
  message(FATAL_ERROR "set VLAFORGE_RUNTIME_ROOT to the VLAForge source root")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

{backend_option}
{backend_find}
add_subdirectory("${{VLAFORGE_RUNTIME_ROOT}}"
                 "${{CMAKE_CURRENT_BINARY_DIR}}/vlaforge_runtime")
add_library(vlaforge_generated_session STATIC session_generated.cpp)
target_include_directories(vlaforge_generated_session PUBLIC
    "${{CMAKE_CURRENT_SOURCE_DIR}}")
target_link_libraries(vlaforge_generated_session PUBLIC
    vlaforge_runtime{backend_library})
target_compile_options(vlaforge_generated_session PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
{runner}"""


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
