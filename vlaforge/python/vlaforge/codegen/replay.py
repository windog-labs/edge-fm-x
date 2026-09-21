"""Compiler-owned callbacks and lifetime policy for eligible bounded replay."""

from __future__ import annotations

import json
from collections.abc import Mapping

from vlaforge.codegen.model import CppArtifactRegionDefinition
from vlaforge.ir.program import Module
from vlaforge.plan.model import PlanModule, Task
from vlaforge.plan.replay import ReplayArtifactFacts, analyze_replay

_GRAPH_PROVIDERS = {
    "aoti": "vlaforge_libtorch_graph_backend_api",
    "torchscript": "vlaforge_libtorch_graph_backend_api",
}


class ReplayCodegenError(ValueError):
    pass


class ReplayCodegen:
    def __init__(
        self,
        plan: PlanModule,
        module: Module,
        artifacts: Mapping[str, CppArtifactRegionDefinition],
    ):
        self.plan, self.module = plan, module
        self.artifacts = artifacts
        self.region_ids = {
            region.name: index for index, region in enumerate(module.regions)
        }
        self.context_devices = tuple(
            sorted({artifact.device for artifact in artifacts.values()})
        )
        facts = {
            name: ReplayArtifactFacts(
                item.device,
                item.residency,
                item.supports_external_cuda_graph,
                item.effect_audit,
                _GRAPH_PROVIDERS.get(item.backend),
                item.supports_execution_context,
            )
            for name, item in artifacts.items()
        }
        self.decisions = {
            task.id: analyze_replay(plan, task, module, facts)
            for task in plan.tasks
            if task.opcode == "vla.for"
            and task.attributes.get("replay", "off") != "off"
        }
        self.candidates = {
            key: value for key, value in self.decisions.items() if value.candidate
        }
        for value in self.decisions.values():
            if value.requested in ("batch-only", "required") and not value.candidate:
                raise ReplayCodegenError(
                    f"loop {value.task_id} requires replay: {', '.join(value.reasons)}"
                )

    def report(self) -> str:
        return (
            json.dumps(
                {
                    "schema": "vlaforge.replay_analysis/1",
                    "status": "compile_time_eligibility_only_not_runtime_capture_evidence",
                    "loops": [
                        {
                            "task_id": item.task_id,
                            "requested": item.requested,
                            "candidate": item.candidate,
                            "reasons": list(item.reasons),
                            "device": item.device,
                            "staging_bytes": item.staging_bytes,
                        }
                        for item in self.decisions.values()
                    ],
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )

    def header_include(self) -> str:
        return (
            '\n#include "vlaforge/runtime/bounded_replay.h"' if self.decisions else ""
        )

    def public_declaration(self) -> str:
        if not self.decisions:
            return ""
        return """
  VLAForgeStatus GetReplayInfo(
      std::uint32_t task_id, VLAForgeBoundedReplayInfo* info) const noexcept;"""

    def c_declaration(self) -> str:
        if not self.decisions:
            return ""
        return """
extern "C" VLAForgeStatus vlaforge_model_session_get_replay_info(
    const VLAForgeSession* session, std::uint32_t task_id,
    VLAForgeBoundedReplayInfo* info);
"""

    def c_definition(self) -> str:
        if not self.decisions:
            return ""
        return """
extern "C" VLAForgeStatus vlaforge_model_session_get_replay_info(
    const VLAForgeSession* session, std::uint32_t task_id,
    VLAForgeBoundedReplayInfo* info) {
  if (session == nullptr) {
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "null Session");
  }
  return session->implementation.GetReplayInfo(task_id, info);
}
"""

    def private_declarations(self) -> str:
        if not self.candidates:
            return ""
        return "\n" + "\n".join(
            [
                "  void DestroyReplays() noexcept;",
                *(
                    f"  static VLAForgeStatus PrepareReplay{index}(void* user_data) noexcept;\n"
                    f"  static VLAForgeStatus StepReplay{index}(void* user_data, std::uint32_t step,\n"
                    "      VLAForgeReplayPhase phase) noexcept;\n"
                    f"  vlaforge::runtime::Status RunReplay{index}() noexcept;"
                    for index in self.candidates
                ),
            ]
        )

    def fields(self) -> str:
        if not self.candidates:
            return ""
        return "\n" + "\n".join(
            [
                "  bool replay_poisoned_ = false;",
                *(
                    f"  VLAForgeBoundedReplay* replay_{index}_ = nullptr;"
                    for index in self.candidates
                ),
            ]
        )

    def guard(self) -> str:
        if not self.candidates:
            return ""
        return """  if (replay_poisoned_) {
    return vlaforge::runtime::Status::Error(
        vlaforge::runtime::StatusCode::kFailedPrecondition, 0u,
        "replay context poisoned; exit the worker process");
  }
"""

    def reset_prefix(self) -> str:
        if not self.candidates:
            return ""
        return self.guard() + "  DestroyReplays();\n" + self.guard()

    def destruction_prefix(self, *, abandon_extra: str = "") -> str:
        if not self.candidates:
            return ""
        return """  DestroyReplays();
  if (replay_poisoned_) {
""" + abandon_extra + """
    // Native graph/weights/storage may still be referenced by a failed capture.
    // Keep them until process exit instead of reporting unsafe cleanup as done.
    arena_.Abandon();
    state_arena_.Abandon();
    for (auto& arena : output_slots_a_) { if (arena) { arena->Abandon(); } }
    for (auto& arena : output_slots_b_) { if (arena) { arena->Abandon(); } }
    region_executables_.fill(nullptr);
    region_plugins_.fill(nullptr);
    for (auto*& context : execution_contexts_) {
      vlaforge_execution_context_poison(context);
      vlaforge_execution_context_destroy(context);
      context = nullptr;
    }
    return;
  }
"""

    def emit_for(self, task: Task) -> list[str] | None:
        if task.id not in self.candidates:
            return None
        count = self._steps(task)
        lines = [
            f"status = RunReplay{task.id}();",
            "if (!status.ok()) { return Fail(status); }",
            "if (trace_.emit != nullptr) {",
            f"  for (std::uint32_t index = 0; index < {count}u; ++index) {{",
        ]
        for task_id in self.candidates[task.id].region_tasks:
            region = self.plan.task(task_id)
            slot = self.region_ids[str(region.attributes["region"])]
            lines.append(
                "    vlaforge::runtime::EmitTrace(trace_, vlaforge::runtime::TraceEvent{"
                f"vlaforge::runtime::TraceKind::kRegion, {task_id}u, {slot}u, "
                "0u, transaction_.id(), state_store_.episode(), run_index_, 0u});"
            )
        return [*lines, "  }", "}"]

    def methods(self) -> str:
        if not self.decisions:
            return ""
        cases = []
        for index, decision in self.decisions.items():
            if decision.candidate:
                cases.append(f"""    case {index}u:
      if (replay_{index}_ != nullptr) {{
        return vlaforge_bounded_replay_get_info(replay_{index}_, info);
      }}
      *info = {{sizeof(*info), VLAFORGE_REPLAY_UNPREPARED, 0u, 0u, 0u, ""}};
      return vlaforge_status_ok();""")
            else:
                reason = json.dumps("; ".join(decision.reasons))
                cases.append(f"""    case {index}u:
      *info = {{sizeof(*info), VLAFORGE_REPLAY_FALLBACK, 0u, 0u, 0u, {reason}}};
      return vlaforge_status_ok();""")
        result = f"""
VLAForgeStatus ModelSession::GetReplayInfo(
    std::uint32_t task_id, VLAForgeBoundedReplayInfo* info) const noexcept {{
  if (info == nullptr || info->struct_size < sizeof(*info)) {{
    return vlaforge_status_error(VLAFORGE_STATUS_INVALID_ARGUMENT, "invalid replay info");
  }}
  switch (task_id) {{
{chr(10).join(cases)}
    default: return vlaforge_status_error(VLAFORGE_STATUS_NOT_FOUND, "unknown replay loop");
  }}
}}
"""
        if not self.candidates:
            return result
        destroy = "\n".join(
            f"  vlaforge_bounded_replay_destroy(replay_{index}_);\n  replay_{index}_ = nullptr;"
            for index in self.candidates
        )
        result += f"""
void ModelSession::DestroyReplays() noexcept {{
{destroy}
  for (auto* context : execution_contexts_) {{
    if (context != nullptr &&
        vlaforge_execution_context_status(context).code != VLAFORGE_STATUS_OK) {{
      replay_poisoned_ = true;
    }}
  }}
}}
"""
        for index in self.candidates:
            result += self._loop_methods(self.plan.task(index))
        return result

    @staticmethod
    def _steps(task: Task) -> int:
        return len(
            range(
                task.attributes["lower"],
                task.attributes["upper"],
                task.attributes["step"],
            )
        )

    def _loop_methods(self, task: Task) -> str:
        index = task.id
        decision = self.candidates[index]
        context_slot = self.context_devices.index(decision.device)
        body = self.plan.block(task.blocks[0])
        terminal = self.plan.task(body.tasks[-1])
        mapping = dict(zip(body.arguments[1:], task.outputs, strict=True))
        mapping.update(
            zip(
                task.attributes["replay_liveins"],
                task.attributes["replay_staging"],
                strict=True,
            )
        )
        body_lines = []
        bind_lines = []
        registrations = []
        provider = None
        for task_id in decision.region_tasks:
            region = self.plan.task(task_id)
            name = str(region.attributes["region"])
            slot = self.region_ids[name]
            if decision.requested != "batch-only":
                provider = _GRAPH_PROVIDERS[self.artifacts[name].backend]
            body_lines.extend(
                [
                    f"  status = self.region_apis_[{slot}u]->run(self.region_executables_[{slot}u]);",
                    "  if (status.code != VLAFORGE_STATUS_OK) { return status; }",
                ]
            )
            registrations.append(
                f"      {{region_executables_[{slot}u], region_apis_[{slot}u], region_execution_extensions_[{slot}u]}}"
            )
            for direction, ids in (
                ("input", region.inputs),
                ("output", region.outputs),
            ):
                for position, source in enumerate(ids):
                    target = mapping.get(source, source)
                    bind_lines.extend(
                        [
                            "    {",
                            "      VLAForgeValueView view{};",
                            "      view.struct_size = sizeof(view);",
                            "      view.kind = VLAFORGE_VALUE_TENSOR;",
                            (
                                "      view.value.tensor = {sizeof(VLAForgeBoundTensor), "
                                f"values_[{target}u], VLAFORGE_LAYOUT_CONTIGUOUS, 1u}};"
                            ),
                            f"      c_status = region_apis_[{slot}u]->bind_{direction}(region_executables_[{slot}u], {position}u, &view);",
                            f"      if (c_status.code != VLAFORGE_STATUS_OK) {{ return FromCStatus(c_status, {index}u); }}",
                            "    }",
                        ]
                    )

        def copies(pairs, *, self_prefix="self."):
            result = []
            for source, destination in pairs:
                result.extend(
                    [
                        f"  status = vlaforge_execution_context_copy({self_prefix}execution_contexts_[{context_slot}u],",
                        f"      &{self_prefix}values_[{destination}u], &{self_prefix}values_[{source}u], {self_prefix}values_[{source}u].size_bytes);",
                        "  if (status.code != VLAFORGE_STATUS_OK) { return status; }",
                    ]
                )
            return "\n".join(result)

        yielded = tuple(mapping.get(value, value) for value in terminal.inputs)
        scratch = tuple(task.attributes.get("carry_scratch", ()))
        carry_pairs = list(zip(yielded, scratch or task.outputs, strict=True))
        if scratch:
            carry_pairs.extend(zip(scratch, task.outputs, strict=True))
        snapshot_pairs = list(
            zip(task.inputs, task.attributes["replay_seeds"], strict=True)
        )
        snapshot_pairs.extend(
            zip(
                task.attributes["replay_liveins"],
                task.attributes["replay_staging"],
                strict=True,
            )
        )
        snapshot_lines = []
        for source, destination in snapshot_pairs:
            snapshot_lines.extend(
                [
                    "  if (c_status.code == VLAFORGE_STATUS_OK) {",
                    f"    c_status = vlaforge_execution_context_copy(context, &values_[{destination}u],",
                    f"        &values_[{source}u], values_[{source}u].size_bytes);",
                    "  }",
                ]
            )
        mode = {
            "batch-only": "VLAFORGE_REPLAY_ORDINARY",
            "required": "VLAFORGE_REPLAY_REQUIRE",
            "prefer": "VLAFORGE_REPLAY_PREFER",
        }[decision.requested]
        provider_expression = "nullptr" if provider is None else f"{provider}()"
        warmups = 0 if decision.requested == "batch-only" else 2
        return f"""
VLAForgeStatus ModelSession::PrepareReplay{index}(void* user_data) noexcept {{
  auto& self = *static_cast<ModelSession*>(user_data);
  auto status = vlaforge_status_ok();
{copies(zip(task.attributes["replay_seeds"], task.outputs, strict=True))}
  return status;
}}

VLAForgeStatus ModelSession::StepReplay{index}(
    void* user_data, std::uint32_t step, VLAForgeReplayPhase phase) noexcept {{
  (void)step;
  (void)phase;
  auto& self = *static_cast<ModelSession*>(user_data);
  auto status = vlaforge_status_ok();
{chr(10).join(body_lines)}
{copies(carry_pairs)}
  return status;
}}

vlaforge::runtime::Status ModelSession::RunReplay{index}() noexcept {{
{self.guard()}  auto* context = execution_contexts_[{context_slot}u];
  auto c_status = vlaforge_execution_context_status(context);
  if (c_status.code != VLAFORGE_STATUS_OK) {{ return FromCStatus(c_status, {index}u); }}
  if (replay_{index}_ == nullptr) {{
{chr(10).join(bind_lines)}
    const VLAForgeReplayRegion regions[] = {{
{",".join(registrations)}
    }};
    const VLAForgeBoundedReplayOptions options{{
        sizeof(options), VLAFORGE_BOUNDED_REPLAY_ABI_VERSION, {mode},
        {self._steps(task)}u, {warmups}u, nullptr, &PrepareReplay{index}, &StepReplay{index}, this,
        regions, {len(registrations)}u}};
    c_status = vlaforge_bounded_replay_create(context, {provider_expression}, &options, &replay_{index}_);
    if (c_status.code != VLAFORGE_STATUS_OK) {{ return FromCStatus(c_status, {index}u); }}
  }}
{chr(10).join(snapshot_lines)}
  const auto drained = vlaforge_execution_context_synchronize(context);
  if (drained.code != VLAFORGE_STATUS_OK) {{
    vlaforge_execution_context_poison(context);
    replay_poisoned_ = true;
    return FromCStatus(drained, {index}u);
  }}
  if (c_status.code != VLAFORGE_STATUS_OK) {{ return FromCStatus(c_status, {index}u); }}
  c_status = vlaforge_bounded_replay_run(replay_{index}_);
  if (vlaforge_execution_context_status(context).code != VLAFORGE_STATUS_OK) {{
    replay_poisoned_ = true;
  }}
  return FromCStatus(c_status, {index}u);
}}
"""
