"""Conservative whole-loop replay analysis and owned reset storage planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from vlaforge.deployment.contract import EffectAudit
from vlaforge.ir.program import Module
from vlaforge.ir.types import TensorType
from vlaforge.plan.model import BufferClass, LogicalBuffer, PlanModule, Task

REPLAY_MODES = ("off", "batch-only", "prefer", "required")


@dataclass(frozen=True, slots=True)
class ReplayArtifactFacts:
    device: str
    residency: str
    supports_external_cuda_graph: bool
    effect_audit: EffectAudit | None
    graph_provider: str | None
    supports_execution_context: bool = False


@dataclass(frozen=True, slots=True)
class ReplayStructure:
    task_id: int
    requested: str
    region_tasks: tuple[int, ...]
    liveins: tuple[int, ...]
    reasons: tuple[str, ...]
    device: str | None = None
    staging_bytes: int = 0

    @property
    def candidate(self) -> bool:
        return self.requested != "off" and not self.reasons


def analyze_replay_structure(
    plan: PlanModule, task: Task, module: Module | None = None
) -> ReplayStructure:
    """Prove scheduling/SSA eligibility, not backend graph compatibility."""

    requested = task.attributes.get("replay", "off")
    reasons: list[str] = []
    if task.opcode != "vla.for" or requested not in REPLAY_MODES:
        return ReplayStructure(
            task.id, str(requested), (), (), ("replay.invalid_request",)
        )
    if requested == "off":
        return ReplayStructure(task.id, requested, (), (), ())
    if len(task.blocks) != 1:
        return ReplayStructure(task.id, requested, (), (), ("replay.invalid_body",))
    body = plan.block(task.blocks[0])
    if not body.tasks or len(body.arguments) != len(task.outputs) + 1:
        return ReplayStructure(task.id, requested, (), (), ("replay.invalid_body",))
    lower = int(task.attributes.get("lower", 0))
    upper = int(task.attributes.get("upper", 0))
    stride = int(task.attributes.get("step", 0))
    if stride <= 0 or upper <= lower or (upper - lower + stride - 1) // stride > 65536:
        reasons.append("replay.invalid_bound")
    terminal = plan.task(body.tasks[-1])
    region_tasks = tuple(plan.task(index) for index in body.tasks[:-1])
    if terminal.opcode != "vla.yield" or not region_tasks:
        reasons.append("replay.invalid_body")
    if any(item.opcode != "vla.invoke" or item.blocks for item in region_tasks):
        reasons.append("replay.host_control_or_effect")
    if any(body.arguments[0] in item.inputs for item in (*region_tasks, terminal)):
        reasons.append("replay.host_induction")
    produced = {index for item in region_tasks for index in item.outputs}
    liveins = tuple(
        sorted(
            {
                index
                for item in (*region_tasks, terminal)
                for index in item.inputs
                if index not in produced and index not in body.arguments
            }
        )
    )
    used = set(task.inputs + task.outputs + body.arguments[1:] + liveins)
    for item in (*region_tasks, terminal):
        used.update(
            index for index in item.inputs + item.outputs if index != body.arguments[0]
        )
    if any(not isinstance(plan.buffer(index).type, TensorType) for index in used):
        reasons.append("replay.non_tensor_value")
    elif any(
        any(dimension is None for dimension in plan.buffer(index).type.shape)
        or plan.buffer(index).type.layout != "contiguous"
        for index in used
    ):
        reasons.append("replay.dynamic_shape_or_layout")
    artifact_ids = [
        item.artifact_id for item in region_tasks if item.opcode == "vla.invoke"
    ]
    if any(
        sum(
            other.artifact_id == index
            for other in plan.tasks
            if other.opcode == "vla.invoke"
        )
        != 1
        for index in artifact_ids
    ):
        reasons.append("replay.shared_region_bindings")
    if module is not None:
        for item in region_tasks:
            if item.opcode != "vla.invoke":
                continue
            region = module.region(str(item.attributes["region"]))
            if not region.pure:
                reasons.append("replay.non_pure_region")
            if region.metadata.get("memoize", False):
                reasons.append("replay.cache_decision_inside_loop")
    return ReplayStructure(
        task.id,
        requested,
        tuple(item.id for item in region_tasks),
        liveins,
        tuple(dict.fromkeys(reasons)),
    )


def prepare_replay_storage(plan: PlanModule, module: Module) -> PlanModule:
    """Reserve stable seed/live-in contents before physical arena planning."""

    buffers = list(plan.buffers)
    tasks = list(plan.tasks)
    for task in plan.tasks:
        if task.opcode != "vla.for" or task.attributes.get("replay", "off") == "off":
            continue
        analysis = analyze_replay_structure(plan, task, module)
        attributes = dict(task.attributes)
        attributes["replay_reasons"] = list(analysis.reasons)
        if analysis.candidate:

            def reserve(
                sources: tuple[int, ...], label: str, owner: int = task.id
            ) -> list[int]:
                result: list[int] = []
                for position, source in enumerate(sources):
                    index = len(buffers)
                    buffers.append(
                        LogicalBuffer(
                            id=index,
                            name=f"replay_{owner}_{label}_{position}",
                            type=plan.buffer(source).type,
                            buffer_class=BufferClass.LOOP_CARRIED,
                            producer_task=owner,
                            source=f"replay:{owner}:{label}",
                        )
                    )
                    result.append(index)
                return result

            attributes["replay_seeds"] = reserve(task.inputs, "seed")
            attributes["replay_liveins"] = list(analysis.liveins)
            attributes["replay_staging"] = reserve(analysis.liveins, "livein")
        tasks[task.id] = replace(task, attributes=attributes)
    return replace(plan, tasks=tuple(tasks), buffers=tuple(buffers))


def analyze_replay_memory(
    plan: PlanModule, task: Task, module: Module
) -> ReplayStructure:
    """Require fixed CUDA arena storage and same-device source snapshots."""

    analysis = analyze_replay_structure(plan, task, module)
    if not analysis.candidate:
        return analysis
    if plan.arena is None:
        return replace(analysis, reasons=("replay.unplanned_memory",))
    device = plan.arena.device
    physical = {
        index: item
        for item in plan.arena.physical_buffers
        for index in item.logical_buffers
    }
    staging = tuple(task.attributes.get("replay_seeds", ())) + tuple(
        task.attributes.get("replay_staging", ())
    )
    reasons = []
    if not device.startswith("cuda:"):
        reasons.append("replay.requires_cuda")
    if len(task.attributes.get("replay_seeds", ())) != len(task.inputs) or any(
        index not in physical for index in staging
    ):
        reasons.append("replay.missing_owned_reset_storage")

    def source_device(index: int) -> str | None:
        buffer = plan.buffer(index)
        producer = (
            None if buffer.producer_task is None else plan.task(buffer.producer_task)
        )
        if producer is not None and producer.opcode == "vla.input.read":
            return next(
                port.device
                for port in module.inputs
                if port.name == producer.attributes["input"]
            )
        if producer is not None and producer.opcode == "vla.snapshot.value":
            snapshot = plan.buffer(producer.inputs[0])
            if snapshot.producer_task is None:
                return None
            state_name = plan.task(snapshot.producer_task).attributes.get("state")
            return next(
                (state.device for state in plan.states if state.name == state_name),
                None,
            )
        return physical[index].device if index in physical else None

    if any(source_device(index) != device for index in task.inputs + analysis.liveins):
        reasons.append("replay.source_device_mismatch")
    return replace(
        analysis,
        device=device,
        reasons=tuple(reasons),
        staging_bytes=sum(
            physical[index].size_bytes for index in staging if index in physical
        ),
    )


def analyze_replay(
    plan: PlanModule,
    task: Task,
    module: Module,
    artifacts: Mapping[str, ReplayArtifactFacts],
) -> ReplayStructure:
    """Join proven memory/schedule facts with independently audited artifacts."""

    analysis = analyze_replay_memory(plan, task, module)
    if not analysis.candidate:
        return analysis
    reasons = []
    providers = set()
    for task_id in analysis.region_tasks:
        name = str(plan.task(task_id).attributes["region"])
        artifact = artifacts.get(name)
        if artifact is None:
            reasons.append("replay.missing_artifact_contract")
            continue
        if artifact.device != analysis.device:
            reasons.append("replay.artifact_device_mismatch")
        if artifact.residency != "session":
            reasons.append("replay.requires_session_residency")
        if analysis.requested == "batch-only":
            if not (artifact.supports_execution_context or artifact.supports_external_cuda_graph):
                reasons.append("replay.backend_execution_context_unavailable")
        elif not artifact.supports_external_cuda_graph or artifact.graph_provider is None:
            reasons.append("replay.backend_graph_provider_unavailable")
        else:
            providers.add(artifact.graph_provider)
        audit = artifact.effect_audit
        if audit is None:
            reasons.append("replay.missing_effect_audit")
        elif not audit.passed or audit.explicit_rng or audit.lifted_states:
            reasons.append("replay.unsupported_artifact_effects")
    if len(providers) > 1:
        reasons.append("replay.incompatible_graph_providers")
    return replace(analysis, reasons=tuple(dict.fromkeys(reasons)))
