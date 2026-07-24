"""Verifier for the internal Scheduled Execution Plan."""

from __future__ import annotations

from dataclasses import dataclass

from vlaforge.plan.model import BufferClass, PlanModule, TaskKind


@dataclass(frozen=True, slots=True)
class PlanDiagnostic:
    rule: str
    message: str
    task_id: int | None = None

    def format(self, plan: PlanModule) -> str:
        task = "" if self.task_id is None else f" task={self.task_id}"
        return (
            f"plan={plan.name} rule={self.rule}{task} "
            f"message={self.message}"
        )


class PlanVerificationError(ValueError):
    def __init__(
        self, plan: PlanModule, diagnostics: tuple[PlanDiagnostic, ...]
    ):
        self.plan = plan
        self.diagnostics = diagnostics
        super().__init__("\n".join(item.format(plan) for item in diagnostics))


def verify_plan(
    plan: PlanModule, *, raise_on_error: bool = True
) -> tuple[PlanDiagnostic, ...]:
    diagnostics: list[PlanDiagnostic] = []
    task_ids = [task.id for task in plan.tasks]
    block_ids = [block.id for block in plan.blocks]
    buffer_ids = [buffer.id for buffer in plan.buffers]
    artifact_ids = [artifact.artifact_id for artifact in plan.artifacts]
    state_ids = [state.state_id for state in plan.states]

    _require_contiguous("task.id", task_ids, diagnostics)
    _require_contiguous("block.id", block_ids, diagnostics)
    _require_contiguous("buffer.id", buffer_ids, diagnostics)
    _require_contiguous("artifact.id", artifact_ids, diagnostics)
    _require_contiguous("state.id", state_ids, diagnostics)

    task_map = {task.id: task for task in plan.tasks}
    block_map = {block.id: block for block in plan.blocks}
    buffer_map = {buffer.id: buffer for buffer in plan.buffers}
    artifact_map = {
        artifact.artifact_id: artifact for artifact in plan.artifacts
    }

    task_membership: dict[int, int] = {}
    for block in plan.blocks:
        if len(block.arguments) != len(set(block.arguments)):
            diagnostics.append(
                PlanDiagnostic(
                    "block.duplicate_argument",
                    f"block {block.id} has duplicate arguments",
                )
            )
        for buffer_id in block.arguments:
            if buffer_id not in buffer_map:
                diagnostics.append(
                    PlanDiagnostic(
                        "block.unknown_argument",
                        f"block {block.id} references buffer {buffer_id}",
                    )
                )
        for task_id in block.tasks:
            if task_id not in task_map:
                diagnostics.append(
                    PlanDiagnostic(
                        "block.unknown_task",
                        f"block {block.id} references task {task_id}",
                    )
                )
                continue
            if task_id in task_membership:
                diagnostics.append(
                    PlanDiagnostic(
                        "block.task_reused",
                        f"task {task_id} appears in blocks "
                        f"{task_membership[task_id]} and {block.id}",
                        task_id,
                    )
                )
            task_membership[task_id] = block.id
    missing_membership = sorted(set(task_map) - set(task_membership))
    if missing_membership:
        diagnostics.append(
            PlanDiagnostic(
                "block.unreachable_tasks",
                f"tasks are not assigned to blocks: {missing_membership}",
            )
        )

    for policy in plan.policies:
        if policy.body_block not in block_map:
            diagnostics.append(
                PlanDiagnostic(
                    "policy.missing_body",
                    f"policy {policy.name} references block {policy.body_block}",
                )
            )
        for buffer_id in policy.inputs:
            buffer = buffer_map.get(buffer_id)
            if buffer is None or not buffer.external:
                diagnostics.append(
                    PlanDiagnostic(
                        "policy.invalid_input",
                        f"policy {policy.name} input {buffer_id} is not external",
                    )
                )

    for task in plan.tasks:
        for dependency in task.dependencies:
            if dependency not in task_map:
                diagnostics.append(
                    PlanDiagnostic(
                        "dependency.missing",
                        f"dependency {dependency} does not exist",
                        task.id,
                    )
                )
        if task.id in task.dependencies:
            diagnostics.append(
                PlanDiagnostic(
                    "dependency.self_cycle",
                    "task depends on itself",
                    task.id,
                )
            )
        for block_id in task.blocks:
            if block_id not in block_map:
                diagnostics.append(
                    PlanDiagnostic(
                        "task.missing_block",
                        f"nested block {block_id} does not exist",
                        task.id,
                    )
                )
        for buffer_id in task.inputs + task.outputs:
            if buffer_id not in buffer_map:
                diagnostics.append(
                    PlanDiagnostic(
                        "buffer.missing",
                        f"buffer {buffer_id} does not exist",
                        task.id,
                    )
                )
        if task.workspace_buffer is not None:
            workspace = buffer_map.get(task.workspace_buffer)
            if workspace is None:
                diagnostics.append(
                    PlanDiagnostic(
                        "workspace.missing_buffer",
                        f"workspace buffer {task.workspace_buffer} does not exist",
                        task.id,
                    )
                )
            elif (
                workspace.buffer_class is not BufferClass.REGION_WORKSPACE
                or workspace.producer_task != task.id
            ):
                diagnostics.append(
                    PlanDiagnostic(
                        "workspace.invalid_buffer",
                        f"buffer {task.workspace_buffer} is not this task's "
                        "region workspace",
                        task.id,
                    )
                )
        for buffer_id in task.outputs:
            buffer = buffer_map.get(buffer_id)
            if buffer is not None and buffer.producer_task != task.id:
                diagnostics.append(
                    PlanDiagnostic(
                        "buffer.producer_mismatch",
                        f"buffer {buffer_id} names producer "
                        f"{buffer.producer_task}",
                        task.id,
                    )
                )

        reachable = _transitive_dependencies(task.id, task_map)
        for buffer_id in task.inputs:
            buffer = buffer_map.get(buffer_id)
            if buffer is None or buffer.producer_task is None:
                continue
            if buffer.producer_task not in reachable:
                diagnostics.append(
                    PlanDiagnostic(
                        "buffer.read_before_produce",
                        f"input buffer {buffer_id} is produced by task "
                        f"{buffer.producer_task} without a dependency path",
                        task.id,
                    )
                )

        if task.kind is TaskKind.REGION:
            artifact = (
                None
                if task.artifact_id is None
                else artifact_map.get(task.artifact_id)
            )
            region_name = str(task.attributes.get("region", ""))
            if artifact is None:
                diagnostics.append(
                    PlanDiagnostic(
                        "artifact.missing",
                        f"region {region_name!r} has no artifact binding",
                        task.id,
                    )
                )
            elif artifact.region_name != region_name:
                diagnostics.append(
                    PlanDiagnostic(
                        "artifact.region_mismatch",
                        f"artifact {artifact.artifact_id} is for "
                        f"{artifact.region_name}, not {region_name}",
                        task.id,
                    )
                )
            elif artifact.workspace_size_bytes > 0 and task.workspace_buffer is None:
                diagnostics.append(
                    PlanDiagnostic(
                        "workspace.binding_missing",
                        f"artifact {artifact.artifact_id} requires "
                        f"{artifact.workspace_size_bytes} bytes",
                        task.id,
                    )
                )

        if task.opcode == "vla.sample_input":
            maximum = task.attributes.get("max_age_ns")
            if maximum is not None and (
                task.freshness_guard is None
                or task.freshness_guard.max_age_ns != int(maximum)
            ):
                diagnostics.append(
                    PlanDiagnostic(
                        "freshness.guard_missing",
                        "bounded input sample lost its freshness guard",
                        task.id,
                    )
                )

        if task.kind is TaskKind.LOOP:
            _verify_loop(task, diagnostics)

        if task.kind is TaskKind.COMMIT:
            if len(task.inputs) < 3:
                diagnostics.append(
                    PlanDiagnostic(
                        "commit.missing_validation",
                        "commit requires transaction, action, and condition",
                        task.id,
                    )
                )
            elif buffer_map.get(task.inputs[2]) is not None:
                producer = buffer_map[task.inputs[2]].producer_task
                if (
                    producer is None
                    or task_map.get(producer) is None
                    or task_map[producer].kind is not TaskKind.VALIDATION
                ):
                    diagnostics.append(
                        PlanDiagnostic(
                            "commit.missing_validation",
                            "commit condition is not produced by validation",
                            task.id,
                        )
                    )

        if task.kind is TaskKind.PUBLISH:
            if len(task.inputs) != 1:
                diagnostics.append(
                    PlanDiagnostic(
                        "publish.invalid_arity",
                        "publish requires one committed action",
                        task.id,
                    )
                )
            else:
                buffer = buffer_map.get(task.inputs[0])
                producer = None if buffer is None else buffer.producer_task
                if (
                    producer is None
                    or task_map.get(producer) is None
                    or task_map[producer].kind is not TaskKind.COMMIT
                ):
                    diagnostics.append(
                        PlanDiagnostic(
                            "publish.before_commit",
                            "published action is not produced by commit",
                            task.id,
                        )
                    )

    diagnostics.extend(_verify_dependency_cycles(task_map))
    diagnostics.extend(_verify_state_layout(plan))
    if plan.arena is not None:
        diagnostics.extend(_verify_arena(plan))
    result = tuple(diagnostics)
    if result and raise_on_error:
        raise PlanVerificationError(plan, result)
    return result


def _require_contiguous(
    rule: str,
    identifiers: list[int],
    diagnostics: list[PlanDiagnostic],
) -> None:
    if identifiers != list(range(len(identifiers))):
        diagnostics.append(
            PlanDiagnostic(
                rule,
                f"expected deterministic contiguous ids, got {identifiers}",
            )
        )


def _transitive_dependencies(
    task_id: int, task_map: dict[int, object]
) -> set[int]:
    visited: set[int] = set()
    stack = list(task_map[task_id].dependencies)
    while stack:
        current = stack.pop()
        if current in visited or current not in task_map:
            continue
        visited.add(current)
        stack.extend(task_map[current].dependencies)
    return visited


def _verify_dependency_cycles(
    task_map: dict[int, object],
) -> tuple[PlanDiagnostic, ...]:
    diagnostics = []
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(task_id: int) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            diagnostics.append(
                PlanDiagnostic(
                    "dependency.cycle",
                    "dependency graph contains a cycle",
                    task_id,
                )
            )
            return
        visiting.add(task_id)
        for dependency in task_map[task_id].dependencies:
            if dependency in task_map:
                visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in task_map:
        visit(task_id)
    return tuple(diagnostics)


def _verify_loop(
    task: object, diagnostics: list[PlanDiagnostic]
) -> None:
    if task.opcode == "vla.for":
        lower = int(task.attributes.get("lower", 0))
        upper = int(task.attributes.get("upper", 0))
        step = int(task.attributes.get("step", 0))
        if step <= 0 or lower >= upper:
            diagnostics.append(
                PlanDiagnostic(
                    "loop.invalid_bound",
                    f"for bounds [{lower}, {upper}) step={step} are invalid",
                    task.id,
                )
            )
        if len(task.blocks) != 1:
            diagnostics.append(
                PlanDiagnostic(
                    "loop.invalid_body",
                    "for loop requires one body block",
                    task.id,
                )
            )
    elif task.opcode == "vla.while":
        maximum = int(task.attributes.get("max_iterations", 0))
        if maximum <= 0:
            diagnostics.append(
                PlanDiagnostic(
                    "loop.invalid_bound",
                    f"while max_iterations={maximum} is invalid",
                    task.id,
                )
            )
        if len(task.blocks) != 2:
            diagnostics.append(
                PlanDiagnostic(
                    "loop.invalid_body",
                    "while loop requires condition and body blocks",
                    task.id,
                )
            )


def _verify_arena(plan: PlanModule) -> tuple[PlanDiagnostic, ...]:
    assert plan.arena is not None
    diagnostics = []
    _require_contiguous(
        "arena.physical_buffer_id",
        [item.id for item in plan.arena.physical_buffers],
        diagnostics,
    )
    logical_ids = {buffer.id for buffer in plan.buffers}
    eligible = {
        buffer.id
        for buffer in plan.buffers
        if not buffer.external and buffer.buffer_class is not BufferClass.EXTERNAL
    }
    mapped = [
        logical_id
        for physical in plan.arena.physical_buffers
        for logical_id in physical.logical_buffers
    ]
    missing = sorted(eligible - set(mapped))
    duplicate = sorted(
        logical_id for logical_id in set(mapped) if mapped.count(logical_id) > 1
    )
    unexpected = sorted(set(mapped) - eligible)
    if missing:
        diagnostics.append(
            PlanDiagnostic(
                "arena.unplanned_buffers",
                f"internal logical buffers have no allocation: {missing}",
            )
        )
    if duplicate:
        diagnostics.append(
            PlanDiagnostic(
                "arena.duplicate_mapping",
                f"logical buffers have multiple allocations: {duplicate}",
            )
        )
    if unexpected:
        diagnostics.append(
            PlanDiagnostic(
                "arena.external_mapping",
                f"external logical buffers were allocated: {unexpected}",
            )
        )
    for physical in plan.arena.physical_buffers:
        unknown = sorted(set(physical.logical_buffers) - logical_ids)
        if unknown:
            diagnostics.append(
                PlanDiagnostic(
                    "arena.unknown_logical_buffer",
                    f"physical buffer {physical.id} references {unknown}",
                )
            )
        if physical.offset + physical.size_bytes > plan.arena.size_bytes:
            diagnostics.append(
                PlanDiagnostic(
                    "arena.out_of_bounds",
                    f"physical buffer {physical.id} exceeds arena size",
                )
            )
        if physical.device != plan.arena.device:
            diagnostics.append(
                PlanDiagnostic(
                    "arena.device_mismatch",
                    f"physical buffer {physical.id} device {physical.device} "
                    f"does not match arena {plan.arena.device}",
                )
            )
        for logical_id in physical.logical_buffers:
            logical = next(
                (
                    buffer
                    for buffer in plan.buffers
                    if buffer.id == logical_id
                ),
                None,
            )
            if logical is None or logical.producer_task is None:
                continue
            consumers = [
                task.id
                for task in plan.tasks
                if logical_id in task.inputs
            ]
            expected_last = max(consumers, default=logical.producer_task)
            if (
                physical.first_task > logical.producer_task
                or physical.last_task < expected_last
            ):
                diagnostics.append(
                    PlanDiagnostic(
                        "arena.lifetime_too_short",
                        f"physical buffer {physical.id} lifetime "
                        f"[{physical.first_task}, {physical.last_task}] "
                        f"does not cover logical buffer {logical_id} "
                        f"[{logical.producer_task}, {expected_last}]",
                    )
                )
    physical = plan.arena.physical_buffers
    for index, left in enumerate(physical):
        for right in physical[index + 1 :]:
            memory_overlap = (
                left.offset < right.offset + right.size_bytes
                and right.offset < left.offset + left.size_bytes
            )
            lifetime_overlap = (
                left.first_task <= right.last_task
                and right.first_task <= left.last_task
            )
            if memory_overlap and lifetime_overlap:
                diagnostics.append(
                    PlanDiagnostic(
                        "arena.overlapping_live_buffers",
                        f"physical buffers {left.id} and {right.id} overlap "
                        "in memory and lifetime",
                    )
                )
    return tuple(diagnostics)


def _verify_state_layout(plan: PlanModule) -> tuple[PlanDiagnostic, ...]:
    diagnostics = []
    for state in plan.states:
        if (
            state.slot_capacity is not None
            and state.slot_capacity < state.required_capacity
        ):
            diagnostics.append(
                PlanDiagnostic(
                    "state.unsafe_slot_reuse",
                    f"state {state.name} capacity={state.slot_capacity} "
                    f"required={state.required_capacity}",
                )
            )
        if plan.arena is not None and state.total_size_bytes is None:
            diagnostics.append(
                PlanDiagnostic(
                    "state.unphysicalized",
                    f"state {state.name} has no physical ring layout",
                )
            )
    physical = [
        state
        for state in plan.states
        if state.offset is not None and state.total_size_bytes is not None
    ]
    for index, left in enumerate(physical):
        assert left.offset is not None
        assert left.total_size_bytes is not None
        for right in physical[index + 1 :]:
            if left.device != right.device:
                continue
            assert right.offset is not None
            assert right.total_size_bytes is not None
            overlap = (
                left.offset < right.offset + right.total_size_bytes
                and right.offset < left.offset + left.total_size_bytes
            )
            if overlap:
                diagnostics.append(
                    PlanDiagnostic(
                        "state.overlapping_rings",
                        f"state rings {left.name} and {right.name} overlap",
                    )
                )
    return tuple(diagnostics)
