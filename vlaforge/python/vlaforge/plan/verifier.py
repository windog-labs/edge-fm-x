"""Structural verifier for passive Scheduled Plan v2."""

from __future__ import annotations

from dataclasses import dataclass

from vlaforge.plan.model import BufferClass, PlanModule, Task, TaskKind


@dataclass(frozen=True, slots=True)
class PlanDiagnostic:
    rule: str
    message: str
    task_id: int | None = None

    def format(self, plan: PlanModule) -> str:
        task = "" if self.task_id is None else f" task={self.task_id}"
        return f"plan={plan.name} rule={self.rule}{task} message={self.message}"


class PlanVerificationError(ValueError):
    def __init__(
        self,
        plan: PlanModule,
        diagnostics: tuple[PlanDiagnostic, ...],
    ):
        self.plan = plan
        self.diagnostics = diagnostics
        super().__init__("\n".join(item.format(plan) for item in diagnostics))


def verify_plan(
    plan: PlanModule,
    *,
    raise_on_error: bool = True,
) -> tuple[PlanDiagnostic, ...]:
    diagnostics: list[PlanDiagnostic] = []
    _require_contiguous(
        "task.id",
        [item.id for item in plan.tasks],
        diagnostics,
    )
    _require_contiguous(
        "block.id",
        [item.id for item in plan.blocks],
        diagnostics,
    )
    _require_contiguous(
        "buffer.id",
        [item.id for item in plan.buffers],
        diagnostics,
    )
    _require_contiguous(
        "artifact.id",
        [item.artifact_id for item in plan.artifacts],
        diagnostics,
    )
    _require_contiguous(
        "state.id",
        [item.state_id for item in plan.states],
        diagnostics,
    )
    _require_contiguous(
        "invocation.id",
        [item.id for item in plan.invocations],
        diagnostics,
    )

    task_map = {task.id: task for task in plan.tasks}
    block_map = {block.id: block for block in plan.blocks}
    buffer_map = {buffer.id: buffer for buffer in plan.buffers}
    artifact_map = {
        artifact.artifact_id: artifact for artifact in plan.artifacts
    }

    invocation_names: set[str] = set()
    for invocation in plan.invocations:
        if invocation.name in invocation_names:
            diagnostics.append(
                PlanDiagnostic(
                    "invocation.duplicate",
                    f"duplicate invocation {invocation.name}",
                )
            )
        invocation_names.add(invocation.name)
        if invocation.body_block not in block_map:
            diagnostics.append(
                PlanDiagnostic(
                    "invocation.missing_body",
                    f"invocation {invocation.name} references "
                    f"block {invocation.body_block}",
                )
            )

    membership: dict[int, int] = {}
    for block in plan.blocks:
        if len(block.arguments) != len(set(block.arguments)):
            diagnostics.append(
                PlanDiagnostic(
                    "block.duplicate_argument",
                    f"block {block.id} has duplicate arguments",
                )
            )
        for buffer_id in block.arguments:
            buffer = buffer_map.get(buffer_id)
            if buffer is None:
                diagnostics.append(
                    PlanDiagnostic(
                        "block.unknown_argument",
                        f"block {block.id} references buffer {buffer_id}",
                    )
                )
            elif not buffer.external:
                diagnostics.append(
                    PlanDiagnostic(
                        "block.internal_argument",
                        f"block argument buffer {buffer_id} must be external "
                        "to that block",
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
            if task_id in membership:
                diagnostics.append(
                    PlanDiagnostic(
                        "block.task_reused",
                        f"task {task_id} appears in blocks "
                        f"{membership[task_id]} and {block.id}",
                        task_id,
                    )
                )
            membership[task_id] = block.id
    missing = sorted(set(task_map) - set(membership))
    if missing:
        diagnostics.append(
            PlanDiagnostic(
                "block.unreachable_tasks",
                f"tasks are not assigned to blocks: {missing}",
            )
        )

    for task in plan.tasks:
        _verify_task(
            task,
            task_map,
            block_map,
            buffer_map,
            artifact_map,
            diagnostics,
        )
        diagnostics.extend(_verify_replay_storage(plan, task))

    diagnostics.extend(_verify_dependency_cycles(task_map))
    diagnostics.extend(_verify_state_layout(plan))
    if plan.arena is not None:
        diagnostics.extend(_verify_arena(plan))
    result = tuple(diagnostics)
    if result and raise_on_error:
        raise PlanVerificationError(plan, result)
    return result


def _verify_replay_storage(plan: PlanModule, task: Task) -> tuple[PlanDiagnostic, ...]:
    from vlaforge.plan.replay import REPLAY_MODES, analyze_replay_structure

    fields = {name for name in task.attributes if name.startswith("replay")}
    if not fields:
        return ()

    def error(message: str) -> tuple[PlanDiagnostic, ...]:
        return (PlanDiagnostic("replay.storage_contract", message, task.id),)

    mode = task.attributes.get("replay", "off")
    if task.opcode != "vla.for" or mode not in REPLAY_MODES:
        return error("replay request requires a loop and a valid mode")
    if mode == "off":
        return () if fields <= {"replay"} else error("disabled replay has storage metadata")
    reasons = task.attributes.get("replay_reasons")
    if not isinstance(reasons, (tuple, list)) or any(not isinstance(item, str) for item in reasons):
        return error("replay requires recorded structural analysis")
    try:
        analysis = analyze_replay_structure(plan, task)
        if set(analysis.reasons) - set(reasons):
            return error("replay analysis hides a structural rejection")
        if reasons:
            if fields - {"replay", "replay_reasons"}:
                return error("rejected replay unexpectedly owns capture storage")
            return ()
        seeds = task.attributes.get("replay_seeds")
        liveins = task.attributes.get("replay_liveins")
        staging = task.attributes.get("replay_staging")
        if any(not isinstance(items, (tuple, list)) for items in (seeds, liveins, staging)):
            return error("replay seeds/live-ins/staging must be explicit buffer lists")
        if tuple(liveins) != analysis.liveins or len(seeds) != len(task.inputs) or len(staging) != len(liveins):
            return error("replay storage arity or live-in set differs from loop data flow")
        reserved = tuple(seeds) + tuple(staging)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in reserved):
            return error("replay storage IDs must be integers")
        if len(set(reserved)) != len(reserved) or set(reserved).intersection(
            task.inputs + task.outputs + tuple(task.attributes.get("carry_scratch", ())) + tuple(liveins)
        ):
            return error("replay seeds/staging must use independent owned buffers")
        for source, destination in zip(task.inputs + tuple(liveins), reserved, strict=True):
            buffer = plan.buffer(destination)
            if buffer.type != plan.buffer(source).type or buffer.external or buffer.producer_task != task.id or buffer.buffer_class is not BufferClass.LOOP_CARRIED:
                return error("replay staging must match source type and loop ownership")
        if plan.arena is not None:
            physical = {index: item for item in plan.arena.physical_buffers for index in item.logical_buffers}
            last = max(plan.block(task.blocks[0]).tasks)
            if any(index not in physical or physical[index].first_task > task.id or physical[index].last_task < last for index in reserved):
                return error("replay reset storage is not live through the full loop")
    except (KeyError, IndexError, TypeError, ValueError):
        return error("replay metadata references malformed loop or buffer IDs")
    return ()


def _verify_task(
    task: Task,
    task_map: dict[int, Task],
    block_map: dict[int, object],
    buffer_map: dict[int, object],
    artifact_map: dict[int, object],
    diagnostics: list[PlanDiagnostic],
) -> None:
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
        producer = None if buffer is None else buffer.producer_task
        if producer is not None and producer not in reachable:
            diagnostics.append(
                PlanDiagnostic(
                    "buffer.read_before_produce",
                    f"buffer {buffer_id} is produced by task {producer} "
                    "without a dependency path",
                    task.id,
                )
            )

    if task.opcode == "vla.for" and len(task.blocks) == 1 and task.blocks[0] in block_map:
        body = block_map[task.blocks[0]]
        count = len(task.outputs)
        terminal = task_map.get(body.tasks[-1]) if body.tasks else None
        if (
            not count or len(task.inputs) != count
            or len(body.arguments) != count + 1
            or terminal is None or terminal.opcode != "vla.yield"
            or len(terminal.inputs) != count
        ):
            diagnostics.append(PlanDiagnostic(
                "loop.carry_shape", "loop inputs, arguments, yields and outputs must match", task.id,
            ))
        else:
            for ids in zip(task.inputs, body.arguments[1:], terminal.inputs, task.outputs):
                values = [buffer_map.get(index) for index in ids]
                if any(v is None for v in values) or len({v.type for v in values}) != 1:
                    diagnostics.append(PlanDiagnostic(
                        "loop.carry_type", "loop carried types must match", task.id,
                    ))
        if count > 1:
            scratch = task.attributes.get("carry_scratch", ())
            valid = isinstance(scratch, (tuple, list)) and len(scratch) == count
            if valid:
                valid = all(isinstance(index, int) and index in buffer_map for index in scratch)
            if valid:
                valid = len(set(scratch)) == count and not set(scratch).intersection(task.inputs + task.outputs)
            if valid:
                valid = all(
                    output in buffer_map
                    and buffer_map[index].type == buffer_map[output].type
                    and buffer_map[index].producer_task == task.id
                    and buffer_map[index].buffer_class is BufferClass.LOOP_CARRIED
                    and not buffer_map[index].external
                    for index, output in zip(scratch, task.outputs)
                )
            if not valid:
                diagnostics.append(PlanDiagnostic(
                    "loop.carry_scratch", "variadic carry requires independent typed scratch buffers", task.id,
                ))

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
        elif (
            artifact.workspace_size_bytes > 0
            and task.workspace_buffer is None
        ):
            diagnostics.append(
                PlanDiagnostic(
                    "workspace.binding_missing",
                    f"artifact {artifact.artifact_id} requires workspace",
                    task.id,
                )
            )
    elif task.artifact_id is not None:
        diagnostics.append(
            PlanDiagnostic(
                "artifact.non_region",
                "only region tasks may bind an artifact",
                task.id,
            )
        )

    if task.workspace_buffer is not None:
        workspace = buffer_map.get(task.workspace_buffer)
        if (
            workspace is None
            or workspace.buffer_class is not BufferClass.REGION_WORKSPACE
            or workspace.producer_task != task.id
        ):
            diagnostics.append(
                PlanDiagnostic(
                    "workspace.invalid_buffer",
                    f"buffer {task.workspace_buffer} is not this task's workspace",
                    task.id,
                )
            )

    if task.kind is TaskKind.LOOP:
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
                    "bounded for requires one body block",
                    task.id,
                )
            )
    if task.kind is TaskKind.BRANCH and len(task.blocks) != 2:
        diagnostics.append(
            PlanDiagnostic(
                "branch.invalid_body",
                "structured if requires then and else blocks",
                task.id,
            )
        )
    if task.kind is TaskKind.COMMIT:
        if len(task.inputs) != 3:
            diagnostics.append(
                PlanDiagnostic(
                    "commit.invalid_arity",
                    "commit requires transaction, output, and condition",
                    task.id,
                )
            )
        elif task.inputs[2] in buffer_map:
            producer = buffer_map[task.inputs[2]].producer_task
            if (
                producer is None
                or producer not in task_map
                or task_map[producer].kind is not TaskKind.VALIDATION
            ):
                diagnostics.append(
                    PlanDiagnostic(
                        "commit.missing_validation",
                        "commit condition is not produced by validation",
                        task.id,
                    )
                )


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
    task_id: int,
    task_map: dict[int, Task],
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
    task_map: dict[int, Task],
) -> tuple[PlanDiagnostic, ...]:
    diagnostics: list[PlanDiagnostic] = []
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


def _verify_state_layout(plan: PlanModule) -> tuple[PlanDiagnostic, ...]:
    diagnostics: list[PlanDiagnostic] = []
    by_device: dict[str, list[object]] = {}
    for state in plan.states:
        if state.slot_capacity is None:
            continue
        if state.slot_capacity < state.required_capacity:
            diagnostics.append(
                PlanDiagnostic(
                    "state.capacity",
                    f"state {state.name} capacity {state.slot_capacity} "
                    f"is below retention {state.required_capacity}",
                )
            )
        assert state.device is not None
        by_device.setdefault(state.device, []).append(state)
    for device, states in by_device.items():
        ordered = sorted(states, key=lambda item: item.offset)
        for left, right in zip(ordered, ordered[1:]):
            assert left.offset is not None
            assert left.total_size_bytes is not None
            assert right.offset is not None
            if left.offset + left.total_size_bytes > right.offset:
                diagnostics.append(
                    PlanDiagnostic(
                        "state.overlap",
                        f"state rings {left.name} and {right.name} overlap "
                        f"on {device}",
                    )
                )
    return tuple(diagnostics)


def _arena_eligible(buffer: object) -> bool:
    return (
        not buffer.external
        and buffer.buffer_class
        not in {
            BufferClass.EXTERNAL_INPUT,
            BufferClass.EXTERNAL_OUTPUT,
        }
    )


def _verify_arena(plan: PlanModule) -> tuple[PlanDiagnostic, ...]:
    assert plan.arena is not None
    diagnostics: list[PlanDiagnostic] = []
    _require_contiguous(
        "arena.physical_buffer_id",
        [item.id for item in plan.arena.physical_buffers],
        diagnostics,
    )
    logical_ids = {buffer.id for buffer in plan.buffers}
    eligible = {buffer.id for buffer in plan.buffers if _arena_eligible(buffer)}
    mapped = [
        logical_id
        for physical in plan.arena.physical_buffers
        for logical_id in physical.logical_buffers
    ]
    missing = sorted(eligible - set(mapped))
    duplicate = sorted(
        item for item in set(mapped) if mapped.count(item) > 1
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
                f"external I/O buffers were allocated: {unexpected}",
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
                    f"physical buffer {physical.id} device mismatch",
                )
            )
    for index, left in enumerate(plan.arena.physical_buffers):
        for right in plan.arena.physical_buffers[index + 1 :]:
            memory_overlap = (
                left.offset < right.offset + right.size_bytes
                and right.offset < left.offset + left.size_bytes
            )
            lifetime_overlap = not (
                left.last_task < right.first_task
                or right.last_task < left.first_task
            )
            if memory_overlap and lifetime_overlap:
                diagnostics.append(
                    PlanDiagnostic(
                        "arena.live_overlap",
                        f"physical buffers {left.id} and {right.id} "
                        "overlap while both are live",
                    )
                )
    return tuple(diagnostics)
