"""Deterministic lowering from normative Semantic IR to an internal plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Block, Module, Operation, Value
from vlaforge.ir.serializer import module_digest
from vlaforge.ir.types import (
    ActionType,
    CommittedActionType,
    EpochType,
    EventType,
    FutureType,
    PendingType,
    ScalarType,
    SnapshotType,
    TransactionType,
)
from vlaforge.plan.model import (
    ArtifactBinding,
    BufferClass,
    DeadlineGuard,
    FreshnessGuard,
    LogicalBuffer,
    PlanBlock,
    PlanModule,
    PlanPolicy,
    StateBinding,
    Task,
    TaskKind,
)
from vlaforge.plan.verifier import verify_plan


@dataclass(frozen=True, slots=True)
class ArtifactVariant:
    backend: str
    variant: str
    artifact_path: str | None = None

    def __post_init__(self) -> None:
        if not self.backend or not self.variant:
            raise ValueError("artifact variant requires backend and name")


def lower_to_plan(
    module: Module,
    *,
    artifact_variants: Mapping[str, ArtifactVariant] | None = None,
    max_in_flight: int = 1,
    consumer_lag: int = 0,
    fallback_snapshots: int = 0,
) -> PlanModule:
    """Lower a verified module without adding new program semantics."""

    verify(module)
    variants = dict(artifact_variants or {})
    unknown_variants = sorted(variants.keys() - {item.name for item in module.regions})
    if unknown_variants:
        raise KeyError(
            f"artifact variants reference unknown regions: {unknown_variants}"
        )
    lowering = _Lowering(
        module,
        variants,
        max_in_flight=max_in_flight,
        consumer_lag=consumer_lag,
        fallback_snapshots=fallback_snapshots,
    )
    plan = lowering.run()
    verify_plan(plan)
    return plan


class _Lowering:
    def __init__(
        self,
        module: Module,
        variants: Mapping[str, ArtifactVariant],
        *,
        max_in_flight: int,
        consumer_lag: int,
        fallback_snapshots: int,
    ):
        self.module = module
        self.tasks: list[Task] = []
        self.blocks: list[PlanBlock] = []
        self.buffers: list[LogicalBuffer] = []
        self.policies: list[PlanPolicy] = []
        self.artifacts = tuple(
            ArtifactBinding(
                artifact_id=index,
                region_name=region.name,
                backend=variants.get(
                    region.name, ArtifactVariant("uncompiled", "default")
                ).backend,
                variant=variants.get(
                    region.name, ArtifactVariant("uncompiled", "default")
                ).variant,
                artifact_path=variants.get(
                    region.name, ArtifactVariant("uncompiled", "default")
                ).artifact_path,
            )
            for index, region in enumerate(module.regions)
        )
        self.artifact_ids = {
            item.region_name: item.artifact_id for item in self.artifacts
        }
        self.states = tuple(
            StateBinding(
                state_id=index,
                name=state.name,
                payload=state.payload,
                retention=state.retention,
                max_in_flight=max_in_flight,
                consumer_lag=consumer_lag,
                fallback_snapshots=fallback_snapshots,
            )
            for index, state in enumerate(module.states)
        )

    def run(self) -> PlanModule:
        for policy_id, policy in enumerate(self.module.policies):
            environment: dict[str, int] = {}
            policy_inputs = []
            for value in policy.inputs:
                buffer_id = self._external_buffer(
                    value,
                    BufferClass.CONTROL
                    if isinstance(value.type, EpochType)
                    else BufferClass.EXTERNAL,
                    f"policy:{policy.name}:argument",
                )
                environment[value.name] = buffer_id
                policy_inputs.append(buffer_id)
            body = self._lower_block(
                policy.body,
                environment,
                source=f"policy:{policy.name}",
            )
            clock = self.module.clock(policy.clock)
            deadline = (
                None
                if clock.deadline_ns is None
                else DeadlineGuard(clock.deadline_ns)
            )
            self.policies.append(
                PlanPolicy(
                    id=policy_id,
                    name=policy.name,
                    clock=policy.clock,
                    inputs=tuple(policy_inputs),
                    body_block=body,
                    deadline_guard=deadline,
                )
            )
        return PlanModule(
            name=self.module.name,
            semantic_digest=module_digest(self.module),
            policies=tuple(self.policies),
            tasks=tuple(sorted(self.tasks, key=lambda item: item.id)),
            blocks=tuple(sorted(self.blocks, key=lambda item: item.id)),
            buffers=tuple(sorted(self.buffers, key=lambda item: item.id)),
            states=self.states,
            artifacts=self.artifacts,
        )

    def _lower_block(
        self,
        block: Block,
        outer_environment: Mapping[str, int],
        *,
        source: str,
    ) -> int:
        block_id = len(self.blocks)
        self.blocks.append(PlanBlock(block_id, (), (), source))
        environment = dict(outer_environment)
        argument_ids = []
        for argument in block.arguments:
            buffer_id = self._external_buffer(
                argument,
                BufferClass.LOOP_CARRIED,
                f"{source}:argument",
            )
            environment[argument.name] = buffer_id
            argument_ids.append(buffer_id)

        task_ids = []
        previous_task: int | None = None
        for operation_index, operation in enumerate(block.operations):
            op_source = f"{source}/{operation_index}:{operation.opcode}"
            task_id = len(self.tasks)
            self.tasks.append(
                Task(
                    id=task_id,
                    kind=TaskKind.CONTROL,
                    opcode="vlaforge.lowering.placeholder",
                    inputs=(),
                    outputs=(),
                    dependencies=(),
                    source_op=operation.opcode,
                    source_location=operation.location,
                )
            )
            input_ids = tuple(environment[name] for name in operation.operands)
            dependencies = {
                producer
                for buffer_id in input_ids
                if (producer := self.buffers[buffer_id].producer_task) is not None
            }
            if previous_task is not None:
                dependencies.add(previous_task)

            output_ids = []
            for result in operation.results:
                buffer_id = len(self.buffers)
                self.buffers.append(
                    LogicalBuffer(
                        id=buffer_id,
                        name=result.name,
                        type=result.type,
                        buffer_class=_buffer_class(operation, result),
                        producer_task=task_id,
                        external=False,
                        source=op_source,
                    )
                )
                environment[result.name] = buffer_id
                output_ids.append(buffer_id)

            nested_blocks = tuple(
                self._lower_block(
                    region,
                    environment,
                    source=f"{op_source}/region:{region_index}",
                )
                for region_index, region in enumerate(operation.regions)
            )
            task = Task(
                id=task_id,
                kind=_task_kind(operation.opcode),
                opcode=operation.opcode,
                inputs=input_ids,
                outputs=tuple(output_ids),
                dependencies=tuple(sorted(dependencies)),
                attributes=dict(operation.attributes),
                blocks=nested_blocks,
                artifact_id=(
                    self.artifact_ids[str(operation.attributes["region"])]
                    if operation.opcode == "vla.invoke"
                    else None
                ),
                source_op=operation.opcode,
                source_location=operation.location or op_source,
                freshness_guard=self._freshness_guard(operation),
                deadline_guard=None,
            )
            self.tasks[task_id] = task
            task_ids.append(task_id)
            previous_task = task_id

        self.blocks[block_id] = PlanBlock(
            id=block_id,
            arguments=tuple(argument_ids),
            tasks=tuple(task_ids),
            source=source,
        )
        return block_id

    def _freshness_guard(
        self, operation: Operation
    ) -> FreshnessGuard | None:
        if operation.opcode == "vla.sample_input":
            maximum = operation.attributes.get("max_age_ns")
            return (
                None
                if maximum is None
                else FreshnessGuard(max_age_ns=int(maximum))
            )
        if operation.opcode == "vla.state.read":
            state = self.module.state(str(operation.attributes["state"]))
            if state.freshness is not None:
                constraint = state.freshness
                if (
                    constraint.max_age_ns is not None
                    or constraint.max_versions is not None
                ):
                    return FreshnessGuard(
                        max_age_ns=constraint.max_age_ns,
                        max_versions=constraint.max_versions,
                    )
        return None

    def _external_buffer(
        self,
        value: Value,
        buffer_class: BufferClass,
        source: str,
    ) -> int:
        buffer_id = len(self.buffers)
        self.buffers.append(
            LogicalBuffer(
                id=buffer_id,
                name=value.name,
                type=value.type,
                buffer_class=buffer_class,
                producer_task=None,
                external=True,
                source=source,
            )
        )
        return buffer_id


def _task_kind(opcode: str) -> TaskKind:
    if opcode == "vla.sample_input":
        return TaskKind.INPUT
    if opcode == "vla.invoke":
        return TaskKind.REGION
    if opcode in {"vla.for", "vla.while"}:
        return TaskKind.LOOP
    if opcode == "vla.if":
        return TaskKind.BRANCH
    if opcode.startswith("vla.state.") or opcode in {
        "vla.snapshot.value",
        "vla.reset",
    }:
        return TaskKind.STATE
    if opcode == "vla.validate":
        return TaskKind.VALIDATION
    if opcode == "vla.txn.commit":
        return TaskKind.COMMIT
    if opcode == "vla.action.publish":
        return TaskKind.PUBLISH
    return TaskKind.CONTROL


def _buffer_class(operation: Operation, result: Value) -> BufferClass:
    opcode = operation.opcode
    if opcode == "vla.sample_input":
        return (
            BufferClass.CONTROL
            if isinstance(result.type, EpochType)
            else BufferClass.EXTERNAL
        )
    if opcode == "vla.state.read" or isinstance(result.type, SnapshotType):
        return BufferClass.STATE_SNAPSHOT
    if opcode == "vla.state.stage_write" or isinstance(result.type, PendingType):
        return BufferClass.STATE_PENDING
    if opcode == "vla.action.create" or isinstance(result.type, ActionType):
        return BufferClass.PENDING_ACTION
    if opcode == "vla.txn.commit" or isinstance(
        result.type, CommittedActionType
    ):
        return BufferClass.COMMITTED_ACTION
    if opcode in {"vla.for", "vla.while"}:
        return BufferClass.LOOP_CARRIED
    if isinstance(
        result.type,
        TransactionType | EpochType | ScalarType | EventType | FutureType,
    ):
        return BufferClass.CONTROL
    return BufferClass.SSA
