"""Deterministic lowering from Invocation IR v0.2 to a passive run plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from vlaforge.analysis.verifier import verify
from vlaforge.ir.program import Block, Module, Operation, Value
from vlaforge.ir.serializer import io_schema_digest, module_digest
from vlaforge.ir.types import (
    CommittedOutputGroupType,
    InputRevisionType,
    PendingOutputGroupType,
    PendingOutputType,
    PendingType,
    ScalarType,
    SnapshotType,
    TransactionType,
)
from vlaforge.plan.model import (
    ArtifactBinding,
    BufferClass,
    LogicalBuffer,
    PlanBlock,
    PlanInvocation,
    PlanModule,
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
    workspace_size_bytes: int = 0
    workspace_alignment: int = 1
    workspace_device: str = "cpu"
    plugin_abi: str = "vlaforge.region_executable/2"

    def __post_init__(self) -> None:
        if (
            not self.backend
            or not self.variant
            or self.workspace_size_bytes < 0
            or self.workspace_alignment < 1
            or not self.workspace_device
            or not self.plugin_abi
        ):
            raise ValueError("artifact variant has an invalid deployment contract")
        if self.workspace_alignment & (self.workspace_alignment - 1):
            raise ValueError("artifact workspace alignment must be a power of two")


def lower_to_plan(
    module: Module,
    *,
    artifact_variants: Mapping[str, ArtifactVariant] | None = None,
) -> PlanModule:
    """Lower one verified caller-driven module without adding runtime policy."""

    verify(module)
    variants = dict(artifact_variants or {})
    unknown = sorted(variants.keys() - {region.name for region in module.regions})
    if unknown:
        raise KeyError(f"artifact variants reference unknown regions: {unknown}")
    plan = _Lowering(module, variants).run()
    verify_plan(plan)
    return plan


class _Lowering:
    def __init__(
        self,
        module: Module,
        variants: Mapping[str, ArtifactVariant],
    ):
        self.module = module
        self.tasks: list[Task] = []
        self.blocks: list[PlanBlock] = []
        self.buffers: list[LogicalBuffer] = []
        self.invocations: list[PlanInvocation] = []
        self.artifacts = tuple(
            ArtifactBinding(
                artifact_id=index,
                region_name=region.name,
                backend=variant.backend,
                variant=variant.variant,
                artifact_path=variant.artifact_path,
                workspace_size_bytes=variant.workspace_size_bytes,
                workspace_alignment=variant.workspace_alignment,
                workspace_device=variant.workspace_device,
                plugin_abi=variant.plugin_abi,
            )
            for index, region in enumerate(module.regions)
            for variant in (
                variants.get(
                    region.name,
                    ArtifactVariant("uncompiled", "default"),
                ),
            )
        )
        self.artifact_ids = {
            artifact.region_name: artifact.artifact_id
            for artifact in self.artifacts
        }
        self.states = tuple(
            StateBinding(
                state_id=index,
                name=state.name,
                payload=state.payload,
                retention=state.retention,
            )
            for index, state in enumerate(module.states)
        )

    def run(self) -> PlanModule:
        for invocation_id, invocation in enumerate(self.module.invocations):
            body = self._lower_block(
                invocation.body,
                {},
                source=f"invocation:{invocation.name}",
            )
            self.invocations.append(
                PlanInvocation(
                    id=invocation_id,
                    name=invocation.name,
                    body_block=body,
                )
            )
        return PlanModule(
            name=self.module.name,
            semantic_digest=module_digest(self.module),
            io_schema_digest=io_schema_digest(self.module),
            invocations=tuple(self.invocations),
            tasks=tuple(self.tasks),
            blocks=tuple(self.blocks),
            buffers=tuple(self.buffers),
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
        argument_ids: list[int] = []
        for argument in block.arguments:
            buffer_id = self._external_buffer(
                argument,
                BufferClass.LOOP_CARRIED,
                f"{source}:argument",
            )
            environment[argument.name] = buffer_id
            argument_ids.append(buffer_id)

        task_ids: list[int] = []
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
            try:
                input_ids = tuple(environment[name] for name in operation.operands)
            except KeyError as error:
                raise ValueError(
                    f"{op_source} references unknown SSA value %{error.args[0]}"
                ) from error
            dependencies = {
                producer
                for buffer_id in input_ids
                if (producer := self.buffers[buffer_id].producer_task) is not None
            }
            if previous_task is not None:
                dependencies.add(previous_task)

            output_ids: list[int] = []
            for result in operation.results:
                buffer_id = len(self.buffers)
                self.buffers.append(
                    LogicalBuffer(
                        id=buffer_id,
                        name=result.name,
                        type=result.type,
                        buffer_class=(
                            BufferClass.DERIVED_CACHE
                            if (
                                operation.opcode == "vla.invoke"
                                and bool(
                                    self.module.region(
                                        str(operation.attributes["region"])
                                    ).metadata.get("memoize", False)
                                )
                            )
                            else _buffer_class(operation, result)
                        ),
                        producer_task=task_id,
                        source=op_source,
                    )
                )
                environment[result.name] = buffer_id
                output_ids.append(buffer_id)

            artifact_id = (
                self.artifact_ids[str(operation.attributes["region"])]
                if operation.opcode == "vla.invoke"
                else None
            )
            workspace_buffer = None
            if artifact_id is not None:
                artifact = self.artifacts[artifact_id]
                if artifact.workspace_size_bytes:
                    workspace_buffer = len(self.buffers)
                    from vlaforge.ir.types import TensorType

                    self.buffers.append(
                        LogicalBuffer(
                            id=workspace_buffer,
                            name=f"task_{task_id}_workspace",
                            type=TensorType(
                                (artifact.workspace_size_bytes,),
                                "u8",
                            ),
                            buffer_class=BufferClass.REGION_WORKSPACE,
                            producer_task=task_id,
                            source=op_source,
                        )
                    )

            nested_blocks = tuple(
                self._lower_block(
                    region,
                    environment,
                    source=f"{op_source}/region:{region_index}",
                )
                for region_index, region in enumerate(operation.regions)
            )
            self.tasks[task_id] = Task(
                id=task_id,
                kind=_task_kind(operation.opcode),
                opcode=operation.opcode,
                inputs=input_ids,
                outputs=tuple(output_ids),
                dependencies=tuple(sorted(dependencies)),
                attributes=dict(operation.attributes),
                blocks=nested_blocks,
                artifact_id=artifact_id,
                workspace_buffer=workspace_buffer,
                source_op=operation.opcode,
                source_location=operation.location or op_source,
            )
            task_ids.append(task_id)
            previous_task = task_id

        self.blocks[block_id] = PlanBlock(
            id=block_id,
            arguments=tuple(argument_ids),
            tasks=tuple(task_ids),
            source=source,
        )
        return block_id

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
    if opcode == "vla.input.read":
        return TaskKind.INPUT
    if opcode == "vla.invoke":
        return TaskKind.REGION
    if opcode == "vla.for":
        return TaskKind.LOOP
    if opcode == "vla.if":
        return TaskKind.BRANCH
    if opcode.startswith("vla.state.") or opcode == "vla.snapshot.value":
        return TaskKind.STATE
    if opcode == "vla.validate":
        return TaskKind.VALIDATION
    if opcode == "vla.txn.commit":
        return TaskKind.COMMIT
    if opcode in {"vla.output.create", "vla.output.group"}:
        return TaskKind.OUTPUT
    return TaskKind.CONTROL


def _buffer_class(operation: Operation, result: Value) -> BufferClass:
    opcode = operation.opcode
    if opcode == "vla.input.read":
        return (
            BufferClass.CONTROL
            if isinstance(result.type, InputRevisionType)
            else BufferClass.EXTERNAL_INPUT
        )
    if opcode == "vla.invoke" and operation.attributes.get("memoize_key"):
        return BufferClass.DERIVED_CACHE
    if opcode == "vla.state.read_latest" or isinstance(
        result.type, SnapshotType
    ):
        return BufferClass.STATE_SNAPSHOT
    if opcode == "vla.state.stage_write" or isinstance(result.type, PendingType):
        return BufferClass.STATE_PENDING
    if opcode == "vla.output.create" or isinstance(
        result.type, PendingOutputType
    ) or isinstance(
        result.type, PendingOutputGroupType
    ):
        return BufferClass.PENDING_OUTPUT
    if opcode == "vla.txn.commit" or isinstance(
        result.type, CommittedOutputGroupType
    ):
        return BufferClass.COMMITTED_OUTPUT
    if opcode == "vla.for":
        return BufferClass.LOOP_CARRIED
    if isinstance(result.type, TransactionType | ScalarType | InputRevisionType):
        return BufferClass.CONTROL
    return BufferClass.SSA
