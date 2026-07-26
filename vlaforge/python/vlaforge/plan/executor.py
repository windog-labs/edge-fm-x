"""Reference executor for passive Scheduled Plan v2."""

from __future__ import annotations

from typing import Callable, Mapping

from vlaforge.analysis.verifier import verify
from vlaforge.interpreter.cache import ExactCache
from vlaforge.interpreter.executor import RunResult
from vlaforge.interpreter.inputs import (
    InputBinding,
    InputStamp,
    default_binding,
    resolve_binding,
)
from vlaforge.interpreter.state_store import StateStore
from vlaforge.interpreter.trace import Trace
from vlaforge.interpreter.transaction import (
    CommittedOutputGroup,
    PendingOutput,
    PendingOutputGroup,
    SnapshotValue,
    Transaction,
)
from vlaforge.ir.program import Module
from vlaforge.ir.serializer import io_schema_digest, module_digest
from vlaforge.plan.model import PlanInvocation, PlanModule, Task
from vlaforge.plan.verifier import verify_plan


RegionCallable = Callable[..., object]
ValidatorCallable = Callable[[object], bool]


class PlanExecutionError(RuntimeError):
    pass


class _YieldSignal(Exception):
    def __init__(self, values: tuple[object, ...]):
        self.values = values


class _ReturnSignal(Exception):
    def __init__(self, values: tuple[object, ...]):
        self.values = values


class PlanExecutor:
    """Execute a compiled plan through BindInput, Run, and ReadOutput."""

    def __init__(
        self,
        plan: PlanModule,
        semantic_module: Module,
        *,
        regions: Mapping[str, RegionCallable],
        validators: Mapping[str, ValidatorCallable],
        initial_state: Mapping[str, object] | None = None,
        expected_schema_digest: str | None = None,
    ):
        verify(semantic_module)
        verify_plan(plan)
        if module_digest(semantic_module) != plan.semantic_digest:
            raise PlanExecutionError(
                "scheduled plan semantic digest does not match source module"
            )
        actual_io_digest = io_schema_digest(semantic_module)
        if actual_io_digest != plan.io_schema_digest:
            raise PlanExecutionError(
                "scheduled plan I/O schema digest does not match source module"
            )
        if (
            expected_schema_digest is not None
            and expected_schema_digest != actual_io_digest
        ):
            raise PlanExecutionError(
                "input/output schema digest mismatch: "
                f"expected={expected_schema_digest} actual={actual_io_digest}"
            )
        self.plan = plan
        self.module = semantic_module
        self.regions = dict(regions)
        self.validators = dict(validators)
        missing_regions = {
            artifact.region_name for artifact in plan.artifacts
        } - set(self.regions)
        if missing_regions:
            raise PlanExecutionError(
                f"missing TensorRegion implementations: {sorted(missing_regions)}"
            )
        self.state_store = StateStore(
            semantic_module,
            initial_values=initial_state,
        )
        self.trace = Trace()
        self.schema_digest = actual_io_digest
        self._bindings: dict[str, InputBinding] = {}
        self._next_auto_revision = 1
        self._run_index = 0
        self._active_transaction: Transaction | None = None
        self._last_outputs: CommittedOutputGroup | None = None
        self._active_revisions: tuple[tuple[str, int], ...] = ()
        self._active_snapshots: dict[str, tuple[int, int]] = {}
        self.cache = ExactCache()

    def bind_input(
        self,
        input_id: int | str,
        value: object,
        stamp: InputStamp | None = None,
    ) -> None:
        port = (
            self.module.inputs[input_id]
            if isinstance(input_id, int)
            and 0 <= input_id < len(self.module.inputs)
            else self.module.input(str(input_id))
        )
        self._bindings[port.name] = InputBinding(
            value,
            InputStamp() if stamp is None else stamp,
        )

    def initialize_state(self, state: str, value: object) -> None:
        self.state_store.initialize(state, value)

    def run(
        self,
        invocation_name: str = "act",
        inputs: Mapping[str, InputBinding] | None = None,
    ) -> RunResult:
        if inputs is not None:
            for name, binding in inputs.items():
                if not isinstance(binding, InputBinding):
                    raise PlanExecutionError(
                        "run inputs must be InputBinding values"
                    )
                self.bind_input(name, binding.value, binding.stamp)
        invocation = self.plan.invocation(invocation_name)
        try:
            resolved = self._resolve_bindings()
        finally:
            self._bindings.clear()
        self._active_revisions = tuple(
            sorted((name, value[1]) for name, value in resolved.items())
        )
        self._active_snapshots = {}
        env: dict[int, object] = {}
        self._active_transaction = None
        run_index = self._run_index
        try:
            self._execute_block(
                invocation,
                invocation.body_block,
                env,
                resolved,
                run_index,
            )
            returns: tuple[object, ...] = ()
        except _ReturnSignal as signal:
            returns = signal.values
        except Exception as error:
            transaction = self._active_transaction
            if transaction is not None and not transaction.closed:
                self.state_store.abort(transaction)
                self._record(
                    invocation,
                    run_index,
                    "vla.txn.abort",
                    "transaction_abort",
                    {
                        "transaction_id": transaction.id,
                        "reason": "execution_error",
                    },
                )
            self._active_transaction = None
            if isinstance(error, PlanExecutionError):
                raise
            raise PlanExecutionError(str(error)) from error
        finally:
            self._run_index += 1

        if len(returns) != 1 or not isinstance(
            returns[0], CommittedOutputGroup
        ):
            raise PlanExecutionError(
                f"invocation @{invocation.name} did not return committed "
                "output group"
            )
        committed = returns[0]
        if self._last_outputs != committed:
            raise PlanExecutionError(
                "returned output group is not the committed output group"
            )
        return RunResult(
            returns=returns,
            committed_outputs=committed,
            trace=self.trace,
            state=self.state_store.inspect(),
        )

    def read_output(self, output_id: int | str = 0) -> object:
        if self._last_outputs is None:
            raise PlanExecutionError("no committed output is available")
        port = (
            self.module.outputs[output_id]
            if isinstance(output_id, int)
            and 0 <= output_id < len(self.module.outputs)
            else self.module.output(str(output_id))
        )
        try:
            return self._last_outputs.output(port.name)
        except KeyError as error:
            raise PlanExecutionError(
                f"latest output group does not contain @{port.name}"
            ) from error

    def read_output_group(self) -> CommittedOutputGroup:
        if self._last_outputs is None:
            raise PlanExecutionError("no committed output group is available")
        return self._last_outputs

    def reset_episode(self, new_episode: int) -> None:
        self.state_store.reset(new_episode)
        self._last_outputs = None
        self.cache.clear()
        self.trace.record(
            "reset",
            "session",
            self._run_index,
            "vla.session.reset",
            {"episode": new_episode},
        )

    def _resolve_bindings(self) -> dict[str, tuple[object, int, int | None]]:
        result: dict[str, tuple[object, int, int | None]] = {}
        for port in self.module.inputs:
            binding = self._bindings.get(port.name)
            if binding is None:
                if port.required:
                    raise PlanExecutionError(
                        f"required input @{port.name} is not bound"
                    )
                binding = default_binding(port)
            revision = binding.stamp.revision
            if revision is None:
                revision = self._next_auto_revision
                self._next_auto_revision += 1
            try:
                value = resolve_binding(port, binding)
            except (TypeError, ValueError) as error:
                raise PlanExecutionError(str(error)) from error
            result[port.name] = (value, revision, binding.stamp.timestamp_ns)
        return result

    def _execute_block(
        self,
        invocation: PlanInvocation,
        block_id: int,
        env: dict[int, object],
        inputs: Mapping[str, tuple[object, int, int | None]],
        run_index: int,
        arguments: tuple[object, ...] = (),
    ) -> None:
        block = self.plan.block(block_id)
        if len(arguments) != len(block.arguments):
            raise PlanExecutionError(
                f"block {block.id} expects {len(block.arguments)} arguments, "
                f"got {len(arguments)}"
            )
        for buffer_id, value in zip(
            block.arguments,
            arguments,
            strict=True,
        ):
            env[buffer_id] = value
        for task_id in block.tasks:
            task = self.plan.task(task_id)
            values = self._execute_task(
                invocation,
                task,
                env,
                inputs,
                run_index,
            )
            if len(values) != len(task.outputs):
                raise PlanExecutionError(
                    f"task {task.id} ({task.opcode}) produced {len(values)} "
                    f"values; plan declares {len(task.outputs)}"
                )
            for buffer_id, value in zip(
                task.outputs,
                values,
                strict=True,
            ):
                env[buffer_id] = value

    def _execute_task(
        self,
        invocation: PlanInvocation,
        task: Task,
        env: dict[int, object],
        inputs: Mapping[str, tuple[object, int, int | None]],
        run_index: int,
    ) -> tuple[object, ...]:
        opcode = task.opcode
        try:
            operands = tuple(env[buffer_id] for buffer_id in task.inputs)
        except KeyError as error:
            raise PlanExecutionError(
                f"task {task.id} reads unavailable buffer {error.args[0]}"
            ) from error

        if opcode == "vla.input.read":
            name = str(task.attributes["input"])
            value, revision, timestamp_ns = inputs[name]
            self._record(
                invocation,
                run_index,
                opcode,
                "input",
                {
                    "input": name,
                    "revision": revision,
                    "timestamp_ns": timestamp_ns,
                    "value": value,
                },
            )
            return value, revision

        if opcode == "vla.txn.begin":
            transaction = self.state_store.begin()
            self._active_transaction = transaction
            self._record(
                invocation,
                run_index,
                opcode,
                "transaction_begin",
                {"transaction_id": transaction.id},
            )
            return (transaction,)

        if opcode == "vla.state.read_latest":
            transaction = _expect(operands[0], Transaction, task)
            if transaction is not self._active_transaction:
                raise PlanExecutionError("state read uses inactive transaction")
            state_name = str(task.attributes["state"])
            snapshot = self.state_store.read_latest(state_name)
            self._active_snapshots[state_name] = (
                snapshot.episode,
                snapshot.version,
            )
            self._record(
                invocation,
                run_index,
                opcode,
                "state_read",
                snapshot,
            )
            return (snapshot,)

        if opcode == "vla.snapshot.value":
            snapshot = _expect(operands[0], SnapshotValue, task)
            return (snapshot.value,)

        if opcode == "vla.invoke":
            region_name = str(task.attributes["region"])
            call_args = tuple(_unwrap(value) for value in operands)
            region = self.module.region(region_name)
            executed = True
            if bool(region.metadata.get("memoize", False)):
                key = self._cache_key(task, region_name)
                cache_revisions, cache_snapshots = self._cache_identity(
                    region_name
                )
                lookup = self.cache.lookup(key)
                if lookup.hit:
                    result = lookup.value
                    executed = False
                else:
                    result = self.regions[region_name](*call_args)
                    self.cache.store(key, result)
                self._record(
                    invocation,
                    run_index,
                    opcode,
                    "cache",
                    {
                        "region": region_name,
                        "hit": lookup.hit,
                        "input_revisions": cache_revisions,
                        "state_snapshots": cache_snapshots,
                    },
                )
            else:
                result = self.regions[region_name](*call_args)
            expected = len(task.outputs)
            if expected == 0:
                values: tuple[object, ...] = ()
            elif expected == 1:
                values = (result,)
            elif not isinstance(result, tuple | list):
                raise PlanExecutionError(
                    f"region @{region_name} must return {expected} values"
                )
            else:
                values = tuple(result)
            if executed:
                self._record(
                    invocation,
                    run_index,
                    opcode,
                    "region",
                    {
                        "region": region_name,
                        "inputs": call_args,
                        "outputs": values,
                    },
                )
            return values

        if opcode == "vla.for":
            current = operands[0]
            for index in range(
                int(task.attributes["lower"]),
                int(task.attributes["upper"]),
                int(task.attributes["step"]),
            ):
                nested = dict(env)
                try:
                    self._execute_block(
                        invocation,
                        task.blocks[0],
                        nested,
                        inputs,
                        run_index,
                        (index, current),
                    )
                except _YieldSignal as signal:
                    if len(signal.values) != 1:
                        raise PlanExecutionError(
                            "vla.for body must yield one iter value"
                        )
                    current = signal.values[0]
                else:
                    raise PlanExecutionError("vla.for body did not yield")
            return (current,)

        if opcode == "vla.if":
            nested = dict(env)
            block_id = task.blocks[0 if bool(operands[0]) else 1]
            try:
                self._execute_block(
                    invocation,
                    block_id,
                    nested,
                    inputs,
                    run_index,
                )
            except _YieldSignal as signal:
                return signal.values
            raise PlanExecutionError("vla.if branch did not yield")

        if opcode == "vla.yield":
            raise _YieldSignal(operands)

        if opcode == "vla.return":
            raise _ReturnSignal(operands)

        if opcode == "vla.state.stage_write":
            transaction = _expect(operands[0], Transaction, task)
            pending = self.state_store.stage(
                transaction,
                str(task.attributes["state"]),
                _unwrap(operands[1]),
            )
            self._record(
                invocation,
                run_index,
                opcode,
                "state_stage",
                pending,
            )
            return (pending,)

        if opcode == "vla.validate":
            contract = str(task.attributes["contract"])
            if contract not in self.validators:
                raise PlanExecutionError(f"missing validator @{contract}")
            valid = bool(self.validators[contract](_unwrap(operands[0])))
            self._record(
                invocation,
                run_index,
                opcode,
                "validation",
                {"contract": contract, "valid": valid},
            )
            return (valid,)

        if opcode == "vla.output.create":
            pending = PendingOutput(
                str(task.attributes["output"]),
                _unwrap(operands[0]),
            )
            self._record(
                invocation,
                run_index,
                opcode,
                "output_pending",
                pending,
            )
            return (pending,)

        if opcode == "vla.output.group":
            group = PendingOutputGroup(
                str(task.attributes["group"]),
                tuple(
                    _expect(item, PendingOutput, task)
                    for item in operands
                ),
            )
            self._record(
                invocation,
                run_index,
                opcode,
                "output_group_pending",
                group,
            )
            return (group,)

        if opcode == "vla.txn.commit":
            transaction = _expect(operands[0], Transaction, task)
            pending = _expect(operands[1], PendingOutputGroup, task)
            if not bool(operands[2]):
                self.state_store.abort(transaction)
                self._active_transaction = None
                self._record(
                    invocation,
                    run_index,
                    opcode,
                    "transaction_abort",
                    {
                        "transaction_id": transaction.id,
                        "reason": "validation",
                    },
                )
                raise PlanExecutionError(
                    f"transaction {transaction.id} failed validation"
                )
            committed_states = self.state_store.commit(transaction)
            committed = CommittedOutputGroup(
                pending.group,
                pending.outputs,
                transaction.id,
                self.state_store.episode,
            )
            self._last_outputs = committed
            self._active_transaction = None
            self._record(
                invocation,
                run_index,
                opcode,
                "transaction_commit",
                {
                    "transaction_id": transaction.id,
                    "states": committed_states,
                    "output": committed,
                },
            )
            return (committed,)

        if opcode == "vla.txn.abort":
            transaction = _expect(operands[0], Transaction, task)
            self.state_store.abort(transaction)
            self._active_transaction = None
            self._record(
                invocation,
                run_index,
                opcode,
                "transaction_abort",
                {
                    "transaction_id": transaction.id,
                    "reason": task.attributes.get("reason", ""),
                },
            )
            return ()

        raise PlanExecutionError(f"unsupported task operation {opcode}")

    def _cache_key(
        self,
        task: Task,
        region_name: str,
    ) -> tuple[object, ...]:
        revisions, snapshots = self._cache_identity(region_name)
        artifact = (
            None
            if task.artifact_id is None
            else self.plan.artifacts[task.artifact_id]
        )
        artifact_identity = (
            None
            if artifact is None
            else (
                artifact.backend,
                artifact.variant,
                artifact.artifact_path,
                artifact.plugin_abi,
            )
        )
        return (
            self.plan.semantic_digest,
            region_name,
            artifact_identity,
            self.state_store.episode,
            revisions,
            snapshots,
        )

    def _cache_identity(
        self, region_name: str
    ) -> tuple[
        tuple[tuple[str, int], ...],
        tuple[tuple[str, tuple[int, int]], ...],
    ]:
        metadata = self.module.region(region_name).metadata
        selected_inputs = metadata.get("cache_input_ports")
        selected_states = metadata.get("cache_state_slots")
        input_names = (
            None
            if selected_inputs is None
            else {str(name) for name in selected_inputs}
        )
        state_names = (
            None
            if selected_states is None
            else {str(name) for name in selected_states}
        )
        revisions = tuple(
            item
            for item in self._active_revisions
            if input_names is None or item[0] in input_names
        )
        snapshots = tuple(
            item
            for item in sorted(self._active_snapshots.items())
            if state_names is None or item[0] in state_names
        )
        return revisions, snapshots

    def _record(
        self,
        invocation: PlanInvocation,
        run_index: int,
        opcode: str,
        kind: str,
        data: object,
    ) -> None:
        self.trace.record(
            kind,
            invocation.name,
            run_index,
            opcode,
            data,
        )


def _unwrap(value: object) -> object:
    if isinstance(value, SnapshotValue):
        return value.value
    return value


def _expect(value: object, expected: type, task: Task):
    if not isinstance(value, expected):
        raise PlanExecutionError(
            f"task {task.id} ({task.opcode}) expected {expected.__name__}, "
            f"got {type(value).__name__}"
        )
    return value
