"""Reference executor for verified Scheduled Execution Plans."""

from __future__ import annotations

from typing import Callable, Mapping

from vlaforge.analysis.verifier import verify
from vlaforge.interpreter.clocks import Epoch, InputSample, resolve_epoch
from vlaforge.interpreter.executor import RunResult
from vlaforge.interpreter.state_store import StateStore
from vlaforge.interpreter.trace import Trace
from vlaforge.interpreter.transaction import (
    CommittedAction,
    EventValue,
    FutureValue,
    PendingAction,
    SnapshotValue,
    Transaction,
)
from vlaforge.ir.attrs import EpochExpr
from vlaforge.ir.program import Module
from vlaforge.ir.serializer import module_digest
from vlaforge.ir.types import EpochType
from vlaforge.plan.model import PlanModule, PlanPolicy, Task
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
    """Execute Plan tasks while preserving normative state/action semantics."""

    def __init__(
        self,
        plan: PlanModule,
        semantic_module: Module,
        *,
        regions: Mapping[str, RegionCallable],
        validators: Mapping[str, ValidatorCallable],
        initial_state: Mapping[str, object] | None = None,
    ):
        verify(semantic_module)
        verify_plan(plan)
        if module_digest(semantic_module) != plan.semantic_digest:
            raise PlanExecutionError(
                "scheduled plan semantic digest does not match source module"
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
            semantic_module, initial_values=initial_state
        )
        self.trace = Trace()
        self._published: list[CommittedAction] = []

    def run_tick(
        self,
        policy_name: str,
        tick: Epoch,
        inputs: Mapping[str, InputSample],
        *,
        arguments: Mapping[str, object] | None = None,
    ) -> RunResult:
        policy = self.plan.policy(policy_name)
        if tick.clock != policy.clock:
            raise PlanExecutionError(
                f"plan policy @{policy.name} requires clock @{policy.clock}, "
                f"got @{tick.clock}"
            )
        env: dict[int, object] = {}
        runtime_arguments = dict(arguments or {})
        for buffer_id in policy.inputs:
            buffer = self.plan.buffer(buffer_id)
            if buffer.name in runtime_arguments:
                env[buffer_id] = runtime_arguments[buffer.name]
            elif buffer.type == EpochType(policy.clock):
                env[buffer_id] = tick
            else:
                raise PlanExecutionError(
                    f"missing policy argument %{buffer.name} for @{policy.name}"
                )
        epochs = {tick.clock: tick}
        known_inputs = {stream.name for stream in self.module.inputs}
        for stream_name, sample in inputs.items():
            if stream_name not in known_inputs:
                raise PlanExecutionError(
                    f"unknown runtime input stream @{stream_name}"
                )
            epochs[sample.epoch.clock] = sample.epoch
        start_actions = len(self._published)
        try:
            self._execute_block(
                policy,
                policy.body_block,
                env,
                tick,
                inputs,
                epochs,
            )
            returns: tuple[object, ...] = ()
        except _ReturnSignal as signal:
            returns = signal.values
        return RunResult(
            returns=returns,
            published_actions=tuple(self._published[start_actions:]),
            trace=self.trace,
            state=self.state_store.inspect(),
        )

    def reset_episode(self, new_episode: int) -> None:
        self.state_store.reset(new_episode)

    def _execute_block(
        self,
        policy: PlanPolicy,
        block_id: int,
        env: dict[int, object],
        tick: Epoch,
        inputs: Mapping[str, InputSample],
        epochs: dict[str, Epoch],
    ) -> None:
        block = self.plan.block(block_id)
        for task_id in block.tasks:
            task = self.plan.task(task_id)
            values = self._execute_task(
                policy, task, env, tick, inputs, epochs
            )
            if len(values) != len(task.outputs):
                raise PlanExecutionError(
                    f"task {task.id} ({task.opcode}) produced {len(values)} "
                    f"values; plan declares {len(task.outputs)}"
                )
            for buffer_id, value in zip(task.outputs, values, strict=True):
                env[buffer_id] = value

    def _execute_task(
        self,
        policy: PlanPolicy,
        task: Task,
        env: dict[int, object],
        tick: Epoch,
        inputs: Mapping[str, InputSample],
        epochs: dict[str, Epoch],
    ) -> tuple[object, ...]:
        opcode = task.opcode
        operands = tuple(env[buffer_id] for buffer_id in task.inputs)

        if opcode == "vla.sample_input":
            stream_name = str(task.attributes["stream"])
            if stream_name not in inputs:
                raise PlanExecutionError(f"missing input sample @{stream_name}")
            sample = inputs[stream_name]
            if sample.epoch.episode != tick.episode:
                raise PlanExecutionError(
                    f"input @{stream_name} belongs to episode "
                    f"{sample.epoch.episode}, current episode is {tick.episode}"
                )
            if sample.epoch.timestamp_ns > tick.timestamp_ns:
                raise PlanExecutionError(
                    f"input @{stream_name} is from the future"
                )
            maximum = (
                None
                if task.freshness_guard is None
                else task.freshness_guard.max_age_ns
            )
            age = tick.timestamp_ns - sample.epoch.timestamp_ns
            if maximum is not None and age > maximum:
                raise PlanExecutionError(
                    f"plan={self.plan.name} rule=freshness.stale_input "
                    f"task={task.id} stream={stream_name} "
                    f"epoch={sample.epoch.sequence} age_ns={age} "
                    f"max_age_ns={maximum}"
                )
            epochs[sample.epoch.clock] = sample.epoch
            self._record(
                policy,
                tick,
                opcode,
                "input",
                {
                    "stream": stream_name,
                    "epoch": sample.epoch,
                    "value": sample.value,
                },
            )
            return sample.value, sample.epoch

        if opcode == "vla.txn.begin":
            transaction = self.state_store.begin(operands[0])
            self._record(
                policy,
                tick,
                opcode,
                "transaction_begin",
                {"transaction_id": transaction.id},
            )
            return (transaction,)

        if opcode == "vla.state.read":
            transaction = _expect(operands[0], Transaction, task)
            state_name = str(task.attributes["state"])
            expression = EpochExpr.from_dict(task.attributes["epoch"])
            resolved = resolve_epoch(
                expression, epochs, fallback=transaction.tick
            )
            version_mode = str(task.attributes.get("version", "latest"))
            snapshot = self.state_store.read(
                state_name,
                episode=tick.episode,
                exact_sequence=(
                    resolved.sequence if version_mode == "exact" else None
                ),
                max_sequence=(
                    None if version_mode == "exact" else resolved.sequence
                ),
            )
            maximum_versions = (
                None
                if task.freshness_guard is None
                else task.freshness_guard.max_versions
            )
            if (
                maximum_versions is not None
                and resolved.sequence - snapshot.epoch.sequence
                > maximum_versions
            ):
                raise PlanExecutionError(
                    f"plan={self.plan.name} rule=freshness.stale_state "
                    f"task={task.id} state={state_name} "
                    f"epoch={resolved.sequence} version={snapshot.version}"
                )
            self._record(
                policy, tick, opcode, "state_read", snapshot
            )
            return (snapshot,)

        if opcode == "vla.snapshot.value":
            return (_expect(operands[0], SnapshotValue, task).value,)

        if opcode == "vla.invoke":
            region_name = str(task.attributes["region"])
            call_args = tuple(_unwrap(value) for value in operands)
            result = self.regions[region_name](*call_args)
            if len(task.outputs) == 0:
                values: tuple[object, ...] = ()
            elif len(task.outputs) == 1:
                values = (result,)
            elif isinstance(result, tuple | list):
                values = tuple(result)
            else:
                raise PlanExecutionError(
                    f"region @{region_name} must return "
                    f"{len(task.outputs)} values"
                )
            self._record(
                policy,
                tick,
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
            body = self.plan.block(task.blocks[0])
            if len(body.arguments) != 2:
                raise PlanExecutionError("vla.for body requires two arguments")
            for index in range(
                int(task.attributes["lower"]),
                int(task.attributes["upper"]),
                int(task.attributes["step"]),
            ):
                nested = dict(env)
                nested[body.arguments[0]] = index
                nested[body.arguments[1]] = current
                try:
                    self._execute_block(
                        policy, body.id, nested, tick, inputs, epochs
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

        if opcode == "vla.while":
            carry = operands
            condition = self.plan.block(task.blocks[0])
            body = self.plan.block(task.blocks[1])
            for _ in range(int(task.attributes["max_iterations"])):
                condition_env = dict(env)
                for buffer_id, value in zip(
                    condition.arguments, carry, strict=True
                ):
                    condition_env[buffer_id] = value
                try:
                    self._execute_block(
                        policy,
                        condition.id,
                        condition_env,
                        tick,
                        inputs,
                        epochs,
                    )
                except _YieldSignal as signal:
                    if not signal.values:
                        raise PlanExecutionError(
                            "while condition yielded no predicate"
                        )
                    predicate = bool(signal.values[0])
                    condition_carry = signal.values[1:] or carry
                else:
                    raise PlanExecutionError(
                        "while condition did not yield"
                    )
                if not predicate:
                    return tuple(condition_carry)
                body_env = dict(env)
                for buffer_id, value in zip(
                    body.arguments, condition_carry, strict=True
                ):
                    body_env[buffer_id] = value
                try:
                    self._execute_block(
                        policy, body.id, body_env, tick, inputs, epochs
                    )
                except _YieldSignal as signal:
                    carry = signal.values
                else:
                    raise PlanExecutionError("while body did not yield")
            raise PlanExecutionError("vla.while exceeded max_iterations")

        if opcode == "vla.if":
            branch = self.plan.block(
                task.blocks[0 if bool(operands[0]) else 1]
            )
            nested = dict(env)
            try:
                self._execute_block(
                    policy, branch.id, nested, tick, inputs, epochs
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
            state_name = str(task.attributes["state"])
            expression = EpochExpr.from_dict(task.attributes["epoch"])
            resolved = resolve_epoch(
                expression, epochs, fallback=transaction.tick
            )
            pending = self.state_store.stage(
                transaction, state_name, resolved, _unwrap(operands[1])
            )
            self._record(
                policy, tick, opcode, "state_stage", pending
            )
            return (pending,)

        if opcode == "vla.validate":
            contract = str(task.attributes["contract"])
            if contract not in self.validators:
                raise PlanExecutionError(f"missing validator @{contract}")
            valid = bool(self.validators[contract](_unwrap(operands[0])))
            self._record(
                policy,
                tick,
                opcode,
                "validation",
                {"contract": contract, "valid": valid},
            )
            return (valid,)

        if opcode == "vla.action.create":
            epoch = _expect(operands[1], Epoch, task)
            action = PendingAction(epoch, _unwrap(operands[0]))
            self._record(
                policy, tick, opcode, "action_pending", action
            )
            return (action,)

        if opcode == "vla.txn.commit":
            transaction = _expect(operands[0], Transaction, task)
            action = _expect(operands[1], PendingAction, task)
            if not bool(operands[2]):
                self.state_store.abort(transaction)
                self._record(
                    policy,
                    tick,
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
            committed_action = CommittedAction(
                action.epoch, action.value, transaction.id
            )
            self._record(
                policy,
                tick,
                opcode,
                "transaction_commit",
                {
                    "transaction_id": transaction.id,
                    "states": committed_states,
                    "action": committed_action,
                },
            )
            return (committed_action,)

        if opcode == "vla.txn.abort":
            transaction = _expect(operands[0], Transaction, task)
            self.state_store.abort(transaction)
            self._record(
                policy,
                tick,
                opcode,
                "transaction_abort",
                {
                    "transaction_id": transaction.id,
                    "reason": task.attributes.get("reason", ""),
                },
            )
            return ()

        if opcode == "vla.action.publish":
            action = _expect(operands[0], CommittedAction, task)
            self._published.append(action)
            self._record(
                policy, tick, opcode, "action_publish", action
            )
            return ()

        if opcode == "vla.reset":
            states = tuple(str(item) for item in task.attributes["states"])
            self.state_store.reset(tick.episode + 1, states)
            self._record(
                policy, tick, opcode, "reset", {"states": states}
            )
            return ()

        if opcode == "vla.async":
            nested = dict(env)
            try:
                self._execute_block(
                    policy, task.blocks[0], nested, tick, inputs, epochs
                )
            except _YieldSignal as signal:
                if len(signal.values) != 1:
                    raise PlanExecutionError(
                        "vla.async must yield one value"
                    )
                future = FutureValue(signal.values[0], completed=True)
            else:
                raise PlanExecutionError("vla.async body did not yield")
            self._record(
                policy, tick, opcode, "async_complete", future.value
            )
            return future, EventValue(True)

        if opcode == "vla.await":
            future = _expect(operands[0], FutureValue, task)
            if not future.completed:
                raise PlanExecutionError(
                    "reference plan received incomplete future"
                )
            self._record(
                policy, tick, opcode, "await", future.value
            )
            return (future.value,)

        raise PlanExecutionError(
            f"unsupported task {task.id}: {task.opcode}"
        )

    def _record(
        self,
        policy: PlanPolicy,
        tick: Epoch,
        op: str,
        kind: str,
        data: object,
    ) -> None:
        self.trace.record(kind, policy.name, tick, op, data)


def _unwrap(value: object) -> object:
    return value.value if isinstance(value, SnapshotValue) else value


def _expect(value: object, expected: type, task: Task):
    if not isinstance(value, expected):
        raise PlanExecutionError(
            f"task {task.id} ({task.opcode}) expected {expected.__name__}, "
            f"got {type(value).__name__}"
        )
    return value
