"""Deterministic interpreter for the normative Python IR semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from vlaforge.analysis.verifier import verify
from vlaforge.interpreter.clocks import Epoch, InputSample, resolve_epoch
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
from vlaforge.ir.program import Block, Module, Operation, Policy
from vlaforge.ir.types import EpochType


RegionCallable = Callable[..., object]
ValidatorCallable = Callable[[object], bool]


class InterpreterError(RuntimeError):
    pass


class _YieldSignal(Exception):
    def __init__(self, values: tuple[object, ...]):
        self.values = values


class _ReturnSignal(Exception):
    def __init__(self, values: tuple[object, ...]):
        self.values = values


@dataclass(frozen=True, slots=True)
class RunResult:
    returns: tuple[object, ...]
    published_actions: tuple[CommittedAction, ...]
    trace: Trace
    state: dict[str, list[dict[str, object]]]


class Interpreter:
    def __init__(
        self,
        module: Module,
        *,
        regions: Mapping[str, RegionCallable],
        validators: Mapping[str, ValidatorCallable],
        initial_state: Mapping[str, object] | None = None,
    ):
        verify(module)
        self.module = module
        self.regions = dict(regions)
        self.validators = dict(validators)
        missing_regions = {region.name for region in module.regions} - set(self.regions)
        if missing_regions:
            raise InterpreterError(
                f"missing TensorRegion implementations: {sorted(missing_regions)}"
            )
        self.state_store = StateStore(module, initial_values=initial_state)
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
        policy = self.module.policy(policy_name)
        if tick.clock != policy.clock:
            raise InterpreterError(
                f"policy @{policy.name} requires clock @{policy.clock}, "
                f"got @{tick.clock}"
            )
        env = dict(arguments or {})
        for argument in policy.inputs:
            if argument.name in env:
                continue
            if argument.type == EpochType(policy.clock):
                env[argument.name] = tick
            else:
                raise InterpreterError(
                    f"missing policy argument %{argument.name} for @{policy.name}"
                )
        epochs = {tick.clock: tick}
        for stream_name, sample in inputs.items():
            epochs[sample.epoch.clock] = sample.epoch
            if stream_name not in {stream.name for stream in self.module.inputs}:
                raise InterpreterError(f"unknown runtime input stream @{stream_name}")
        start_actions = len(self._published)
        try:
            self._execute_block(policy, policy.body, env, tick, inputs, epochs)
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
        policy: Policy,
        block: Block,
        env: dict[str, object],
        tick: Epoch,
        inputs: Mapping[str, InputSample],
        epochs: dict[str, Epoch],
    ) -> None:
        for operation in block.operations:
            results = self._execute_operation(
                policy, operation, env, tick, inputs, epochs
            )
            if len(results) != len(operation.results):
                raise InterpreterError(
                    f"{operation.opcode} produced {len(results)} values; "
                    f"IR declares {len(operation.results)}"
                )
            for result, value in zip(operation.results, results, strict=True):
                env[result.name] = value

    def _execute_operation(
        self,
        policy: Policy,
        operation: Operation,
        env: dict[str, object],
        tick: Epoch,
        inputs: Mapping[str, InputSample],
        epochs: dict[str, Epoch],
    ) -> tuple[object, ...]:
        opcode = operation.opcode
        operands = tuple(env[name] for name in operation.operands)

        if opcode == "vla.sample_input":
            stream_name = str(operation.attributes["stream"])
            if stream_name not in inputs:
                raise InterpreterError(f"missing input sample @{stream_name}")
            sample = inputs[stream_name]
            if sample.epoch.episode != tick.episode:
                raise InterpreterError(
                    f"input @{stream_name} belongs to episode "
                    f"{sample.epoch.episode}, current episode is {tick.episode}"
                )
            if sample.epoch.timestamp_ns > tick.timestamp_ns:
                raise InterpreterError(
                    f"input @{stream_name} is from the future: "
                    f"{sample.epoch.timestamp_ns}>{tick.timestamp_ns}"
                )
            max_age = operation.attributes.get("max_age_ns")
            age = tick.timestamp_ns - sample.epoch.timestamp_ns
            if max_age is not None and age > int(max_age):
                raise InterpreterError(
                    f"program={self.module.name} rule=freshness.stale_input "
                    f"op={opcode} stream={stream_name} epoch={sample.epoch.sequence} "
                    f"age_ns={age} max_age_ns={max_age}"
                )
            epochs[sample.epoch.clock] = sample.epoch
            self._record(policy, tick, opcode, "input", {
                "stream": stream_name,
                "epoch": sample.epoch,
                "value": sample.value,
            })
            return sample.value, sample.epoch

        if opcode == "vla.txn.begin":
            transaction = self.state_store.begin(operands[0])
            self._record(policy, tick, opcode, "transaction_begin", {
                "transaction_id": transaction.id
            })
            return (transaction,)

        if opcode == "vla.state.read":
            transaction = _expect(operands[0], Transaction, opcode)
            state_name = str(operation.attributes["state"])
            expression = EpochExpr.from_dict(operation.attributes["epoch"])
            resolved = resolve_epoch(expression, epochs, fallback=transaction.tick)
            version_mode = str(operation.attributes.get("version", "latest"))
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
            slot = self.module.state(state_name)
            if (
                slot.freshness is not None
                and slot.freshness.max_versions is not None
                and resolved.sequence - snapshot.epoch.sequence
                > slot.freshness.max_versions
            ):
                raise InterpreterError(
                    f"program={self.module.name} rule=freshness.stale_state "
                    f"op={opcode} state={state_name} epoch={resolved.sequence} "
                    f"version={snapshot.version}"
                )
            self._record(policy, tick, opcode, "state_read", snapshot)
            return (snapshot,)

        if opcode == "vla.snapshot.value":
            snapshot = _expect(operands[0], SnapshotValue, opcode)
            return (snapshot.value,)

        if opcode == "vla.invoke":
            region_name = str(operation.attributes["region"])
            call_args = tuple(_unwrap(value) for value in operands)
            result = self.regions[region_name](*call_args)
            expected = len(operation.results)
            if expected == 0:
                values: tuple[object, ...] = ()
            elif expected == 1:
                values = (result,)
            else:
                if not isinstance(result, tuple | list):
                    raise InterpreterError(
                        f"region @{region_name} must return {expected} values"
                    )
                values = tuple(result)
            self._record(policy, tick, opcode, "region", {
                "region": region_name,
                "inputs": call_args,
                "outputs": values,
            })
            return values

        if opcode == "vla.for":
            current = operands[0]
            induction_name = str(operation.attributes["induction"])
            iter_name = str(operation.attributes["iter_arg"])
            body = operation.regions[0]
            for index in range(
                int(operation.attributes["lower"]),
                int(operation.attributes["upper"]),
                int(operation.attributes["step"]),
            ):
                nested = dict(env)
                nested[induction_name] = index
                nested[iter_name] = current
                try:
                    self._execute_block(policy, body, nested, tick, inputs, epochs)
                except _YieldSignal as signal:
                    if len(signal.values) != 1:
                        raise InterpreterError("vla.for body must yield one iter value")
                    current = signal.values[0]
                else:
                    raise InterpreterError("vla.for body did not yield")
            return (current,)

        if opcode == "vla.while":
            carry = operands
            condition_block, body_block = operation.regions
            for _ in range(int(operation.attributes["max_iterations"])):
                condition_env = dict(env)
                for argument, value in zip(
                    condition_block.arguments, carry, strict=True
                ):
                    condition_env[argument.name] = value
                try:
                    self._execute_block(
                        policy, condition_block, condition_env, tick, inputs, epochs
                    )
                except _YieldSignal as signal:
                    if not signal.values:
                        raise InterpreterError("while condition yielded no predicate")
                    predicate = bool(signal.values[0])
                    condition_carry = signal.values[1:] or carry
                else:
                    raise InterpreterError("while condition did not yield")
                if not predicate:
                    return tuple(condition_carry)
                body_env = dict(env)
                for argument, value in zip(
                    body_block.arguments, condition_carry, strict=True
                ):
                    body_env[argument.name] = value
                try:
                    self._execute_block(
                        policy, body_block, body_env, tick, inputs, epochs
                    )
                except _YieldSignal as signal:
                    carry = signal.values
                else:
                    raise InterpreterError("while body did not yield")
            raise InterpreterError("vla.while exceeded max_iterations")

        if opcode == "vla.if":
            block = operation.regions[0 if bool(operands[0]) else 1]
            nested = dict(env)
            try:
                self._execute_block(policy, block, nested, tick, inputs, epochs)
            except _YieldSignal as signal:
                return signal.values
            raise InterpreterError("vla.if branch did not yield")

        if opcode == "vla.yield":
            raise _YieldSignal(operands)

        if opcode == "vla.return":
            raise _ReturnSignal(operands)

        if opcode == "vla.state.stage_write":
            transaction = _expect(operands[0], Transaction, opcode)
            state_name = str(operation.attributes["state"])
            expression = EpochExpr.from_dict(operation.attributes["epoch"])
            resolved = resolve_epoch(expression, epochs, fallback=transaction.tick)
            pending = self.state_store.stage(
                transaction, state_name, resolved, _unwrap(operands[1])
            )
            self._record(policy, tick, opcode, "state_stage", pending)
            return (pending,)

        if opcode == "vla.validate":
            contract = str(operation.attributes["contract"])
            if contract not in self.validators:
                raise InterpreterError(f"missing validator @{contract}")
            valid = bool(self.validators[contract](_unwrap(operands[0])))
            self._record(policy, tick, opcode, "validation", {
                "contract": contract,
                "valid": valid,
            })
            return (valid,)

        if opcode == "vla.action.create":
            epoch = _expect(operands[1], Epoch, opcode)
            action = PendingAction(epoch, _unwrap(operands[0]))
            self._record(policy, tick, opcode, "action_pending", action)
            return (action,)

        if opcode == "vla.txn.commit":
            transaction = _expect(operands[0], Transaction, opcode)
            action = _expect(operands[1], PendingAction, opcode)
            if not bool(operands[2]):
                self.state_store.abort(transaction)
                self._record(policy, tick, opcode, "transaction_abort", {
                    "transaction_id": transaction.id,
                    "reason": "validation",
                })
                raise InterpreterError(
                    f"transaction {transaction.id} failed validation"
                )
            committed_states = self.state_store.commit(transaction)
            committed_action = CommittedAction(
                action.epoch, action.value, transaction.id
            )
            self._record(policy, tick, opcode, "transaction_commit", {
                "transaction_id": transaction.id,
                "states": committed_states,
                "action": committed_action,
            })
            return (committed_action,)

        if opcode == "vla.txn.abort":
            transaction = _expect(operands[0], Transaction, opcode)
            self.state_store.abort(transaction)
            self._record(policy, tick, opcode, "transaction_abort", {
                "transaction_id": transaction.id,
                "reason": operation.attributes.get("reason", ""),
            })
            return ()

        if opcode == "vla.action.publish":
            action = _expect(operands[0], CommittedAction, opcode)
            self._published.append(action)
            self._record(policy, tick, opcode, "action_publish", action)
            return ()

        if opcode == "vla.reset":
            states = tuple(str(item) for item in operation.attributes["states"])
            self.state_store.reset(tick.episode + 1, states)
            self._record(policy, tick, opcode, "reset", {"states": states})
            return ()

        if opcode == "vla.async":
            nested = dict(env)
            try:
                self._execute_block(
                    policy, operation.regions[0], nested, tick, inputs, epochs
                )
            except _YieldSignal as signal:
                if len(signal.values) != 1:
                    raise InterpreterError("vla.async must yield one value")
                future = FutureValue(signal.values[0], completed=True)
            else:
                raise InterpreterError("vla.async body did not yield")
            self._record(policy, tick, opcode, "async_complete", future.value)
            return future, EventValue(True)

        if opcode == "vla.await":
            future = _expect(operands[0], FutureValue, opcode)
            if not future.completed:
                raise InterpreterError("reference interpreter received incomplete future")
            self._record(policy, tick, opcode, "await", future.value)
            return (future.value,)

        raise InterpreterError(f"unsupported operation: {opcode}")

    def _record(
        self,
        policy: Policy,
        tick: Epoch,
        op: str,
        kind: str,
        data: object,
    ) -> None:
        self.trace.record(kind, policy.name, tick, op, data)


def _unwrap(value: object) -> object:
    if isinstance(value, SnapshotValue):
        return value.value
    return value


def _expect(value: object, expected: type, opcode: str):
    if not isinstance(value, expected):
        raise InterpreterError(
            f"{opcode} expected {expected.__name__}, got {type(value).__name__}"
        )
    return value
