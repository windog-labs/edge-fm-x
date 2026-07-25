"""Deterministic passive interpreter for Invocation IR v0.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from vlaforge.analysis.verifier import verify
from vlaforge.interpreter.cache import ExactCache
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
from vlaforge.ir.program import Block, Invocation, Module, Operation
from vlaforge.ir.serializer import io_schema_digest, module_digest


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
    committed_outputs: CommittedOutputGroup
    trace: Trace
    state: dict[str, list[dict[str, object]]]


class Interpreter:
    """A caller-driven Session: BindInput, Run, then ReadOutput."""

    def __init__(
        self,
        module: Module,
        *,
        regions: Mapping[str, RegionCallable],
        validators: Mapping[str, ValidatorCallable],
        initial_state: Mapping[str, object] | None = None,
        expected_schema_digest: str | None = None,
    ):
        verify(module)
        self.module = module
        self.regions = dict(regions)
        self.validators = dict(validators)
        missing_regions = {
            region.name for region in module.regions
        } - set(self.regions)
        if missing_regions:
            raise InterpreterError(
                f"missing TensorRegion implementations: {sorted(missing_regions)}"
            )
        self.schema_digest = io_schema_digest(module)
        self.model_digest = module_digest(module)
        if (
            expected_schema_digest is not None
            and expected_schema_digest != self.schema_digest
        ):
            raise InterpreterError(
                "input/output schema digest mismatch: "
                f"expected={expected_schema_digest} actual={self.schema_digest}"
            )
        self.state_store = StateStore(module, initial_values=initial_state)
        self.trace = Trace()
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
                    raise InterpreterError(
                        "run inputs must be InputBinding values"
                    )
                self.bind_input(name, binding.value, binding.stamp)
        invocation = self.module.invocation(invocation_name)
        try:
            resolved = self._resolve_bindings()
        finally:
            self._bindings.clear()
        self._active_revisions = tuple(
            sorted((name, value[1]) for name, value in resolved.items())
        )
        self._active_snapshots = {}
        env: dict[str, object] = {}
        self._active_transaction = None
        start_index = self._run_index
        try:
            self._execute_block(
                invocation,
                invocation.body,
                env,
                resolved,
                start_index,
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
                    start_index,
                    Operation("vla.txn.abort"),
                    "transaction_abort",
                    {
                        "transaction_id": transaction.id,
                        "reason": "execution_error",
                    },
                )
            self._active_transaction = None
            if isinstance(error, InterpreterError):
                raise
            raise InterpreterError(str(error)) from error
        finally:
            self._run_index += 1

        if len(returns) != 1 or not isinstance(
            returns[0], CommittedOutputGroup
        ):
            raise InterpreterError(
                f"invocation @{invocation.name} did not return committed "
                "output group"
            )
        committed = returns[0]
        if self._last_outputs != committed:
            raise InterpreterError(
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
            raise InterpreterError("no committed output is available")
        port = (
            self.module.outputs[output_id]
            if isinstance(output_id, int)
            and 0 <= output_id < len(self.module.outputs)
            else self.module.output(str(output_id))
        )
        try:
            return self._last_outputs.output(port.name)
        except KeyError as error:
            raise InterpreterError(
                f"latest output group does not contain @{port.name}"
            ) from error

    def read_output_group(self) -> CommittedOutputGroup:
        if self._last_outputs is None:
            raise InterpreterError("no committed output group is available")
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
                    raise InterpreterError(
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
                raise InterpreterError(str(error)) from error
            result[port.name] = (value, revision, binding.stamp.timestamp_ns)
        return result

    def _execute_block(
        self,
        invocation: Invocation,
        block: Block,
        env: dict[str, object],
        inputs: Mapping[str, tuple[object, int, int | None]],
        run_index: int,
    ) -> None:
        for operation in block.operations:
            results = self._execute_operation(
                invocation,
                operation,
                env,
                inputs,
                run_index,
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
        invocation: Invocation,
        operation: Operation,
        env: dict[str, object],
        inputs: Mapping[str, tuple[object, int, int | None]],
        run_index: int,
    ) -> tuple[object, ...]:
        opcode = operation.opcode
        operands = tuple(env[name] for name in operation.operands)

        if opcode == "vla.input.read":
            name = str(operation.attributes["input"])
            value, revision, timestamp_ns = inputs[name]
            self._record(
                invocation,
                run_index,
                operation,
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
                operation,
                "transaction_begin",
                {"transaction_id": transaction.id},
            )
            return (transaction,)

        if opcode == "vla.state.read_latest":
            transaction = _expect(operands[0], Transaction, opcode)
            if transaction is not self._active_transaction:
                raise InterpreterError("state read uses inactive transaction")
            state_name = str(operation.attributes["state"])
            snapshot = self.state_store.read_latest(state_name)
            self._active_snapshots[state_name] = (
                snapshot.episode,
                snapshot.version,
            )
            self._record(
                invocation,
                run_index,
                operation,
                "state_read",
                snapshot,
            )
            return (snapshot,)

        if opcode == "vla.snapshot.value":
            snapshot = _expect(operands[0], SnapshotValue, opcode)
            return (snapshot.value,)

        if opcode == "vla.invoke":
            region_name = str(operation.attributes["region"])
            call_args = tuple(_unwrap(value) for value in operands)
            region = self.module.region(region_name)
            cache_key = None
            cache_hit = False
            executed = True
            result: object
            if bool(region.metadata.get("memoize", False)):
                cache_key = self._cache_key(region_name)
                lookup = self.cache.lookup(cache_key)
                cache_hit = lookup.hit
                if lookup.hit:
                    result = lookup.value
                    executed = False
                else:
                    result = self.regions[region_name](*call_args)
                    self.cache.store(cache_key, result)
                self._record(
                    invocation,
                    run_index,
                    operation,
                    "cache",
                    {
                        "region": region_name,
                        "hit": cache_hit,
                        "input_revisions": self._active_revisions,
                        "state_snapshots": tuple(
                            sorted(self._active_snapshots.items())
                        ),
                    },
                )
            else:
                result = self.regions[region_name](*call_args)
            expected = len(operation.results)
            if expected == 0:
                values: tuple[object, ...] = ()
            elif expected == 1:
                values = (result,)
            elif not isinstance(result, tuple | list):
                raise InterpreterError(
                    f"region @{region_name} must return {expected} values"
                )
            else:
                values = tuple(result)
            if executed:
                self._record(
                    invocation,
                    run_index,
                    operation,
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
                    self._execute_block(
                        invocation,
                        body,
                        nested,
                        inputs,
                        run_index,
                    )
                except _YieldSignal as signal:
                    if len(signal.values) != 1:
                        raise InterpreterError(
                            "vla.for body must yield one iter value"
                        )
                    current = signal.values[0]
                else:
                    raise InterpreterError("vla.for body did not yield")
            return (current,)

        if opcode == "vla.if":
            block = operation.regions[0 if bool(operands[0]) else 1]
            nested = dict(env)
            try:
                self._execute_block(
                    invocation,
                    block,
                    nested,
                    inputs,
                    run_index,
                )
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
            pending = self.state_store.stage(
                transaction,
                state_name,
                _unwrap(operands[1]),
            )
            self._record(
                invocation,
                run_index,
                operation,
                "state_stage",
                pending,
            )
            return (pending,)

        if opcode == "vla.validate":
            contract = str(operation.attributes["contract"])
            if contract not in self.validators:
                raise InterpreterError(f"missing validator @{contract}")
            valid = bool(self.validators[contract](_unwrap(operands[0])))
            self._record(
                invocation,
                run_index,
                operation,
                "validation",
                {"contract": contract, "valid": valid},
            )
            return (valid,)

        if opcode == "vla.output.create":
            pending = PendingOutput(
                str(operation.attributes["output"]),
                _unwrap(operands[0]),
            )
            self._record(
                invocation,
                run_index,
                operation,
                "output_pending",
                pending,
            )
            return (pending,)

        if opcode == "vla.output.group":
            group = PendingOutputGroup(
                str(operation.attributes["group"]),
                tuple(
                    _expect(item, PendingOutput, opcode)
                    for item in operands
                ),
            )
            self._record(
                invocation,
                run_index,
                operation,
                "output_group_pending",
                group,
            )
            return (group,)

        if opcode == "vla.txn.commit":
            transaction = _expect(operands[0], Transaction, opcode)
            pending = _expect(operands[1], PendingOutputGroup, opcode)
            if not bool(operands[2]):
                self.state_store.abort(transaction)
                self._active_transaction = None
                self._record(
                    invocation,
                    run_index,
                    operation,
                    "transaction_abort",
                    {
                        "transaction_id": transaction.id,
                        "reason": "validation",
                    },
                )
                raise InterpreterError(
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
                operation,
                "transaction_commit",
                {
                    "transaction_id": transaction.id,
                    "states": committed_states,
                    "output": committed,
                },
            )
            return (committed,)

        if opcode == "vla.txn.abort":
            transaction = _expect(operands[0], Transaction, opcode)
            self.state_store.abort(transaction)
            self._active_transaction = None
            self._record(
                invocation,
                run_index,
                operation,
                "transaction_abort",
                {
                    "transaction_id": transaction.id,
                    "reason": operation.attributes.get("reason", ""),
                },
            )
            return ()

        raise InterpreterError(f"unsupported operation {opcode}")

    def _cache_key(self, region_name: str) -> tuple[object, ...]:
        return (
            self.model_digest,
            region_name,
            self.state_store.episode,
            self._active_revisions,
            tuple(sorted(self._active_snapshots.items())),
        )

    def _record(
        self,
        invocation: Invocation,
        run_index: int,
        operation: Operation,
        kind: str,
        data: object,
    ) -> None:
        self.trace.record(
            kind,
            invocation.name,
            run_index,
            operation.opcode,
            data,
        )


def _unwrap(value: object) -> object:
    if isinstance(value, SnapshotValue):
        return value.value
    return value


def _expect(value: object, expected: type, opcode: str):
    if not isinstance(value, expected):
        raise InterpreterError(
            f"{opcode} expected {expected.__name__}, "
            f"got {type(value).__name__}"
        )
    return value
