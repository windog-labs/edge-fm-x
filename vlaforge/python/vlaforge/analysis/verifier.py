"""Whole-program verifier for the VLAForge Python reference IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from vlaforge.ir.attrs import Effect, EpochExpr
from vlaforge.ir.program import Block, Module, Operation, Policy, Value
from vlaforge.ir.types import (
    ActionType,
    CommittedActionType,
    EpochType,
    FutureType,
    PendingType,
    ScalarType,
    SnapshotType,
    TransactionType,
)
from vlaforge.ir.versioning import require_supported


ALLOWED_OPS = {
    "vla.sample_input",
    "vla.txn.begin",
    "vla.state.read",
    "vla.snapshot.value",
    "vla.invoke",
    "vla.for",
    "vla.while",
    "vla.if",
    "vla.yield",
    "vla.return",
    "vla.state.stage_write",
    "vla.validate",
    "vla.action.create",
    "vla.txn.commit",
    "vla.txn.abort",
    "vla.action.publish",
    "vla.reset",
    "vla.async",
    "vla.await",
}


@dataclass(frozen=True, slots=True)
class Diagnostic:
    rule: str
    message: str
    program: str
    policy: str | None = None
    op: str | None = None
    state: str | None = None
    epoch: str | None = None
    version: str | None = None
    location: str | None = None

    def __str__(self) -> str:
        context = [f"program={self.program}", f"rule={self.rule}"]
        for key in ("policy", "op", "state", "epoch", "version", "location"):
            value = getattr(self, key)
            if value is not None:
                context.append(f"{key}={value}")
        return f"[{', '.join(context)}] {self.message}"


class VerificationError(ValueError):
    def __init__(self, diagnostics: Iterable[Diagnostic]):
        self.diagnostics = tuple(diagnostics)
        super().__init__("\n".join(str(item) for item in self.diagnostics))


class _Verifier:
    def __init__(self, module: Module):
        self.module = module
        self.diagnostics: list[Diagnostic] = []
        self.clocks = {item.name: item for item in module.clocks}
        self.inputs = {item.name: item for item in module.inputs}
        self.states = {item.name: item for item in module.states}
        self.regions = {item.name: item for item in module.regions}

    def error(
        self,
        rule: str,
        message: str,
        *,
        policy: Policy | None = None,
        operation: Operation | None = None,
        state: str | None = None,
        epoch: str | None = None,
        version: str | None = None,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                rule=rule,
                message=message,
                program=self.module.name,
                policy=None if policy is None else policy.name,
                op=None if operation is None else operation.opcode,
                state=state,
                epoch=epoch,
                version=version,
                location=None if operation is None else operation.location,
            )
        )

    def run(self) -> None:
        try:
            require_supported(self.module.schema_version)
        except ValueError as exc:
            self.error("schema.version", str(exc))

        self._verify_declarations()
        for policy in self.module.policies:
            self._verify_policy(policy)

    def _verify_declarations(self) -> None:
        for stream in self.module.inputs:
            if stream.clock not in self.clocks:
                self.error(
                    "input.clock",
                    f"input @{stream.name} references unknown clock @{stream.clock}",
                )
        for state in self.module.states:
            if state.version_clock not in self.clocks:
                self.error(
                    "state.clock",
                    f"state @{state.name} references unknown clock "
                    f"@{state.version_clock}",
                    state=state.name,
                )
            if (
                state.freshness is not None
                and state.freshness.max_versions is not None
                and state.retention <= state.freshness.max_versions
            ):
                self.error(
                    "state.retention",
                    f"retention={state.retention} cannot satisfy "
                    f"max_versions={state.freshness.max_versions}; need at least "
                    f"{state.freshness.max_versions + 1}",
                    state=state.name,
                    version=f"retention:{state.retention}",
                )
        for region in self.module.regions:
            if not region.pure:
                effects = ",".join(effect.value for effect in region.effects)
                self.error(
                    "region.hidden_effect",
                    f"TensorRegion @{region.name} must be pure; explicit state/RNG "
                    f"must be inputs and outputs, found effects [{effects}]",
                )
            if region.effects != (Effect.PURE,):
                self.error(
                    "region.effect_contract",
                    f"TensorRegion @{region.name} has unsupported effects",
                )

    def _verify_policy(self, policy: Policy) -> None:
        if policy.clock not in self.clocks:
            self.error(
                "policy.clock",
                f"policy @{policy.name} references unknown clock @{policy.clock}",
                policy=policy,
            )
        definitions = {value.name: value for value in policy.inputs}
        self._verify_block(
            policy,
            policy.body,
            definitions,
            awaited=set(),
            staged={},
            active_async={},
        )
        outcomes = self._commit_outcomes(policy.body)
        for count, aborted in outcomes:
            if aborted:
                continue
            if count == 0:
                self.error(
                    "commit.zero",
                    "successful policy path reaches exit without exactly one commit",
                    policy=policy,
                    version="commits:0",
                )
            elif count > 1:
                self.error(
                    "commit.double",
                    f"successful policy path contains {count} commits",
                    policy=policy,
                    version=f"commits:{count}",
                )

    def _verify_block(
        self,
        policy: Policy,
        block: Block,
        inherited: Mapping[str, Value],
        *,
        awaited: set[str],
        staged: dict[str, set[str]],
        active_async: dict[str, tuple[set[str], set[str]]],
    ) -> dict[str, Value]:
        definitions = dict(inherited)
        for argument in block.arguments:
            if argument.name in definitions:
                self.error(
                    "ssa.duplicate",
                    f"block argument %{argument.name} shadows an existing value",
                    policy=policy,
                )
            definitions[argument.name] = argument

        for index, operation in enumerate(block.operations):
            if operation.opcode not in ALLOWED_OPS:
                self.error(
                    "op.unknown",
                    f"unknown core operation {operation.opcode!r}",
                    policy=policy,
                    operation=operation,
                )
            for operand in operation.operands:
                if operand not in definitions:
                    self.error(
                        "ssa.read_before_definition",
                        f"operand %{operand} is not defined before use at op #{index}",
                        policy=policy,
                        operation=operation,
                        version=operand,
                    )
            for result in operation.results:
                if result.name in definitions:
                    self.error(
                        "ssa.duplicate",
                        f"SSA value %{result.name} is defined more than once",
                        policy=policy,
                        operation=operation,
                        version=result.name,
                    )

            self._verify_operation(
                policy,
                operation,
                definitions,
                awaited=awaited,
                staged=staged,
                active_async=active_async,
            )

            for result in operation.results:
                definitions[result.name] = result

            for region in operation.regions:
                self._verify_block(
                    policy,
                    region,
                    definitions,
                    awaited=set(awaited),
                    staged={name: set(items) for name, items in staged.items()},
                    active_async=dict(active_async),
                )
        return definitions

    def _operand(
        self,
        definitions: Mapping[str, Value],
        operation: Operation,
        index: int,
    ) -> Value | None:
        if index >= len(operation.operands):
            return None
        return definitions.get(operation.operands[index])

    def _verify_operation(
        self,
        policy: Policy,
        operation: Operation,
        definitions: Mapping[str, Value],
        *,
        awaited: set[str],
        staged: dict[str, set[str]],
        active_async: dict[str, tuple[set[str], set[str]]],
    ) -> None:
        if operation.opcode == "vla.sample_input":
            stream_name = str(operation.attributes.get("stream", ""))
            stream = self.inputs.get(stream_name)
            if stream is None:
                self.error(
                    "input.unknown",
                    f"sample references unknown input @{stream_name}",
                    policy=policy,
                    operation=operation,
                )
                return
            if len(operation.results) != 2:
                self.error(
                    "input.results",
                    "sample_input must return payload and epoch",
                    policy=policy,
                    operation=operation,
                )
            elif operation.results[0].type != stream.payload or operation.results[
                1
            ].type != EpochType(stream.clock):
                self.error(
                    "input.type",
                    f"sample results do not match input @{stream_name}",
                    policy=policy,
                    operation=operation,
                    epoch=stream.clock,
                )
            max_age = operation.attributes.get("max_age_ns")
            if max_age is not None and int(max_age) < 0:
                self.error(
                    "freshness.invalid",
                    "max_age_ns must be non-negative",
                    policy=policy,
                    operation=operation,
                    epoch=f"max_age_ns:{max_age}",
                )

        elif operation.opcode == "vla.txn.begin":
            epoch = self._operand(definitions, operation, 0)
            if epoch is not None and not isinstance(epoch.type, EpochType):
                self.error(
                    "txn.epoch_type",
                    "transaction must begin from an epoch value",
                    policy=policy,
                    operation=operation,
                )
            if len(operation.results) != 1 or not isinstance(
                operation.results[0].type, TransactionType
            ):
                self.error(
                    "txn.result_type",
                    "transaction begin must return TransactionType",
                    policy=policy,
                    operation=operation,
                )

        elif operation.opcode == "vla.state.read":
            state_name = str(operation.attributes.get("state", ""))
            state = self.states.get(state_name)
            version = str(operation.attributes.get("version", "latest"))
            epoch_data = operation.attributes.get("epoch", {})
            try:
                epoch = EpochExpr.from_dict(epoch_data)
            except Exception as exc:
                self.error(
                    "state.epoch",
                    f"invalid state epoch expression: {exc}",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    version=version,
                )
                return
            if state is None:
                self.error(
                    "state.unknown",
                    f"read references unknown state @{state_name}",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                    version=version,
                )
                return
            if epoch.clock not in {None, state.version_clock}:
                self.error(
                    "state.wrong_version_clock",
                    f"state @{state_name} is versioned by @{state.version_clock}, "
                    f"not @{epoch.clock}",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                    version=version,
                )
            txn = self._operand(definitions, operation, 0)
            if txn is not None and not isinstance(txn.type, TransactionType):
                self.error(
                    "state.read_transaction",
                    "state read requires a transaction operand",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                    version=version,
                )
            if (
                len(operation.results) != 1
                or operation.results[0].type
                != SnapshotType(state_name, state.payload)
            ):
                self.error(
                    "state.snapshot_type",
                    f"state read result must be snapshot<{state_name}>",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                    version=version,
                )

        elif operation.opcode == "vla.invoke":
            region_name = str(operation.attributes.get("region", ""))
            region = self.regions.get(region_name)
            if region is None:
                self.error(
                    "region.unknown",
                    f"invoke references unknown TensorRegion @{region_name}",
                    policy=policy,
                    operation=operation,
                )
                return
            operand_types = tuple(
                definitions[name].type
                for name in operation.operands
                if name in definitions
            )
            expected = tuple(value.type for value in region.inputs)
            if operand_types != expected:
                self.error(
                    "region.input_types",
                    f"invoke @{region_name} expects {expected}, got {operand_types}",
                    policy=policy,
                    operation=operation,
                )
            result_types = tuple(value.type for value in operation.results)
            if result_types != region.outputs:
                self.error(
                    "region.output_types",
                    f"invoke @{region_name} returns {region.outputs}, "
                    f"op declares {result_types}",
                    policy=policy,
                    operation=operation,
                )

        elif operation.opcode == "vla.snapshot.value":
            snapshot = self._operand(definitions, operation, 0)
            if snapshot is not None and not isinstance(snapshot.type, SnapshotType):
                self.error(
                    "state.snapshot_unwrap",
                    "snapshot.value requires SnapshotType",
                    policy=policy,
                    operation=operation,
                )
            elif (
                snapshot is not None
                and (
                    len(operation.results) != 1
                    or operation.results[0].type != snapshot.type.payload
                )
            ):
                self.error(
                    "state.snapshot_payload_type",
                    f"snapshot.value must return {snapshot.type.payload}",
                    policy=policy,
                    operation=operation,
                    state=snapshot.type.state,
                )

        elif operation.opcode == "vla.state.stage_write":
            state_name = str(operation.attributes.get("state", ""))
            state = self.states.get(state_name)
            txn = self._operand(definitions, operation, 0)
            value = self._operand(definitions, operation, 1)
            epoch_data = operation.attributes.get("epoch", {})
            try:
                epoch = EpochExpr.from_dict(epoch_data)
            except Exception as exc:
                self.error(
                    "state.epoch",
                    f"invalid staged epoch expression: {exc}",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                )
                return
            if state is None:
                self.error(
                    "state.unknown",
                    f"stage_write references unknown state @{state_name}",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                )
                return
            if epoch.clock != state.version_clock:
                self.error(
                    "state.wrong_version_clock",
                    f"state @{state_name} is versioned by @{state.version_clock}, "
                    f"not @{epoch.clock}",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                )
            if txn is not None and not isinstance(txn.type, TransactionType):
                self.error(
                    "state.write_transaction",
                    "stage_write requires a transaction operand",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                )
            if value is not None and isinstance(value.type, SnapshotType):
                value_type = value.type.payload
            else:
                value_type = None if value is None else value.type
            if value_type != state.payload:
                self.error(
                    "state.write_type",
                    f"state @{state_name} expects {state.payload}, got {value_type}",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                )
            if (
                len(operation.results) != 1
                or operation.results[0].type != PendingType(state_name, state.payload)
            ):
                self.error(
                    "state.pending_type",
                    f"stage_write result must be pending<{state_name}>",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                )
            txn_name = operation.operands[0] if operation.operands else "<missing>"
            if state_name in staged.setdefault(txn_name, set()):
                self.error(
                    "state.double_write",
                    f"state @{state_name} is staged twice in transaction %{txn_name}",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                )
            staged[txn_name].add(state_name)
            if state.authoritative and bool(operation.attributes.get("inplace", False)):
                self.error(
                    "state.authoritative_inplace",
                    f"authoritative state @{state_name} cannot be overwritten "
                    "in-place before action commit",
                    policy=policy,
                    operation=operation,
                    state=state_name,
                    epoch=epoch.clock,
                )

        elif operation.opcode == "vla.validate":
            if len(operation.results) != 1 or operation.results[0].type != ScalarType(
                "bool"
            ):
                self.error(
                    "validate.type",
                    "validator must return bool",
                    policy=policy,
                    operation=operation,
                )

        elif operation.opcode == "vla.action.create":
            if len(operation.operands) >= 2:
                epoch = definitions.get(operation.operands[1])
                if epoch is not None and not isinstance(epoch.type, EpochType):
                    self.error(
                        "action.epoch_type",
                        "action_create epoch operand must have EpochType",
                        policy=policy,
                        operation=operation,
                    )
            if len(operation.results) != 1 or not isinstance(
                operation.results[0].type, ActionType
            ):
                self.error(
                    "action.type",
                    "action_create must return ActionType",
                    policy=policy,
                    operation=operation,
                )

        elif operation.opcode == "vla.await":
            future = self._operand(definitions, operation, 0)
            if future is not None and not isinstance(future.type, FutureType):
                self.error(
                    "async.await_type",
                    "await operand must be FutureType",
                    policy=policy,
                    operation=operation,
                )
            if operation.operands:
                awaited.add(operation.operands[0])
                active_async.pop(operation.operands[0], None)

        elif operation.opcode == "vla.async":
            reads = {str(item) for item in operation.attributes.get("reads", ())}
            writes = {str(item) for item in operation.attributes.get("writes", ())}
            for future_name, (other_reads, other_writes) in active_async.items():
                conflicts = (writes & (other_reads | other_writes)) | (
                    other_writes & reads
                )
                if conflicts:
                    self.error(
                        "async.state_race",
                        f"async task conflicts with un-awaited %{future_name} on "
                        f"states {sorted(conflicts)}",
                        policy=policy,
                        operation=operation,
                        state=",".join(sorted(conflicts)),
                    )
            if operation.results:
                active_async[operation.results[0].name] = (reads, writes)

        elif operation.opcode == "vla.txn.commit":
            txn = self._operand(definitions, operation, 0)
            action = self._operand(definitions, operation, 1)
            condition = self._operand(definitions, operation, 2)
            if txn is not None and not isinstance(txn.type, TransactionType):
                self.error(
                    "commit.transaction_type",
                    "commit requires TransactionType",
                    policy=policy,
                    operation=operation,
                )
            if action is not None and not isinstance(action.type, ActionType):
                self.error(
                    "commit.action_type",
                    "commit requires an uncommitted ActionType",
                    policy=policy,
                    operation=operation,
                )
            if condition is not None:
                if condition.type != ScalarType("bool"):
                    self.error(
                        "commit.condition_type",
                        "commit condition must be bool",
                        policy=policy,
                        operation=operation,
                    )
                producer = _find_producer(policy.body, condition.name)
                if producer is None or producer.opcode != "vla.validate":
                    self.error(
                        "commit.validator_dominance",
                        f"commit condition %{condition.name} is not produced by a "
                        "dominating validator",
                        policy=policy,
                        operation=operation,
                    )
            for required in operation.attributes.get("required_futures", ()):
                required_name = str(required)
                if required_name not in awaited:
                    self.error(
                        "commit.future_not_awaited",
                        f"required future %{required_name} is not awaited before commit",
                        policy=policy,
                        operation=operation,
                        version=required_name,
                    )
            if len(operation.results) != 1 or not isinstance(
                operation.results[0].type, CommittedActionType
            ):
                self.error(
                    "commit.result_type",
                    "commit must return CommittedActionType",
                    policy=policy,
                    operation=operation,
                )

        elif operation.opcode == "vla.action.publish":
            action = self._operand(definitions, operation, 0)
            if action is not None and not isinstance(
                action.type, CommittedActionType
            ):
                self.error(
                    "action.publish_before_commit",
                    "only CommittedActionType may be published",
                    policy=policy,
                    operation=operation,
                )

        elif operation.opcode in {"vla.return", "vla.yield"}:
            for operand in operation.operands:
                value = definitions.get(operand)
                if value is not None and isinstance(value.type, PendingType):
                    self.error(
                        "state.pending_escape",
                        f"pending state %{operand} escapes transaction scope",
                        policy=policy,
                        operation=operation,
                        state=value.type.state,
                        version=operand,
                    )

        elif operation.opcode == "vla.reset":
            for state_name in operation.attributes.get("states", ()):
                if state_name not in self.states:
                    self.error(
                        "reset.unknown_state",
                        f"reset references unknown state @{state_name}",
                        policy=policy,
                        operation=operation,
                        state=str(state_name),
                    )

    def _commit_outcomes(self, block: Block) -> set[tuple[int, bool]]:
        outcomes: set[tuple[int, bool]] = {(0, False)}
        for operation in block.operations:
            next_outcomes: set[tuple[int, bool]] = set()
            for count, terminated in outcomes:
                if terminated:
                    next_outcomes.add((count, terminated))
                    continue
                if operation.opcode == "vla.txn.commit":
                    next_outcomes.add((count + 1, False))
                elif operation.opcode == "vla.txn.abort":
                    next_outcomes.add((count, True))
                elif operation.opcode == "vla.if" and len(operation.regions) == 2:
                    for branch in operation.regions:
                        for branch_count, branch_aborted in self._commit_outcomes(
                            branch
                        ):
                            next_outcomes.add(
                                (count + branch_count, branch_aborted)
                            )
                else:
                    nested_commits = sum(
                        1
                        for region in operation.regions
                        for nested in _walk_operations(region)
                        if nested.opcode == "vla.txn.commit"
                    )
                    next_outcomes.add((count + nested_commits, False))
            outcomes = next_outcomes
        return outcomes


def _walk_operations(block: Block) -> Iterable[Operation]:
    for operation in block.operations:
        yield operation
        for region in operation.regions:
            yield from _walk_operations(region)


def _find_producer(block: Block, value_name: str) -> Operation | None:
    for operation in _walk_operations(block):
        if any(result.name == value_name for result in operation.results):
            return operation
    return None


def verify(module: Module, *, raise_on_error: bool = True) -> tuple[Diagnostic, ...]:
    verifier = _Verifier(module)
    verifier.run()
    diagnostics = tuple(verifier.diagnostics)
    if diagnostics and raise_on_error:
        raise VerificationError(diagnostics)
    return diagnostics
