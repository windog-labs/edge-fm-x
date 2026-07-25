"""Whole-program verifier for Invocation IR v0.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from vlaforge.ir.attrs import Effect
from vlaforge.ir.program import Block, Invocation, Module, Operation, Value
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
from vlaforge.ir.versioning import require_supported


ALLOWED_OPS = {
    "vla.input.read",
    "vla.txn.begin",
    "vla.state.read_latest",
    "vla.snapshot.value",
    "vla.invoke",
    "vla.for",
    "vla.if",
    "vla.yield",
    "vla.return",
    "vla.state.stage_write",
    "vla.validate",
    "vla.output.create",
    "vla.output.group",
    "vla.txn.commit",
    "vla.txn.abort",
}


@dataclass(frozen=True, slots=True)
class Diagnostic:
    rule: str
    message: str
    program: str
    invocation: str | None = None
    op: str | None = None
    state: str | None = None
    value: str | None = None
    location: str | None = None

    def __str__(self) -> str:
        context = [f"program={self.program}", f"rule={self.rule}"]
        for key in ("invocation", "op", "state", "value", "location"):
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
        self.inputs = {item.name: item for item in module.inputs}
        self.outputs = {item.name: item for item in module.outputs}
        self.states = {item.name: item for item in module.states}
        self.regions = {item.name: item for item in module.regions}

    def error(
        self,
        rule: str,
        message: str,
        *,
        invocation: Invocation | None = None,
        operation: Operation | None = None,
        state: str | None = None,
        value: str | None = None,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                rule=rule,
                message=message,
                program=self.module.name,
                invocation=(
                    None if invocation is None else invocation.name
                ),
                op=None if operation is None else operation.opcode,
                state=state,
                value=value,
                location=None if operation is None else operation.location,
            )
        )

    def run(self) -> None:
        try:
            require_supported(self.module.schema_version)
        except ValueError as error:
            self.error("schema.version", str(error))
        if not self.module.invocations:
            self.error("invocation.missing", "module has no invocation")
        for region in self.module.regions:
            illegal = set(region.effects) - {Effect.PURE}
            if illegal or not region.pure:
                self.error(
                    "region.effect",
                    f"TensorRegion @{region.name} must be pure",
                )
        for invocation in self.module.invocations:
            self._verify_invocation(invocation)

    def _verify_invocation(self, invocation: Invocation) -> None:
        definitions: dict[str, Value] = {}
        validators: set[str] = set()
        self._verify_block(
            invocation.body,
            invocation,
            definitions,
            validators,
            nested=False,
        )
        outcomes = self._commit_outcomes(invocation.body)
        for count, aborted in outcomes:
            if aborted:
                continue
            if count == 0:
                self.error(
                    "commit.zero",
                    "successful invocation path reaches exit without commit",
                    invocation=invocation,
                )
            elif count > 1:
                self.error(
                    "commit.double",
                    f"successful invocation path contains {count} commits",
                    invocation=invocation,
                )

    def _verify_block(
        self,
        block: Block,
        invocation: Invocation,
        inherited: Mapping[str, Value],
        inherited_validators: set[str],
        *,
        nested: bool,
    ) -> dict[str, Value]:
        definitions = dict(inherited)
        validators = set(inherited_validators)
        for argument in block.arguments:
            if argument.name in definitions:
                self.error(
                    "ssa.duplicate",
                    f"block argument %{argument.name} is already defined",
                    invocation=invocation,
                    value=argument.name,
                )
            definitions[argument.name] = argument

        for operation in block.operations:
            if operation.opcode not in ALLOWED_OPS:
                self.error(
                    "op.unknown",
                    f"unknown operation {operation.opcode}",
                    invocation=invocation,
                    operation=operation,
                )
                continue
            for operand in operation.operands:
                if operand not in definitions:
                    self.error(
                        "ssa.undefined",
                        f"operand %{operand} is not defined",
                        invocation=invocation,
                        operation=operation,
                        value=operand,
                    )
            for result in operation.results:
                if result.name in definitions:
                    self.error(
                        "ssa.duplicate",
                        f"result %{result.name} is already defined",
                        invocation=invocation,
                        operation=operation,
                        value=result.name,
                    )
            self._verify_operation(
                operation,
                invocation,
                definitions,
                validators,
                nested=nested,
            )
            for result in operation.results:
                definitions[result.name] = result
            if operation.opcode == "vla.validate" and operation.results:
                validators.add(operation.results[0].name)
        return definitions

    def _verify_operation(
        self,
        operation: Operation,
        invocation: Invocation,
        definitions: Mapping[str, Value],
        validators: set[str],
        *,
        nested: bool,
    ) -> None:
        opcode = operation.opcode

        if opcode == "vla.input.read":
            name = str(operation.attributes.get("input", ""))
            port = self.inputs.get(name)
            if port is None:
                self.error(
                    "input.unknown",
                    f"unknown input @{name}",
                    invocation=invocation,
                    operation=operation,
                )
            if len(operation.results) != 2:
                self.error(
                    "input.results",
                    "input.read requires payload and revision results",
                    invocation=invocation,
                    operation=operation,
                )
            elif port is not None:
                if operation.results[0].type != port.payload:
                    self.error(
                        "input.payload_type",
                        "input payload result type does not match declaration",
                        invocation=invocation,
                        operation=operation,
                    )
                if not isinstance(
                    operation.results[1].type, InputRevisionType
                ):
                    self.error(
                        "input.revision_type",
                        "input revision result requires InputRevisionType",
                        invocation=invocation,
                        operation=operation,
                    )
            if operation.operands:
                self.error(
                    "input.operands",
                    "input.read has no SSA operands",
                    invocation=invocation,
                    operation=operation,
                )
            return

        if opcode == "vla.txn.begin":
            if (
                operation.operands
                or len(operation.results) != 1
                or not isinstance(operation.results[0].type, TransactionType)
            ):
                self.error(
                    "txn.signature",
                    "txn.begin returns one transaction and takes no operand",
                    invocation=invocation,
                    operation=operation,
                )
            return

        if opcode == "vla.state.read_latest":
            state_name = str(operation.attributes.get("state", ""))
            state = self.states.get(state_name)
            transaction = self._type(operation, 0, definitions)
            if not isinstance(transaction, TransactionType):
                self.error(
                    "state.transaction_type",
                    "state.read_latest requires a transaction",
                    invocation=invocation,
                    operation=operation,
                    state=state_name,
                )
            if state is None:
                self.error(
                    "state.unknown",
                    f"unknown state @{state_name}",
                    invocation=invocation,
                    operation=operation,
                    state=state_name,
                )
            if len(operation.results) != 1 or not isinstance(
                operation.results[0].type, SnapshotType
            ):
                self.error(
                    "state.snapshot_type",
                    "state.read_latest must return SnapshotType",
                    invocation=invocation,
                    operation=operation,
                    state=state_name,
                )
            elif state is not None and operation.results[0].type != SnapshotType(
                state_name, state.payload
            ):
                self.error(
                    "state.snapshot_type",
                    "snapshot state/payload does not match declaration",
                    invocation=invocation,
                    operation=operation,
                    state=state_name,
                )
            return

        if opcode == "vla.snapshot.value":
            snapshot = self._type(operation, 0, definitions)
            if not isinstance(snapshot, SnapshotType):
                self.error(
                    "snapshot.operand_type",
                    "snapshot.value requires SnapshotType",
                    invocation=invocation,
                    operation=operation,
                )
            elif (
                len(operation.results) != 1
                or operation.results[0].type != snapshot.payload
            ):
                self.error(
                    "snapshot.result_type",
                    "snapshot.value result must match snapshot payload",
                    invocation=invocation,
                    operation=operation,
                )
            return

        if opcode == "vla.invoke":
            region_name = str(operation.attributes.get("region", ""))
            region = self.regions.get(region_name)
            if region is None:
                self.error(
                    "region.unknown",
                    f"unknown TensorRegion @{region_name}",
                    invocation=invocation,
                    operation=operation,
                )
                return
            operand_types = tuple(
                self._type_at(name, definitions)
                for name in operation.operands
            )
            expected_inputs = tuple(value.type for value in region.inputs)
            if operand_types != expected_inputs:
                self.error(
                    "region.input_type",
                    f"region @{region_name} input signature mismatch",
                    invocation=invocation,
                    operation=operation,
                )
            if tuple(value.type for value in operation.results) != region.outputs:
                self.error(
                    "region.output_type",
                    f"region @{region_name} output signature mismatch",
                    invocation=invocation,
                    operation=operation,
                )
            return

        if opcode == "vla.for":
            self._verify_for(operation, invocation, definitions, validators)
            return

        if opcode == "vla.if":
            self._verify_if(operation, invocation, definitions, validators)
            return

        if opcode == "vla.yield":
            if not nested:
                self.error(
                    "control.yield_scope",
                    "yield is only legal inside a structured region",
                    invocation=invocation,
                    operation=operation,
                )
            return

        if opcode == "vla.return":
            if len(operation.operands) != 1 or not isinstance(
                self._type(operation, 0, definitions),
                CommittedOutputGroupType,
            ):
                self.error(
                    "output.return_type",
                    "invocation must return one committed output group",
                    invocation=invocation,
                    operation=operation,
                )
            return

        if opcode == "vla.state.stage_write":
            state_name = str(operation.attributes.get("state", ""))
            state = self.states.get(state_name)
            transaction = self._type(operation, 0, definitions)
            value_type = self._type(operation, 1, definitions)
            if not isinstance(transaction, TransactionType):
                self.error(
                    "state.transaction_type",
                    "stage_write requires a transaction",
                    invocation=invocation,
                    operation=operation,
                    state=state_name,
                )
            if state is None:
                self.error(
                    "state.unknown",
                    f"unknown state @{state_name}",
                    invocation=invocation,
                    operation=operation,
                    state=state_name,
                )
            elif value_type != state.payload:
                self.error(
                    "state.payload_type",
                    "staged payload does not match StateSlot",
                    invocation=invocation,
                    operation=operation,
                    state=state_name,
                )
            if (
                state is not None
                and (
                    len(operation.results) != 1
                    or operation.results[0].type
                    != PendingType(state_name, state.payload)
                )
            ):
                self.error(
                    "state.pending_type",
                    "stage_write result must be PendingType",
                    invocation=invocation,
                    operation=operation,
                    state=state_name,
                )
            return

        if opcode == "vla.validate":
            if (
                len(operation.operands) != 1
                or len(operation.results) != 1
                or operation.results[0].type != ScalarType("bool")
                or not str(operation.attributes.get("contract", ""))
            ):
                self.error(
                    "validation.signature",
                    "validate requires value, contract, and bool result",
                    invocation=invocation,
                    operation=operation,
                )
            return

        if opcode == "vla.output.create":
            value_type = self._type(operation, 0, definitions)
            output_name = str(operation.attributes.get("output", ""))
            output = self.outputs.get(output_name)
            if output is None:
                self.error(
                    "output.unknown",
                    f"unknown output @{output_name}",
                    invocation=invocation,
                    operation=operation,
                )
            if (
                value_type is None
                or output is None
                or value_type != output.payload
                or len(operation.results) != 1
                or operation.results[0].type
                != PendingOutputType(output_name, value_type)
            ):
                self.error(
                    "output.pending_type",
                    "output.create must return PendingOutputType<payload>",
                    invocation=invocation,
                    operation=operation,
                )
            return

        if opcode == "vla.output.group":
            group_name = str(operation.attributes.get("group", ""))
            pending_types = tuple(
                self._type_at(name, definitions)
                for name in operation.operands
            )
            if (
                not group_name
                or not pending_types
                or any(
                    not isinstance(item, PendingOutputType)
                    for item in pending_types
                )
                or len(operation.results) != 1
                or operation.results[0].type
                != PendingOutputGroupType(group_name, pending_types)
            ):
                self.error(
                    "output.group_type",
                    "output.group requires named pending outputs and "
                    "matching PendingOutputGroupType",
                    invocation=invocation,
                    operation=operation,
                )
                return
            output_names = [item.output for item in pending_types]
            if len(output_names) != len(set(output_names)):
                self.error(
                    "output.group_duplicate",
                    "output group contains duplicate named outputs",
                    invocation=invocation,
                    operation=operation,
                )
            for item in pending_types:
                port = self.outputs.get(item.output)
                if port is None or port.group != group_name:
                    self.error(
                        "output.group_membership",
                        f"output @{item.output} is not declared in "
                        f"group @{group_name}",
                        invocation=invocation,
                        operation=operation,
                    )
            return

        if opcode == "vla.txn.commit":
            transaction = self._type(operation, 0, definitions)
            pending = self._type(operation, 1, definitions)
            condition_name = (
                operation.operands[2]
                if len(operation.operands) > 2
                else ""
            )
            condition = self._type(operation, 2, definitions)
            if not isinstance(transaction, TransactionType):
                self.error(
                    "commit.transaction_type",
                    "commit requires TransactionType",
                    invocation=invocation,
                    operation=operation,
                )
            if not isinstance(pending, PendingOutputGroupType):
                self.error(
                    "commit.output_type",
                    "commit requires PendingOutputGroupType",
                    invocation=invocation,
                    operation=operation,
                )
            if condition != ScalarType("bool"):
                self.error(
                    "commit.condition_type",
                    "commit condition must be bool",
                    invocation=invocation,
                    operation=operation,
                )
            if condition_name not in validators:
                self.error(
                    "commit.validator_dominance",
                    "commit condition must be produced by dominating validate",
                    invocation=invocation,
                    operation=operation,
                    value=condition_name,
                )
            if (
                isinstance(pending, PendingOutputGroupType)
                and (
                    len(operation.results) != 1
                    or operation.results[0].type
                    != CommittedOutputGroupType(
                        pending.group,
                        pending.outputs,
                    )
                )
            ):
                self.error(
                    "commit.result_type",
                    "commit result must be CommittedOutputGroupType",
                    invocation=invocation,
                    operation=operation,
                )
            return

        if opcode == "vla.txn.abort":
            if not isinstance(
                self._type(operation, 0, definitions), TransactionType
            ):
                self.error(
                    "txn.abort_type",
                    "txn.abort requires TransactionType",
                    invocation=invocation,
                    operation=operation,
                )

    def _verify_for(
        self,
        operation: Operation,
        invocation: Invocation,
        definitions: Mapping[str, Value],
        validators: set[str],
    ) -> None:
        if len(operation.regions) != 1 or len(operation.results) != 1:
            self.error(
                "control.for_shape",
                "vla.for requires one body and one carried result",
                invocation=invocation,
                operation=operation,
            )
            return
        lower = int(operation.attributes.get("lower", 0))
        upper = int(operation.attributes.get("upper", 0))
        step = int(operation.attributes.get("step", 0))
        if step <= 0 or upper <= lower:
            self.error(
                "control.for_bound",
                "vla.for requires a finite positive iteration range",
                invocation=invocation,
                operation=operation,
            )
        body = operation.regions[0]
        if len(body.arguments) != 2:
            self.error(
                "control.for_args",
                "vla.for body requires induction and carry arguments",
                invocation=invocation,
                operation=operation,
            )
            return
        initial = self._type(operation, 0, definitions)
        if (
            initial != body.arguments[1].type
            or operation.results[0].type != body.arguments[1].type
        ):
            self.error(
                "control.for_carry",
                "vla.for initial, carry, and result types must match",
                invocation=invocation,
                operation=operation,
            )
        nested = self._verify_block(
            body,
            invocation,
            definitions,
            validators,
            nested=True,
        )
        if (
            not body.operations
            or body.operations[-1].opcode != "vla.yield"
            or len(body.operations[-1].operands) != 1
            or self._type_at(body.operations[-1].operands[0], nested)
            != operation.results[0].type
        ):
            self.error(
                "control.for_yield",
                "vla.for body must yield one carried value",
                invocation=invocation,
                operation=operation,
            )

    def _verify_if(
        self,
        operation: Operation,
        invocation: Invocation,
        definitions: Mapping[str, Value],
        validators: set[str],
    ) -> None:
        if self._type(operation, 0, definitions) != ScalarType("bool"):
            self.error(
                "control.if_condition",
                "vla.if condition must be bool",
                invocation=invocation,
                operation=operation,
            )
        if len(operation.regions) != 2:
            self.error(
                "control.if_regions",
                "vla.if requires then and else regions",
                invocation=invocation,
                operation=operation,
            )
            return
        expected = tuple(value.type for value in operation.results)
        for branch in operation.regions:
            nested = self._verify_block(
                branch,
                invocation,
                definitions,
                validators,
                nested=True,
            )
            if not branch.operations or branch.operations[-1].opcode != "vla.yield":
                self.error(
                    "control.if_yield",
                    "every vla.if branch must end in yield",
                    invocation=invocation,
                    operation=operation,
                )
                continue
            actual = tuple(
                self._type_at(name, nested)
                for name in branch.operations[-1].operands
            )
            if actual != expected:
                self.error(
                    "control.if_yield",
                    "branch yield types must match vla.if results",
                    invocation=invocation,
                    operation=operation,
                )

    @staticmethod
    def _type(
        operation: Operation,
        index: int,
        definitions: Mapping[str, Value],
    ):
        if index >= len(operation.operands):
            return None
        value = definitions.get(operation.operands[index])
        return None if value is None else value.type

    @staticmethod
    def _type_at(name: str, definitions: Mapping[str, Value]):
        value = definitions.get(name)
        return None if value is None else value.type

    def _commit_outcomes(self, block: Block) -> set[tuple[int, bool]]:
        outcomes: set[tuple[int, bool]] = {(0, False)}
        for operation in block.operations:
            next_outcomes: set[tuple[int, bool]] = set()
            for count, aborted in outcomes:
                if aborted:
                    next_outcomes.add((count, True))
                elif operation.opcode == "vla.txn.commit":
                    next_outcomes.add((count + 1, False))
                elif operation.opcode == "vla.txn.abort":
                    next_outcomes.add((count, True))
                elif operation.opcode == "vla.if":
                    for branch in operation.regions:
                        for branch_count, branch_aborted in self._commit_outcomes(
                            branch
                        ):
                            next_outcomes.add(
                                (
                                    count + branch_count,
                                    branch_aborted,
                                )
                            )
                else:
                    next_outcomes.add((count, False))
            outcomes = next_outcomes
        return outcomes


def verify(
    module: Module,
    *,
    raise_on_error: bool = True,
) -> tuple[Diagnostic, ...]:
    verifier = _Verifier(module)
    verifier.run()
    diagnostics = tuple(verifier.diagnostics)
    if diagnostics and raise_on_error:
        raise VerificationError(diagnostics)
    return diagnostics
