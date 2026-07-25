"""Typed builders for the small Invocation IR v0.2 operation set."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from vlaforge.ir.program import Block, Operation, Value
from vlaforge.ir.types import (
    CommittedOutputGroupType,
    InputRevisionType,
    IRType,
    PendingOutputGroupType,
    PendingOutputType,
    PendingType,
    ScalarType,
    SnapshotType,
    TransactionType,
)


def op(
    opcode: str,
    *,
    results: Iterable[Value] = (),
    operands: Iterable[str] = (),
    attributes: Mapping[str, Any] | None = None,
    regions: Iterable[Block] = (),
    location: str | None = None,
) -> Operation:
    return Operation(
        opcode,
        tuple(results),
        tuple(operands),
        attributes or {},
        tuple(regions),
        location,
    )


def input_read(
    value_name: str,
    revision_name: str,
    payload: IRType,
    port: str,
) -> Operation:
    return op(
        "vla.input.read",
        results=(
            Value(value_name, payload),
            Value(revision_name, InputRevisionType()),
        ),
        attributes={"input": port},
    )


def transaction_begin(name: str) -> Operation:
    return op(
        "vla.txn.begin",
        results=(Value(name, TransactionType()),),
    )


def state_read_latest(
    name: str,
    payload: IRType,
    state: str,
    transaction: str,
) -> Operation:
    return op(
        "vla.state.read_latest",
        results=(Value(name, SnapshotType(state, payload)),),
        operands=(transaction,),
        attributes={"state": state},
    )


def snapshot_value(
    name: str,
    payload: IRType,
    snapshot: str,
) -> Operation:
    return op(
        "vla.snapshot.value",
        results=(Value(name, payload),),
        operands=(snapshot,),
    )


def invoke(
    result_names: Iterable[str],
    result_types: Iterable[IRType],
    region: str,
    operands: Iterable[str],
) -> Operation:
    results = tuple(
        Value(name, result_type)
        for name, result_type in zip(result_names, result_types, strict=True)
    )
    return op(
        "vla.invoke",
        results=results,
        operands=operands,
        attributes={"region": region},
    )


def for_loop(
    result: Value,
    initial: str,
    induction: Value,
    iter_arg: Value,
    body: Block,
    *,
    lower: int,
    upper: int,
    step: int = 1,
) -> Operation:
    return op(
        "vla.for",
        results=(result,),
        operands=(initial,),
        attributes={
            "induction": induction.name,
            "iter_arg": iter_arg.name,
            "lower": lower,
            "upper": upper,
            "step": step,
        },
        regions=(Block((induction, iter_arg), body.operations),),
    )


def if_op(
    results: Iterable[Value],
    condition: str,
    then_block: Block,
    else_block: Block,
) -> Operation:
    return op(
        "vla.if",
        results=results,
        operands=(condition,),
        regions=(then_block, else_block),
    )


def yield_values(*operands: str) -> Operation:
    return op("vla.yield", operands=operands)


def return_values(*operands: str) -> Operation:
    return op("vla.return", operands=operands)


def stage_write(
    pending_name: str,
    payload: IRType,
    state: str,
    transaction: str,
    value: str,
) -> Operation:
    return op(
        "vla.state.stage_write",
        results=(Value(pending_name, PendingType(state, payload)),),
        operands=(transaction, value),
        attributes={"state": state},
    )


def validate(name: str, value: str, contract: str) -> Operation:
    return op(
        "vla.validate",
        results=(Value(name, ScalarType("bool")),),
        operands=(value,),
        attributes={"contract": contract},
    )


def output_create(
    name: str,
    value: str,
    payload: IRType,
    output: str,
) -> Operation:
    return op(
        "vla.output.create",
        results=(Value(name, PendingOutputType(output, payload)),),
        operands=(value,),
        attributes={"output": output},
    )


def output_group(
    name: str,
    group: str,
    outputs: Iterable[tuple[str, PendingOutputType]],
) -> Operation:
    items = tuple(outputs)
    return op(
        "vla.output.group",
        results=(
            Value(
                name,
                PendingOutputGroupType(
                    group,
                    tuple(item[1] for item in items),
                ),
            ),
        ),
        operands=tuple(item[0] for item in items),
        attributes={"group": group},
    )


def transaction_commit(
    committed_name: str,
    output_types: Iterable[PendingOutputType],
    group: str,
    transaction: str,
    output_group: str,
    condition: str,
) -> Operation:
    outputs = tuple(output_types)
    return op(
        "vla.txn.commit",
        results=(
            Value(
                committed_name,
                CommittedOutputGroupType(group, outputs),
            ),
        ),
        operands=(transaction, output_group, condition),
    )


def transaction_abort(transaction: str, *, reason: str = "") -> Operation:
    return op(
        "vla.txn.abort",
        operands=(transaction,),
        attributes={"reason": reason},
    )
